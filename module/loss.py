# Date: 2018.10.28
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def cross_entropy_loss(logit, label):
    """
    get cross entropy loss
    Args:
        logit: logit
        label: true label

    Returns:

    """
    criterion = nn.CrossEntropyLoss().cuda()
    loss = criterion(logit, label)
    return loss


class InverseWeightCrossEntroyLoss(nn.Module):
    def __init__(self,
                 class_num,
                 ignore_index=255
                 ):
        super(InverseWeightCrossEntroyLoss, self).__init__()
        self.class_num = class_num
        self.ignore_index=ignore_index

    def forward(self, logit, label):
        """
       get inverse cross entropy loss
        Args:
            logit: a tensor, [batch_size, num_class, image_size, image_size]
            label: a tensor, [batch_size, image_size, image_size]
        Returns:

        """
        inverse_weight = self.get_inverse_weight(label)
        cross_entropy = nn.CrossEntropyLoss(weight=inverse_weight,
                                            ignore_index=self.ignore_index).cuda()
        inv_w_loss = cross_entropy(logit, label)
        return inv_w_loss

    def get_inverse_weight(self, label):
        mask = (label >= 0) & (label < self.class_num)
        label = label[mask]
        # reduce dim
        total_num = len(label)
        # get unique label, convert unique label to list
        percentage = torch.bincount(label, minlength=self.class_num) / float(total_num)
        # get inverse
        w_for_each_class = 1 / torch.log(1.02 + percentage)
        # convert to tensor
        return w_for_each_class.float()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, ignore_index=255, reduction=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        
        alpha = torch.from_numpy(self.alpha).to(y_pred.device).view(1, -1, 1, 1)

        p = torch.softmax(y_pred, dim=1)

        ignore_mask = (y_true == self.ignore_index)

        # one hot encoding
        y_index = torch.clone(y_true).to(y_pred.device)
        y_index[ignore_mask] = 0
        one_hot_y_true = torch.zeros(y_pred.shape, dtype=torch.float).to(y_pred.device)
        one_hot_y_true.scatter_(1, y_index.unsqueeze_(dim=1).long(), torch.ones(one_hot_y_true.shape).to(y_pred.device))

        pt = (p * one_hot_y_true).sum(dim=1)
        modular_factor = (1 - pt).pow(self.gamma)
        
        cls_balance_factor = (alpha.float() * one_hot_y_true.float()).sum(dim=1)
        modular_factor.mul_(cls_balance_factor)

        losses = F.cross_entropy(y_pred, y_true, ignore_index=self.ignore_index, reduction='none')
        losses.mul_(modular_factor)

        if self.reduction:
            valid_mask = (y_true != self.ignore_index).float()
            mean_loss = losses.sum() / valid_mask.sum()
            return mean_loss
        return losses



class DiceLoss(nn.Module):
    def __init__(self,
                 smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def _dice_coeff(self, pred, target):
        """
        Args:
            pred: [N, 1] within [0, 1]
            target: [N, 1]
        Returns:
        """

        smooth = self.smooth
        inter = torch.sum(pred * target)
        z = pred.sum() + target.sum() + smooth
        return (2 * inter + smooth) / z

    def forward(self, pred, target):
        return 1. - self._dice_coeff(pred, target)



def som(loss, ratio):
    # 1. keep num
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)
    # 2. select loss
    top_loss, _ = loss.reshape(-1).topk(num_hns, -1)
    loss_mask = (top_loss != 0)
    # 3. mean loss

    return torch.sum(top_loss[loss_mask]) / (loss_mask.sum() + 1e-6)



class JointLoss(nn.Module):
    def __init__(self, ignore_index=255, sample='SOM', ratio=0.2):
        super(JointLoss, self).__init__()
        assert sample in ['SOM', 'OHEM']
        self.ignore_index = ignore_index
        self.sample = sample
        self.ratio = ratio
        print('Sample:', sample)
       

    def forward(self, cls_pred, binary_pred, cls_true, instance_mask=None):
        valid_mask = (cls_true != self.ignore_index)
        fgp = torch.sigmoid(binary_pred)
        clsp = torch.softmax(cls_pred, dim=1)
        # numerator
        joint_prob = torch.clone(clsp)
        joint_prob[:, 0, :, :] = (1-fgp).squeeze(dim=1) * clsp[:, 0, :, :]
        joint_prob[:, 1:, :, :] = fgp * clsp[:, 1:, :, :]
        # # normalization factor, [B x 1 X H X W]
        Z = torch.sum(joint_prob, dim=1, keepdim=True)
        # cls prob, [B, N, H, W]
        p_ci = joint_prob / Z

        losses = F.nll_loss(torch.log(p_ci), cls_true.long(), ignore_index=self.ignore_index, reduction='none')
        
        if self.sample == 'SOM':
            return som(losses, self.ratio)
        elif self.sample == 'OHEM':
            seg_weight = ohem_weight(p_ci, cls_true.long(), thresh=self.ratio)
            return (seg_weight * losses).sum() / seg_weight.sum()
        else:
            return losses.sum() / valid_mask.sum()







def ohem_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor,
                       ignore_index: int = -1,
                       thresh: float = 0.7,
                       min_kept: int = 100000):
    # y_pred: [N, C, H, W]
    # y_true: [N, H, W]
    # seg_weight: [N, H, W]
    y_true = y_true.unsqueeze(1)
    with torch.no_grad():
        assert y_pred.shape[2:] == y_true.shape[2:]
        assert y_true.shape[1] == 1
        seg_label = y_true.squeeze(1).long()
        batch_kept = min_kept * seg_label.size(0)
        valid_mask = seg_label != ignore_index
        seg_weight = y_pred.new_zeros(size=seg_label.size())
        valid_seg_weight = seg_weight[valid_mask]

        seg_prob = F.softmax(y_pred, dim=1)

        tmp_seg_label = seg_label.clone().unsqueeze(1)
        tmp_seg_label[tmp_seg_label == ignore_index] = 0
        seg_prob = seg_prob.gather(1, tmp_seg_label).squeeze(1)
        sort_prob, sort_indices = seg_prob[valid_mask].sort()

        if sort_prob.numel() > 0:
            min_threshold = sort_prob[min(batch_kept,
                                          sort_prob.numel() - 1)]
        else:
            min_threshold = 0.0
        threshold = max(min_threshold, thresh)
        valid_seg_weight[seg_prob[valid_mask] < threshold] = 1.

    seg_weight[valid_mask] = valid_seg_weight

    losses = F.cross_entropy(y_pred, y_true.squeeze(1), ignore_index=ignore_index, reduction='none')
    losses = losses * seg_weight

    return losses.sum() / seg_weight.sum()

if __name__ == '__main__':
    torch.random.manual_seed(233)
    cls_pred = torch.randn(2, 5, 4, 4)
    binary_pred = torch.randn(2, 1, 4, 4)
    cls_true = torch.ones(2, 4, 4)
    jloss = JointLoss()
    l = jloss(cls_pred, binary_pred, cls_true)
    print(l)
    # y_true = torch.tensor([1, 1, 0]).float()
    # y_pred = torch.tensor([np.nan, 0, 0.2])
    # # l = F.binary_cross_entropy(y_pred, y_true.float(), reduction='none')
    # # l_m = F.cross_entropy(y_pred, y_true.long(), reduction='none')
    # # print(l)
    # # print(l_m)
    #
    # loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    # print(loss)
    # bceloss = BceWithLogitsLoss()
    # y_pred = torch.tensor([2, 3, 1, 2]).float()
    # y_true = torch.tensor([0, 1, 1, 255]).float()
    # loss = bceloss(torch.clone(y_pred), torch.clone(y_true))
    # print(loss)
    #
    # binary_true = y_true.clone()
    # binary_true[(y_true > 0) * (y_true < 5)] = 1
    # mask = torch.ones_like(y_true).float()
    # mask[y_true == 255] = 0
    # binary_true[y_true == 255] = 0
    # binary_losses = F.binary_cross_entropy_with_logits(y_pred, binary_true.float(), weight=mask, reduction='none')
    # print(binary_losses.sum() / mask.sum())


    pass