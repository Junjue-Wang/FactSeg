export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='isaid.factseg'
model_dir='./log'
vis_dir='./log/vis-60000'
ckpt_path='./log/model-60000.pth'
image_dir='./isaid_segm/val/images'
mask_dir='./isaid_segm/val/masks'

python ./tools/isaid_eval.py \
    --config_path=${config_path} \
    --ckpt_path=${ckpt_path} \
    --image_dir=${image_dir} \
    --mask_dir=${mask_dir} \
    --vis_dir=${vis_dir} \
    --log_dir=${model_dir} \
    --patch_size=896