export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='isaid.factseg'
model_dir='./log'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9992 apex_train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    --opt_level='O1'

ckpt_path=${model_dir}'/model-60000.pth'
image_dir='./isaid_segm/val/images'
mask_dir='./isaid_segm/val/masks'

vis_dir=${model_dir}'/vis-60000'

python ./tools/isaid_eval.py \
    --config_path=${config_path} \
    --ckpt_path=${ckpt_path} \
    --image_dir=${image_dir} \
    --mask_dir=${mask_dir} \
    --vis_dir=${vis_dir} \
    --log_dir=${model_dir} \
    --patch_size=896

