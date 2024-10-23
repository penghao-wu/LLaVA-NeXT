export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

export WANDB_API_KEY="618eb3b78242f01000855a123d29e2ac98a60f30" &&
export WANDB_PROJECT="compressv" &&

LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

DATA_PATH="/mnt/sfs-common/krhu/penghao_workspace/data/jsons/share-captioner_coco_lcs_sam_1246k_1107.json"
IMAGE_FOLDER="/mnt/sfs-common/krhu/penghao_workspace/data"

NUM_GPUS=4
NNODES=1
RANK=0
ADDR="127.0.0.1"
PORT=29700

############### Pretrain ################

PROMPT_VERSION=qwen_1_5

# BASE_RUN_NAME="compressv_qwen05b_CLIP_mlp_sepsa_prevalue_2scalecat_oproj448_layer0_shareGPT4V_square_pretrain_stage2joint_GPU"
BASE_RUN_NAME="compressv_qwen05b_CLIP_mlp_2scaleold_dim448_layer10_sepnorm_shareGPT4V_pad_pretrain_GPU_lr1e3"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
HF_MODEL_ID="llava15-pretrain"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter,mm_vision_mlp" \
    --mm_vision_mlp_lr 1e-3 \
    --mm_vision_select_layer -2 \
    --max_num_image_crops 1 \
    --per_crop_token_len 576 \
    --compress_reduce_factor 4 \
    --compress_v True \
    --compress_v_start_layer 10 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 300 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \

