export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0.1000

LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

DATA_PATH="/home/cirrascale/penghaowu_workspace/data/jsons/share-captioner_coco_lcs_sam_1246k_1107.json"
IMAGE_FOLDER="/home/cirrascale/penghaowu_workspace/data"

NUM_GPUS=8
NNODES=1
RANK=0
ADDR="127.0.0.1"
PORT=29500


############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llava15-fast-layer12-dim896-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_sharegpt4v_plain_bs512"
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
    --mm_vision_mlp_lr 1e-4 \
    --mm_vision_select_layer -2 \
    --fast_vision True \
    --fast_vision_start_layer 12 \
    --concise_reduce_factor 4 \
    --max_num_image_crops 1 \
    --per_crop_token_len 576 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --save_steps 1000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
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

# You can delete the sdpa attn_implementation if you want to use flash attn