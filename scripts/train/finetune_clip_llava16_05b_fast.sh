export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# export WANDB_RESUME="must" &&
# export WANDB_RUN_ID="wfod18nw" &&
export WANDB_API_KEY="618eb3b78242f01000855a123d29e2ac98a60f30" &&
export WANDB_PROJECT="compressv" &&


LLM_VERSION="Qwen/Qwen2-0.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

NUM_GPUS=4
NNODES=1
RANK=0
ADDR="127.0.0.1"
PORT=29800

############### Pretrain ################

DATA_PATH="/mnt/sfs-common/krhu/penghao_workspace/data/jsons/llava_next_raw_format_processed.json"
IMAGE_FOLDER="/mnt/sfs-common/krhu/penghao_workspace/data/llava_next"

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="compressv_qwen05b_CLIP_mlp_2scaleold_dim448_layer12_sepnorm_shareGPT4V_pad_pretrain_GPU_lr1e3"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

MID_RUN_NAME="compressv_anyres_qwen05b_CLIP_mlp_2scaleold_dim448_layer12_sepnorm_padPTlr1e3_finetune_738k_bs512_GPU"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --pretrain_mm_mlp_adapter="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \
    --mm_tunable_parts="mm_mlp_adapter,mm_vision_mlp,mm_language_model" \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --max_num_image_crops 5 \
    --per_crop_token_len 576 \
    --compress_reduce_factor 4 \
    --compress_v True \
    --compress_v_start_layer 12 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $MID_RUN_NAME \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn