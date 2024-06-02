# nohup sh scripts/dpo/vicuna_7b.sh >> LLaVA-7b-v1.5-anyreslocal-144-smr-sft95-seed3407.log 2>&1 &

DATA_DIR="/mnt/workspace/xue.w/yf/data"
MODEL_DIR="/mnt/data/xue.w/yf/checkpoint"
padding=anyres 
MODEL_NAME=SliME-vicuna-13b

PROJECTOR_DIR=$MODEL_DIR/$MODEL_NAME-pt

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path finetune \
    --image_folder $DATA_DIR \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $PROJECTOR_DIR/mm_projector.bin \
    --pretrain_mm_re_sampler $PROJECTOR_DIR/sampler.bin \
    --mm_projector_type gated \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --task SFT \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $MODEL_DIR/$MODEL_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --mm_patch_merge_type spatial \
    --image_aspect_ratio $padding \
    --mm_resampler_type cosine \
    --mm_resampler_topp 0.95 \
    --mm_resampler_dim 144 \
    --mm_resampler_temp 1.0 \
   