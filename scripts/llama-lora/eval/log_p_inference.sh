#!/bin/bash
# nohup sh scripts/dpo/7b_full.sh >> 7b_full_only_sft_10k.log 2>&1 &

MODEL_DIR="/mnt/data/xue.w/yf/checkpoint"
REF_NAME=13b_rlhfv #7b_self_35kvl_feed_
MODEL_NAME=test

# --dataset_name "RLHF-V-Dataset" \
# --save_steps 40 \
# --max_steps 320 \
# --save_strategy "steps" \ liuhaotian/llava-v1.5-13b
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --ref_names $REF_NAME \
    --version v1 \
    --task DPO \
    --image_folder /mnt/workspace/xue.w/yf/data/ \
    --data_path data/llava_7b_v1_preference.json \
    --image_to_caption_file data/image_to_caption.json \
    --dataset_name playground/Preference_data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --num_train_epochs 5\
    --save_strategy "steps" \
    --save_steps 40 \
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
    --report_to none \
    --dpo_use_average True \
    --dpo_token_weighted False \
    --dpo_token_weight 1.1 \
    --dpo_beta 0.1 \
    --loss_type sigmoid