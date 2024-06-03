
DATA_DIR=/mnt/data/xue.w/yf/data/llava/llava_pretrain/images
MODEL_DIR="/mnt/data/xue.w/yf/checkpoint"
padding=anyres
MODEL_NAME=SliME-Llama-3-8B-lora

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --version plain \
    --data_path pretrain \
    --image_folder $DATA_DIR \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type gated \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $MODEL_DIR/$MODEL_NAME-proj \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
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
    --mm_resampler_topp 1.0 \
    --mm_resampler_dim 144 \
    --mm_resampler_temp 1.0 \
    --use_global_only True \
    --mm_learnable_gated 0 \

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --pretrain_mm_mlp_adapter $MODEL_DIR/$MODEL_NAME-proj/mm_projector.bin \
    --version plain \
    --data_path pretrain \
    --image_folder $DATA_DIR \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type gated \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $MODEL_DIR/$MODEL_NAME-atten \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
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
    --mm_resampler_topp 1.0 \
    --mm_resampler_dim 144 \
    --mm_resampler_temp 1.0 \
    --use_global_only True \
    --mm_learnable_gated 1 \


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --version plain \
    --data_path pretrain \
    --image_folder $DATA_DIR \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $MODEL_DIR/$MODEL_NAME-atten/mm_projector.bin \
    --mm_projector_type gated \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $MODEL_DIR/$MODEL_NAME-pt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
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
    --mm_resampler_topp 1.0 \
    --mm_resampler_dim 144 \
    --mm_resampler_temp 1.0 \
    --use_local_only True