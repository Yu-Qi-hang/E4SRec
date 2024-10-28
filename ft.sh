export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=$1,$2 torchrun --nproc_per_node=2 --master_port=1234 finetune.py \
    --base_model llama-2-7b-hf \
    --data_path Beauty \
    --task_type sequential \
    --cache_dir cache_dir/ \
    --output_dir output_dir/ \
    --batch_size 16 \
    --micro_batch_size 1 \
    --num_epochs 3 \
    --learning_rate 0.0003 \
    --cutoff_len 4096 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --warmup_steps 100