export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=$1,$2 torchrun --nproc_per_node=2 --master_port=1234 inference.py \
    --base_model llama-2-7b-hf \
    --data_path Beauty \
    --task_type sequential \
    --checkpoint_dir checkpoint_dir \
    --cache_dir cache_dir/ \
    --output_dir output_dir/ \
    --batch_size 16 \
    --micro_batch_size 1