
CUDA_VISIBLE_DEVICES=4,5 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model /workspace/llm/models/Qwen/Qwen3-0.6B \
    --dataset AI-MO/NuminaMath-TIR#100 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --beta 0.0 \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --steps_per_generation 4 \
    --importance_sampling_level sequence \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_generations 16 \
    --train_type full \
    --reward_funcs accuracy \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_max_model_len 16384 \
    --max_completion_length 8192 \
    --offload_optimizer true \
    --offload_model true \
    --sleep_level 1 \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --log_completions true
