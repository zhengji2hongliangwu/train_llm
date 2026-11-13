VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=6 vllm serve /workspace/llm/models/Qwen/Qwen3-0.6B \
--host 0.0.0.0 \
--port 19008 \
--tensor-parallel-size 1 \
--seed 1024 \
--served-model-name Qwen3-0.6B \
--max-model-len 20000 \
--max-num-batched-tokens 20000 \
--max-num-seqs 16 \
--trust-remote-code \
--gpu-memory-utilization 0.25 \
--enable-auto-tool-choice \
--tool-call-parser hermes \
--disable-log-request \
--reasoning-parser qwen3 \
--kv-cache-dtype auto > ./logs/qwen3_0.6b_vllm_base.log # 2>&1 &
    