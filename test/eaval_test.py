from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='deepseek-v3.2-exp', #'qwen-plus',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key='YOUR_API_KEY',
    eval_type='server',
    datasets=['tool_bench'],
    limit=5,
    eval_batch_size=5,
    generation_config={
        'max_tokens': 1000,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.7,  # 采样温度 (qwen 报告推荐值)
        'top_p': 0.8,  # top-p采样 (qwen 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen 报告推荐值)
        'extra_body':{'chat_template_kwargs': {'enable_thinking': False}}  # 关闭思考模式
    }
)
run_task(task_cfg=task_cfg)