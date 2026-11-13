# ==================== SwanLab配置 ====================
# SwanLab API令牌，用于身份验证
from typing import Any
import os
import swanlab
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments
from evalscope.utils.logger import get_logger

swanlab_token='YOUR_SWANLAB_TOKEN'
# SwanLab项目名称
swanlab_project='test'
# SwanLab工作空间名称
swanlab_workspace='demo'
# SwanLab实验名称
swanlab_exp_name='qwen3-0.6B-perf'
# SwanLab模式：'cloud'云端模式，'local'本地模式
swanlab_mode='cloud'
# 飞书Webhook URL，用于通知
swanlab_lark_webhook_url='YOUR_LARK_WEBHOOK_URL'
# 飞书Webhook密钥
swanlab_lark_secret='YOUR_LARK_SECRET'

os.environ['DASHSCOPE_API_KEY'] = "YOUR_API_KEY"
API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions' # 'http://localhost:18008/v1' #
MODEL_ID = 'qwen-plus' #"Qwen3-0.6B" #


logger = get_logger(__name__)

# 初始化SwanLab
swanlab.init(
    experiment_name=swanlab_exp_name,
    project=swanlab_project,
    workspace=swanlab_workspace,
    mode=swanlab_mode,
    token=swanlab_token,
    lark_webhook_url=swanlab_lark_webhook_url,
    lark_webhook_secret=swanlab_lark_secret,
    resume=True,
    id='mibswwsj8wkrpwjffbr5s'
)

print(swanlab._experiment)


task_cfg = Arguments(
    parallel=[1, 2],
    number=[2, 5],
    model=MODEL_ID,
    url=API_URL,
    api='openai',
    api_key='YOUR_API_KEY',
    dataset='random',
    min_tokens=1024,
    max_tokens=1024,
    prefix_length=0,
    min_prompt_length=1024,
    max_prompt_length=1024,
    tokenizer_path='/workspace/llm/models/Qwen/Qwen3-0.6B',
    extra_args={'ignore_eos': True},
    swanlab_api_key=swanlab_token,
    # name='qwen3-0.6B-perf',
    outputs_dir='./outputs/evaloutput/perf',
)

# 记录任务配置到SwanLab
swanlab.config.update(task_cfg.__dict__)


results = run_perf_benchmark(task_cfg)

print('results:\n',results)

# 将评估结果记录到SwanLab
if results:
    # 解析终端输出的results格式
    # results是一个包含多个元组的列表，每个元组包含两个字典
    concurrency_data = []  # 存储并发数数据用于图表展示
    throughput_data = []   # 存储吞吐量数据用于图表展示
    latency_data = []      # 存储延迟数据用于图表展示
    
    for i, result_tuple in enumerate(results):
        if len(result_tuple) >= 2:
            summary_data = result_tuple[0]  # 摘要数据
            detailed_data = result_tuple[1]  # 详细数据（百分位数等）
            
            # 记录摘要数据
            for key, value in summary_data.items():
                # 将键名转换为更清晰的格式
                clean_key = key.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_per_')
                swanlab.log({f"perf_test_{i}_{clean_key}": value})
                
                # 收集用于图表的数据
                if 'Number of concurrency' in key:
                    concurrency_data.append(value)
                elif 'Output token throughput' in key:
                    throughput_data.append(value)
                elif 'Average latency' in key:
                    latency_data.append(value)
            
            # 记录详细数据（百分位数等）
            if 'Percentiles' in detailed_data:
                percentiles = detailed_data['Percentiles']
                # 记录各个指标的百分位数数据
                for metric_name in ['TTFT', 'ITL', 'TPOT', 'Latency', 'Output', 'Total']:
                    if f"{metric_name} (s)" in detailed_data or metric_name in detailed_data or f"{metric_name} (tok/s)" in detailed_data:
                        metric_key = f"{metric_name} (s)" if f"{metric_name} (s)" in detailed_data else metric_name
                        metric_key = f"{metric_name} (tok/s)" if f"{metric_name} (tok/s)" in detailed_data else metric_key
                        if metric_key in detailed_data:
                            values = detailed_data[metric_key]
                            for j, percentile in enumerate(percentiles):
                                if j < len(values):
                                    swanlab.log({
                                        f"perf_test_{i}_{metric_name}_percentile_{percentile}": values[j]
                                    })
            
            # 创建表格来展示详细数据
            # 使用echarts表格替代swanlab.Table
            table_columns = ["Metric", "10%", "25%", "50%", "66%", "75%", "80%", "90%", "95%", "98%", "99%"]
            table_data = []
            
            # 填充表格数据
            for metric_name in ['TTFT_s', 'ITL_s', 'TPOT_s', 'Latency_s', 'Output_tok_s', 'Total_tok_s']:
                clean_metric_name = metric_name.replace('_s', ' (s)').replace('_tok_s', ' (tok/s)')
                if clean_metric_name in detailed_data:
                    values = detailed_data[clean_metric_name]
                    if len(values) >= 10:
                        row_data = [
                            metric_name.replace('_s', '').replace('_tok_s', ' (tok/s)'),
                            values[0], values[1], values[2], values[3], values[4],
                            values[5], values[6], values[7], values[8], values[9]
                        ]
                        table_data.append(row_data)
            
            # 使用echarts表格记录到SwanLab
            if table_data:
                table = swanlab.echarts.Table()
                table.add(table_columns, table_data)
                swanlab.log({f"perf_test_{i}_detailed_percentiles": table})
    
    # 创建并发数与吞吐量对比图表
    if len(concurrency_data) == len(throughput_data) and len(concurrency_data) > 0:
        # 使用echarts折线图替代swanlab.Table
        line = swanlab.echarts.Line()
        line.add_xaxis(concurrency_data)
        line.add_yaxis("Throughput (tok/s)", throughput_data)
        swanlab.log({"concurrency_vs_throughput": line})
    
    # 创建并发数与平均延迟对比图表
    if len(concurrency_data) == len(latency_data) and len(concurrency_data) > 0:
        # 使用echarts折线图替代swanlab.Table
        line = swanlab.echarts.Line()
        line.add_xaxis(concurrency_data)
        line.add_yaxis("Average Latency (s)", latency_data)
        swanlab.log({"concurrency_vs_latency": line})

# 记录输出目录
output_dir_text = swanlab.Text(task_cfg.work_dir)
swanlab.log({"output_directory": output_dir_text})


# 记录错误信息到SwanLab
logger.error(f"Error occurred during performance benchmark: {str(e)}")
swanlab.log({"error": str(e)})
raise  # 重新抛出异常，让上层调用者决定如何处理

# 确保SwanLab总是被关闭
swanlab.finish()

