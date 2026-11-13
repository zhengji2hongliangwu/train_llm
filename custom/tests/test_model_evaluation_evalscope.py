# ==================== SwanLab配置 ====================
# SwanLab API令牌，用于身份验证
swanlab_token='YOUR_SWANLAB_TOKEN'
# SwanLab项目名称
swanlab_project='demo_project'
# SwanLab工作空间名称
swanlab_workspace='demo'
# SwanLab实验名称
swanlab_exp_name='qwen3-0.6B-eval'
# SwanLab模式：'cloud'云端模式，'local'本地模式
swanlab_mode='cloud'
# 飞书Webhook URL，用于通知
swanlab_lark_webhook_url='YOUR_LARK_WEBHOOK_URL'
# 飞书Webhook密钥
swanlab_lark_secret='YOUR_LARK_SECRET'

import os
import swanlab
from evalscope import TaskConfig, run_task
# from evalscope.constants import JudgeStrategy
os.environ['DASHSCOPE_API_KEY'] = "YOUR_API_KEY"
API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1' # 'http://localhost:18008/v1' #
MODEL_ID = 'qwen-plus' #"Qwen3-0.6B" #
from evalscope.summarizer import Summarizer

from evalscope.utils.logger import get_logger

logger = get_logger(__name__)

# 初始化SwanLab
swanlab.init(
    experiment_name=swanlab_exp_name,
    project=swanlab_project,
    workspace=swanlab_workspace,
    mode=swanlab_mode,
    token=swanlab_token,
    lark_webhook_url=swanlab_lark_webhook_url,
    lark_webhook_secret=swanlab_lark_secret
)

# 创建任务配置
task_cfg = TaskConfig(
    # model='Qwen/Qwen2.5-0.5B-Instruct',
    model=MODEL_ID,
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    api_url=API_URL,
    eval_type='server',
    datasets=[
        'general_qa',
    ],
    dataset_args={
        'general_qa': {
            'dataset_id': '/workspace/llm/eval/evalscope/custom_eval/text/qa',
            'subset_list': [
                'example'
            ],
            # 'filters': {'remove_until': '</think>'}  # 过滤掉思考的内容
        }
    },
    # judge 相关参数

    # # 使用更大的模型评分
    # judge_model_args={
    #     'model_id': 'qwen-plus',
    #     'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    #     'api_key': os.getenv('DASHSCOPE_API_KEY'),
    #     'generation_config': {
    #         'temperature': 0.0,
    #         'max_tokens': 4096
    #     },
    #     # 直接打分
    #     'score_type': 'numeric',
    # },
    # # judge 并发数
    # judge_worker_num=5,
    # # 使用 LLM 进行评测
    # judge_strategy=JudgeStrategy.LLM,

    work_dir='./outputs/evaloutput/general_qa/',
    limit=5,
)

# 记录任务配置到SwanLab
swanlab.config.update(task_cfg)

results = run_task(task_cfg=task_cfg)

print('results:\n',results)


report_list = Summarizer.get_report_from_cfg(task_cfg)
logger.info(f'>>The report list:\n{report_list}')

# 将评估结果记录到SwanLab
if results:
    for dataset_name, report in results.items():
        # 记录总体得分
        swanlab.log({f"{dataset_name}_score": report.score})
        # 记录各个指标的得分
        for metric in report.metrics:
            swanlab.log({f"{dataset_name}_{metric.name}": metric.score})
            # 记录各个类别的得分
            for category in metric.categories:
                for subset in category.subsets:
                    swanlab.log({f"{dataset_name}_{metric.name}_{subset.name}": subset.score})

# 记录输出目录
output_dir_text = swanlab.Text(task_cfg.work_dir)
swanlab.log({"output_directory": output_dir_text})

# 关闭SwanLab
swanlab.finish()

