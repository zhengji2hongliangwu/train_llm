# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional
import json  # 需要添加json模块

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset
from swift.llm import get_model_tokenizer, get_template
from swift.llm import sft_main
from swift.llm.train.sft import SwiftSft
import os

max_len = 10
class ToolCallPreprocessor_Custom(ResponsePreprocessor):
    """处理工具调用数据的预处理器"""
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 主要的数据处理逻辑
        '''
        将tool_call转换为messages格式拼接在assistant的content中
        '''
        output = {"messages": [], "tools": []}
        output["channel"] = row.get("channel", "chat")
        output["tools"] = row.get("tools", [])
        messages = []
        
        for i, message in enumerate(row['messages']):
            # 添加基础消息
            messages_now = {
                'role': message['role'],
                'content': message['content']
            }
            # messages.append(messages_now)
            # 处理assistant的工具调用
            if message['role'] == 'assistant':
                if message.get('tool_calls', None) and len(message['tool_calls']) > 0:
                    for tool_call in message['tool_calls']:
                        messages_now['content'] += "\n<tool_call>\n" + json.dumps({
                            'name': tool_call['function']['name'],
                            'arguments': json.loads(tool_call['function']['arguments'])
                        })+"\n</tool_call>"

                    output["channel"] = "tool"  # 可以兼容channel loss技术
            messages.append(messages_now)
            if max_len and len(messages) >= max_len:
                break
        output["messages"] = messages
        return output



def register_datasets_all(datasetload_dict: Dict[str, Dict[str, Any]]) -> None:
    for dataset_name, dataset_config in datasetload_dict.items():
        register_dataset(
            DatasetMeta(
                dataset_name=dataset_name,
                dataset_path=dataset_config['dataset_path'],
                preprocess_func=dataset_config.get('preprocess_func', None)
            )
        )

def create_datasetload_dict(dataset_folder: str, dataset_names:list|Any=None, preprocess_func=None) -> Dict[str, Dict[str, Any]]:
    '''
    datasetload_dict = {
    'gesture_1013': {
        'dataset_path': '/workspace/llm/dataset/self_build/origin_data/20251013/custom_dataset_1013/dataset_output/gesture_1013.jsonl',
        # 'enable_channel_loss': True,
        'preprocess_func': ToolCallPreprocessor_Custom(),
        
    }
    '''
    datasetload_dict = {}
    if dataset_names is None:
        dataset_names = os.listdir(dataset_folder)
        dataset_names = [dataset_name.split('.jsonl')[0] for dataset_name in dataset_names]
    for dataset_name in dataset_names:
        datasetload_dict[dataset_name] = {
            'dataset_path': f'{dataset_folder}/{dataset_name}.jsonl',
            'preprocess_func': preprocess_func,
        }
    return datasetload_dict

if __name__ == '__main__':
    datasetload_dict = create_datasetload_dict(
        dataset_folder='/workspace/llm/dataset/self_build/origin_data/20251013/custom_dataset_1013/dataset_output',
        dataset_names=None,#['gesture_1013', 'initialize_1013'],
        preprocess_func=ToolCallPreprocessor_Custom(),
    )
    print(datasetload_dict)
    dataset_names = list(datasetload_dict.keys())
    # 测试数据集加载，load_dataset默认兼容标准的alpaca，sharegpt，以及openai messages格式
    register_datasets_all(datasetload_dict)
    
    dataset = load_dataset(dataset_names)[0]#,remove_unused_columns=False, enable_channel_loss=True

    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')

    data = dataset[0]
    _, tokenizer = get_model_tokenizer('/workspace/llm/models/Qwen/Qwen3-0.6B', load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    template.set_mode('train')
    encoded = template.encode(data, return_template_inputs=True)
    print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
    print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
    print(f'images: {encoded["template_inputs"].images}')


    from swift.llm import sft_main, TrainArguments  # , SwiftSft
    from typing import Optional, Union, List

    # 设置CUDA设备
    # 指定使用哪个GPU进行训练，'7'表示使用第7号GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    
    # 创建训练参数
    args = TrainArguments(
        # ==================== 模型配置 ====================
        # 预训练模型路径或名称，指定要微调的基础模型
        model='/workspace/llm/models/Qwen/Qwen3-0.6B',
        # 训练类型：'full'表示全参数微调，'lora'表示LoRA微调，'qlora'表示量化LoRA微调
        train_type='full',
        # 训练数据集名称列表，可以注册多个数据集
        dataset=dataset_names,
        # 是否启用通道损失，用于多任务学习或特殊损失函数
        enable_channel_loss=True,
        
        # ==================== 数据集配置 ====================
        # 数据集分割比例，将训练集按此比例分割出验证集
        split_dataset_ratio=0.05,
        # 数据集预处理进程数，控制数据预处理的并行度
        dataset_num_proc=4,

        
        # ==================== 训练配置 ====================
        # 训练轮数，整个数据集训练的遍数
        num_train_epochs=3,
        # 每个设备的训练批次大小，根据GPU内存调整
        per_device_train_batch_size=4,
        # 每个设备的评估批次大小
        per_device_eval_batch_size=4,
        # 学习率，控制模型参数更新的步长
        learning_rate=1e-5,
        # 梯度累积步数，模拟更大批次训练，解决显存不足问题
        gradient_accumulation_steps=16,

        
        # ==================== 日志和保存配置 ====================
        # 保存检查点的步数间隔
        save_steps=10,
        # 最多保存的检查点数量，超出时会删除旧的
        save_total_limit=2,
        # 日志记录步数间隔
        logging_steps=5,
        # 学习率预热比例，在训练开始时线性增加学习率
        warmup_ratio=0.1,
        # 输出目录，保存训练结果和检查点
        output_dir='outputs/training/tool_training_10-14',
        
        # ==================== 数据加载配置 ====================
        # 数据加载器的工作进程数
        dataloader_num_workers=4,
        # 输入序列的最大长度，超过会被截断
        max_length=4096,
        
        # ==================== 评估配置 ====================
        # 评估策略：'steps'按步数评估，'epoch'按轮次评估
        eval_strategy='steps',
        # 评估步数间隔，每多少步评估一次
        eval_steps=2,
        # 是否使用评估作用域
        eval_use_evalscope=True,
        # 评估数据集名称列表
        eval_dataset=dataset_names,
        # 评估时使用的样本数量限制
        eval_limit=10,
        # 额外的评估参数，用于配置实验跟踪工具
        extra_eval_args={'swanlab': {'run_name': 'qwen3-0.6B-tool-10-14'}},
        
        # ==================== 自定义注册路径 ====================
        # 自定义数据集注册文件的路径
        # custom_register_path='examples/custom_dataset/dataset.py',
        
        # ==================== SwanLab配置 ====================
        # SwanLab API令牌，用于身份验证
        swanlab_token='YOUR_SWANLAB_TOKEN',
        # SwanLab项目名称
        swanlab_project='test',
        # SwanLab工作空间名称
        swanlab_workspace='demo',
        # SwanLab实验名称
        swanlab_exp_name='qwen3-0.6B',
        # SwanLab模式：'cloud'云端模式，'local'本地模式
        swanlab_mode='cloud',
        # 飞书Webhook URL，用于通知
        swanlab_lark_webhook_url='YOUR_LARK_WEBHOOK_URL',
        # 飞书Webhook密钥
        swanlab_lark_secret='YOUR_LARK_SECRET',
        
        # ==================== 其他配置 ====================
        # 运行名称，用于标识本次训练
        run_name='qwen3-0.6B',
        # 报告到的工具，用于实验跟踪和日志记录
        report_to='swanlab'
    )
    
    sft = SwiftSft(args)
    result = sft.main()
    
    # # 新版本的调用方式，直接使用sft_main函数
    # result = sft_main(args)
    print(result)


