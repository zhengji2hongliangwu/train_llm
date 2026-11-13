# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional
import json  # 需要添加json模块

from swift.llm import DatasetMeta,AutoPreprocessor, ResponsePreprocessor,RowPreprocessor, load_dataset, register_dataset
from swift.llm import get_model_tokenizer, get_template
from swift.llm import sft_main
from swift.llm.train.sft import SwiftSft
import os
# from custom.dataset.utils.preprocessor import ToolCallPreprocessor_CustomQwen3 as ToolCallPreprocessor


class CustomPreprocessor(ResponsePreprocessor):
    """处理工具调用数据的预处理器"""
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 主要的数据处理逻辑
        '''
        将tool_call转换为messages格式拼接在assistant的content中
        '''
        output = {"messages": [], "tools": []}
        output["tools"] = row.get("tools", [])
        if output["tools"]:
            output["channel"] = "tool"
        else:
            output["channel"] = ""
        messages = []
        
        for i, message in enumerate(row['messages']):
            # 添加基础消息
            messages_now = message
            # messages.append(messages_now)
            # 处理assistant的工具调用
            if message['role'] == 'assistant':
                if message.get('tool_calls', None) and len(message['tool_calls']) > 0:
                    for tool_call in message['tool_calls']:
                        messages_now['content'] += "\n<tool_call>\n" + json.dumps({
                            'name': tool_call['function']['name'],
                            'arguments': json.loads(tool_call['function']['arguments'])
                        },ensure_ascii=False)+"\n</tool_call>"
                    if not output["channel"]:
                        output["channel"] = "tool"  # 可以兼容channel loss技术
                else:
                    if not output["channel"]:
                        output["channel"] = "chat"  # 可以兼容channel loss技术
            if messages_now['role'] in ['tool','tool_response','function_response','observation']:
                messages_now['role'] = 'tool_response'  # 可以兼容channel loss技术
            messages.append(messages_now)
        output["messages"] = messages
        return output


class CustomAutoPreprocessor(RowPreprocessor):
    """
    一个自动预处理器，能够根据输入的预处理器类型（CustomPreprocessor 或 AutoPreprocessor）
    来处理数据，并确保输出中包含 channel 字段。
    """
    def __init__(
        self,
        Preprocessor: Optional[RowPreprocessor] = None,
        dataset_sample: Optional[int] = None,
        channel: str = "chat",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_sample = dataset_sample
        self.channel = channel
        self.preprocessor = None

        if Preprocessor is not None:
            # 实例化传入的预处理器
            if Preprocessor == AutoPreprocessor:
                # 对于 AutoPreprocessor，我们需要先创建实例，然后获取其内部的预处理器
                self.auto_preprocessor = AutoPreprocessor(dataset_sample=dataset_sample, **kwargs)
            else:
                # 对于 CustomPreprocessor 等直接继承自 RowPreprocessor 的类
                self.preprocessor = Preprocessor(dataset_sample=dataset_sample, **kwargs)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理单条数据。
        """
        output = None

        if self.preprocessor is not None:
            # 如果是 CustomPreprocessor 等，直接调用其 preprocess 方法
            output = self.preprocessor.preprocess(row)
        elif hasattr(self, 'auto_preprocessor'):
            # 如果是 AutoPreprocessor，我们需要模拟一个数据集来获取其内部的预处理器
            # 这是一个简化的处理方式，实际使用时可能需要根据具体情况调整
            dummy_dataset = Dataset.from_dict({'dummy': [row]})
            processed_dataset = self.auto_preprocessor(dummy_dataset)
            if len(processed_dataset) > 0:
                output = processed_dataset[0]

        if output is None:
            return None

        # 确保 channel 字段存在
        if 'channel' not in output or not output.get('channel', ""):
            output['channel'] = self.channel
        return output

    def __call__(
        self,
        dataset,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        strict: bool = False,
    ):
        """
        处理整个数据集。
        这个方法使得 CustomAutoPreprocessor 也可以像 AutoPreprocessor 一样处理整个数据集。
        """
        if hasattr(self, 'auto_preprocessor'):
            # 如果是 AutoPreprocessor，直接调用其 __call__ 方法
            dataset = self.auto_preprocessor(
                dataset,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                strict=strict
            )
        elif self.preprocessor is not None and hasattr(self.preprocessor, '__call__'):
            # 如果预处理器本身支持处理整个数据集，就调用它的 __call__ 方法
            dataset = self.preprocessor(
                dataset,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                strict=strict
            )
        else:
            # 否则，使用 RowPreprocessor 的默认方式处理整个数据集
            dataset = super().__call__(
                dataset,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                strict=strict
            )

        # 确保所有样本都有 channel 字段
        def add_channel(row):
            if 'channel' not in row or not row['channel']:
                row['channel'] = self.channel
            return row

        dataset = dataset.map(add_channel, num_proc=num_proc, load_from_cache_file=load_from_cache_file)

        return dataset


    
def register_datasets(datasets_list:Dict[str,Dict[str,Any]]):
    
    for dataset_name, dataset in datasets_list.items():
        if dataset.get('preprocess') == 'auto':
            preprocess_func = CustomAutoPreprocessor(Preprocessor=AutoPreprocessor,dataset_sample=dataset.get('dataset_sample',None),channel=dataset.get('channel','chat'))
        elif dataset.get('preprocess') == 'custom':
            preprocess_func = CustomAutoPreprocessor(Preprocessor=CustomPreprocessor,dataset_sample=dataset.get('dataset_sample',None),channel=dataset.get('channel','chat'))
        else:
            preprocess_func = None
        register_dataset(
            DatasetMeta(
                dataset_path=dataset['dataset_path'],
                dataset_name=dataset_name,
                preprocess_func=preprocess_func,
                tags=dataset.get('channel',[]),
            ))

if __name__ == '__main__':
    datasets_dict = json.load(open('config_traing_data.json'))
    dataseteval_dict = json.load(open('config_test_data.json'))

    print(datasets_dict)
    dataset_names = list(datasets_dict.keys())
    dataseteval_names = list(dataseteval_dict.keys())

    # 测试数据集加载，load_dataset默认兼容标准的alpaca，sharegpt，以及openai messages格式    dataset = load_dataset(dataset_names)[0]#,remove_unused_columns=False, enable_channel_loss=True
    register_datasets(datasets_dict)
    register_datasets(dataseteval_dict)


    from swift.llm import sft_main, TrainArguments  # , SwiftSft
    from typing import Optional, Union, List


    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    
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
        num_train_epochs=5,
        # 每个设备的训练批次大小，根据GPU内存调整
        per_device_train_batch_size=4,
        # 每个设备的评估批次大小
        per_device_eval_batch_size=4,
        # 学习率，控制模型参数更新的步长
        learning_rate=1e-5,
        # 梯度累积步数，模拟更大批次训练，解决显存不足问题
        gradient_accumulation_steps=32,

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
        output_dir='outputs/training/run_20251118',
        
        # ==================== 数据加载配置 ====================
        # 数据加载器的工作进程数
        dataloader_num_workers=4,
        # 输入序列的最大长度，超过会被截断
        max_length=4096,
        
        # ==================== 评估配置 ====================
        # 评估策略：'steps'按步数评估，'epoch'按轮次评估
        eval_strategy='steps',
        # 评估步数间隔，每多少步评估一次
        eval_steps=10,
        # 是否使用评估作用域
        eval_use_evalscope=True,
        # 评估数据集名称列表
        eval_dataset=dataseteval_names,
        # # 评估时使用的样本数量限制
        eval_limit=200,
        # 额外的评估参数，用于配置实验跟踪工具
        # extra_eval_args={'swanlab': {'run_name': 'qwen3-0.6B-tool-11-18'}},
        
        # ==================== 自定义注册路径 ====================
        # 自定义数据集注册文件的路径
        # custom_register_path='examples/custom_dataset/dataset.py',
        
        # ==================== SwanLab配置 ====================
        # SwanLab API令牌，用于身份验证
        swanlab_token='YOUR_SWANLAB_TOKEN',
        # SwanLab项目名称
        swanlab_project='demo_project',
        # SwanLab工作空间名称
        swanlab_workspace='demo',
        # SwanLab实验名称
        swanlab_exp_name='qwen3-0.6B-1125',
        # SwanLab模式：'cloud'云端模式，'local'本地模式
        swanlab_mode='cloud',
        # 飞书Webhook URL，用于通知
        swanlab_lark_webhook_url='YOUR_LARK_WEBHOOK_URL',
        # 飞书Webhook密钥
        swanlab_lark_secret='YOUR_LARK_SECRET',
        
        # ==================== 其他配置 ====================
        # 运行名称，用于标识本次训练
        run_name='qwen3-0.6B-1125',
        # 报告到的工具，用于实验跟踪和日志记录
        report_to='swanlab'
    )
    
    sft = SwiftSft(args)
    result = sft.main()
    
    # # 新版本的调用方式，直接使用sft_main函数
    # result = sft_main(args)
    print(result)


