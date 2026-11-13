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

datasets_dict = json.load(open('config_traing_data.json'))
dataseteval_dict = json.load(open('config_test_data.json'))

# print(datasets_dict)
dataset_names = list(datasets_dict.keys())
dataseteval_names = list(dataseteval_dict.keys())
print(f'dataset_names: {" ".join(dataset_names)}')
print(f'dataseteval_names: {" ".join(dataseteval_names)}')
# 测试数据集加载，load_dataset默认兼容标准的alpaca，sharegpt，以及openai messages格式    dataset = load_dataset(dataset_names)[0]#,remove_unused_columns=False, enable_channel_loss=True
register_datasets(datasets_dict)
register_datasets(dataseteval_dict)

if __name__ == '__main__':
    dataset = load_dataset(dataseteval_names)[0]
    print(f'dataset: {dataset}')
    # print(f'dataset[0]: {dataset[0]}')
