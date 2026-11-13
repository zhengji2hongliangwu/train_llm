# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional
import json  # 需要添加json模块

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


import os


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
        
    },
    'initialize_1013': {
        'dataset_path': '/workspace/llm/dataset/self_build/origin_data/20251013/custom_dataset_1013/dataset_output/initialize_1013.jsonl',
        # 'enable_channel_loss': True,
        'preprocess_func': ToolCallPreprocessor_Custom(),
    },
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



