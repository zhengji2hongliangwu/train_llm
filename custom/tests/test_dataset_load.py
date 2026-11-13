# Copyright (c) Alibaba, Inc. and its affiliates.


from typing import Any, Dict, Optional
import json  # 需要添加json模块

from swift.llm import DatasetMeta, ResponsePreprocessor, load_dataset, register_dataset


class ToolCallPreprocessor(ResponsePreprocessor):
    """处理工具调用数据的预处理器"""
    
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 主要的数据处理逻辑
        '''
        将tool_call转换为messages格式拼接在assistant的content中
        '''
        output = {"messages": [], "tools": []}
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
                        },ensure_ascii=False)+"\n</tool_call>"
                        # messages.append({
                        #     'role': 'tool_call',
                        #     'content': json.dumps({
                        #         'name': tool_call['function']['name'],
                        #         'arguments': json.loads(tool_call['function']['arguments'])
                        #     })
                        # })
                    output["channel"] = "tool"  # 可以兼容channel loss技术
            messages.append(messages_now)
            if len(messages) > 2:
                break
        output["messages"] = messages
        return output


# 注册数据集，使用新的预处理器
'''
# 当前自定义数据格式为openai接口返回的数据格式 messages，并添加了输入的tools字段，输出的messages中有tool_call字段如
{messages:[{role:'user',content:'...'},{role:'tool_call',content:'...',tool_call:[]}],tools:[...]}
'''
register_dataset(
    DatasetMeta(
        dataset_path='/workspace/llm/dataset/self_build/origin_data/20251013/custom_dataset_1013/dataset_output/gesture_1013.jsonl',
        dataset_name='gesture_1013',
        preprocess_func=ToolCallPreprocessor(),  # 取消注释并使用新的预处理器
    ))

register_dataset(
    DatasetMeta(
        dataset_path='/workspace/llm/dataset/self_build/origin_data/20251013/custom_dataset_1013/dataset_output/initialize_1013.jsonl',
        dataset_name='initialize_1013',
        preprocess_func=ToolCallPreprocessor(),  # 取消注释并使用新的预处理器
    ))


if __name__ == '__main__':
    # 测试数据集加载，load_dataset默认兼容标准的alpaca，sharegpt，以及openai messages格式
    dataset = load_dataset(['gesture_1013', 'initialize_1013'],split_dataset_ratio=0.1)
    print(f'dataset_deal: {dataset}')
    dataset = dataset[0]
    print(f'dataset: {dataset}')
    print(f'dataset[0]: {dataset[0]}')


    '''
    load_dataset 参数
    datasets: Union[List[str], str], #注册好的书籍名称
    *,
    split_dataset_ratio: float = 0.,  # 数据集分割比例
    seed: Union[int, np.random.RandomState, None] = 42,  # 随机种子
    num_proc: int = 1,  # 处理进程数
    load_from_cache_file: bool = True,  # 是否从缓存文件加载
    shuffle: bool = False,  # 是否打乱数据
    streaming: bool = False,  # 是否使用流式处理
    interleave_prob: Optional[List[float]] = None,  # 交错概率
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted',  # 停止策略
    shuffle_buffer_size: int = 1000,  # 打乱缓冲区大小
    use_hf: Optional[bool] = None,  # 是否使用HuggingFace
    hub_token: Optional[str] = None,  # Hub令牌
    strict: bool = False,  # 严格模式
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',  # 下载模式
    columns: Optional[Dict[str, str]] = None,  # 列映射
    remove_unused_columns: bool = True,  # 是否移除未使用的列
    # 自我认知
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,  # 模型名称（中文，英文）
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,  # 模型作者（中文，英文）
    '''