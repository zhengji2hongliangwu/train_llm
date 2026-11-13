
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional
import json  # 需要添加json模块
from swift.llm import ResponsePreprocessor


class ToolCallPreprocessor_CustomQwen3(ResponsePreprocessor):
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
                    output["channel"] = "tool"  # 可以兼容channel loss技术
            messages.append(messages_now)
            if len(messages) > 2:
                break
        output["messages"] = messages
        return output
if __name__ == '__main__':
    preprocessor = ToolCallPreprocessor_Qwen3()
    row = {
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？", "tool_calls": [
                {"id": "call_123", "function": {"name": "get_weather", "arguments": "{\"location\": \"北京\"}"}}
            ]}
        ],
        "tools": [
            {"name": "get_weather", "description": "获取指定位置的天气信息"}
        ]
    }
    processed_row = preprocessor.preprocess(row)
    print(processed_row)