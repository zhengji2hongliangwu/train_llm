import os
import json
# os.environ["MAX_PIXELS"] = "1003520" # 输入图像最大像素，多模态模型时用到
from swift.llm import get_model_tokenizer, get_template
from dataclasses import asdict
_, tokenizer = get_model_tokenizer('/workspace/llm/models/Qwen/Qwen3-0.6B', load_model=False)
template = get_template(tokenizer.model_meta.template, tokenizer)

## 不支持直接openai返回的格式即role 支持tool_call 作为role 读取contant里内容作为toolcall内容，但是不支持多个tool_calllist
data = {
    'messages': [
        {'role': 'system', 'content': '你是智能服务助手。'},
        {'role': 'user', 'content': '你好'},
        {'role': 'assistant', 'content': '你好！有什么我可以帮助你的吗？'},
        {'role': 'user', 'content': '我想知道北京的天气'},
        {'role': 'assistant', 'content': '让我查查\n\n'},
        {'role': 'tool_call', 'content': json.dumps({"name": "get_current_weather", "arguments": {"location": "北京", "unit": "摄氏度"}})},
        {'role': 'tool_call', 'content': json.dumps({"name": "get_current_weather", "arguments": {"location": "北京", "unit": "摄氏度"}})},
        # {'role': 'assistant', 'content': '让我查查\n\n<tool_call>\n\n{"name": "get_current_weather", "arguments": {"location": "北京", "unit": "摄氏度"}}\n\n</tool_call>'},
        # 不支持 {'role': 'assistant', 'content': '让我查查', 'tool_call': [{"type":"function","function":{"name":"gesture","arguments":{"hand":"left"}}}]},
        {'role': 'tool', 'content': '25'},
        {'role': 'assistant', 'content': '北京的天气是25度，天气晴朗。'},

    ],
    'tools': [
        {
            'type': 'function',
            'function': {
                'name': 'get_current_weather',
                'arguments': '{"location": "string", "unit": "string"}'
            }
        }
    ],
    # "system": "你是智能服务助手。"
}

# --enable_channel_loss true
# 如果你要使用channel loss，你需要设置--enable_channel_loss true，并在数据集中增加"channel"字段。channel loss兼容packing/padding_free/loss_scale等技术。

template.default_system = '你是智能服务助手。'
print(template)
template.set_mode('train')
encoded = template.encode(data, return_template_inputs=True)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
print(f'images: {encoded["template_inputs"].images}')