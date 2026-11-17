# train_llm

基于 [ms-swift 3.10.2](https://github.com/modelscope/ms-swift) 的工具调用（tool-call）大模型训练与评测示例，覆盖数据预处理、SFT、GRPO/GSPO 强化学习、vLLM 部署与评测。示例模型为 Qwen3-0.6B。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `trainingexample/trainning.py` | SFT 主入口：注册数据集、自定义预处理器、启动训练 |
| `trainingexample/datatest.py` | 数据预检：加载数据集并打印模板编码后的 input_ids / labels |
| `trainingexample/config_traing_data.json` | 训练数据集注册配置 |
| `trainingexample/config_test_data.json` | 评测数据集注册配置 |
| `trainingexample/custom/` | 自定义数据集 / 模型 / 训练 / 推理的最小示例 |
| `trainingexample/grpo_qwen3/` | GRPO/GSPO：`dataset.py`、`gspo_0.6b.sh`、`plugin.py`（自定义 reward、多轮调度）、`orm.py`（reward 基类集合） |
| `custom/dataset/utils/preprocessor.py` | 工具调用数据预处理器（独立版本） |
| `custom/tests/` | 历史调试脚本，非正式单测（部分文件为空或依赖本地绝对路径） |
| `infer/` | vLLM OpenAI 兼容服务启动脚本 |
| `test/eaval_test.py` | evalscope `tool_bench` 评测示例 |

## 环境安装
但
```bash
conda create -n train_llm python=3.11
conda activate train_llm
pip install uv
# flash_attn 需要关闭 build isolation
pip install --no-build-isolation -r requirements.txt
```

主要依赖：`torch==2.7.1`、`transformers==4.56.2`、`ms-swift==3.10.2`、`trl==0.20.0`、`deepspeed==0.17.5`、`vllm==0.10.2`。

## 数据格式

每条样本是 OpenAI messages 格式，可带 `tools` 字段：

```json
{
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}}]},
    {"role": "tool", "content": "..."}
  ]
}
```

预处理（`preprocess: "custom"`）会做三件事：

1. 把 `assistant.tool_calls` 拼成 `<tool_call>{"name": ..., "arguments": ...}</tool_call>` 追加到 content；
2. 把 `tool` / `function_response` / `observation` 统一成 `tool_response`；
3. 写出 `channel` 字段（`tool` / `chat` / `self_condition`），配合 `enable_channel_loss=True` 按通道掩码 loss。

`preprocess: "auto"` 则交给 swift 的 `AutoPreprocessor`，仅补 `channel`。

## 使用方法

### 1. 数据预检

```bash
cd trainingexample
python datatest.py
```

确认数据集能正常加载、`<tool_call>` 拼接与 labels 掩码符合预期后再训练。

### 2. SFT

```bash
cd trainingexample
python trainning.py
```

训练前按需修改 `trainning.py` 中的 `TrainArguments`：

| 参数 | 示例 | 说明 |
| --- | --- | --- |
| `model` | `/workspace/llm/models/Qwen/Qwen3-0.6B` | 基座模型路径 |
| `train_type` | `full` / `lora` / `qlora` | 训练方式 |
| `dataset` | `config_traing_data.json` 的 key 列表 | 训练集（脚本自动读取） |
| `eval_dataset` | `config_test_data.json` 的 key 列表 | 评测集（脚本自动读取） |
| `enable_channel_loss` | `True` | 按 channel 掩码 loss |
| `eval_use_evalscope` | `True` | 训练中调用 evalscope 评测 |
| `output_dir` | `outputs/training/run_xxx` | 输出目录 |

### 3. GRPO / GSPO 强化学习

```bash
cd trainingexample/grpo_qwen3
sh gspo_0.6b.sh
```

- `swift rlhf --rlhf_type grpo`，2 卡 + `--deepspeed zero3`，vLLM colocate 采样；
- `--external_plugins plugin.py` 挂载自定义 reward（`ToolUseCorrectnessReward`、`ToolUseFormatReward`、`ToolUseLengthReward`）与多轮调度；
- `orm.py` 内含通用 reward 基类（格式、重复惩罚、soft overlong 等），可扩展自己的指标。

### 4. 部署推理（vLLM）

```bash
sh infer/qwen3_vllm_openai.sh            # 基座模型，端口 19008
sh infer/qwen3_vllm_openai_checkpoint.sh # 微调 checkpoint，端口 18008
```

服务为 OpenAI 兼容接口。工具调用依赖 `--enable-auto-tool-choice --tool-call-parser hermes`，去掉后模型输出的 `<tool_call>` 不会解析为结构化 `tool_calls`。

### 5. 评测

```bash
python test/eaval_test.py
```

基于 evalscope 在 `tool_bench` 上评测 OpenAI 兼容服务，需先填好 `api_url` / `api_key`。

## 注意事项

1. 数据、模型、输出路径大量硬编码为 `/workspace/llm/...`（含两个 config json），换机器必须全量替换。
2. 训练与服务脚本写死了 `CUDA_VISIBLE_DEVICES`，按实际卡号修改。
3. `swanlab_token`、`swanlab_lark_webhook_url` 等仍是 `YOUR_*` 占位符，训练前替换为真实值；不上报时可改 `report_to`。
4. 文件名保留原拼写：`trainning.py`、`config_traing_data.json`（不是 training），引用时注意别写错。
5. `trainingexample/custom/*.sh` 沿用上游 swift 示例路径（`examples/custom/...`），直接运行前需改为本目录路径。
6. `infer/requirement.txt` 里的 `vllm==0.9.2` 与主 `requirements.txt` 的 `0.10.2` 不一致，只部署推理时注意统一版本。
