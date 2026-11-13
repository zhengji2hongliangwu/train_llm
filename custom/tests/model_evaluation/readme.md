针对您的工具调用准确性评估需求，我来为您设计一套完整的评测方案。基于您提供的数据集格式和API信息，这里是一个端到端的评估实现：

## 🎯 评测方案设计

### 1. 核心评测指标

```python
import json
import re
import requests
from typing import Dict, List, Any, Tuple

class ToolCallEvaluator:
    def __init__(self, api_url: str, model_id: str):
        self.api_url = api_url
        self.model_id = model_id
        self.metrics = {
            'tool_selection_accuracy': 0,
            'parameter_accuracy': 0,
            'format_correctness': 0,
            'end_to_end_accuracy': 0,
            'natural_response_quality': 0
        }
        self.total_samples = 0
        self.tool_selection_correct = 0
        self.parameter_correct = 0
        self.format_correct = 0
        self.end_to_end_correct = 0
```

### 2. 关键评测函数

```python
    def call_model_api(self, messages: List[Dict], tools: List[Dict]) -> Dict:
        """调用本地部署的模型API"""
        payload = {
            "model": self.model_id,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            return response.json()
        except Exception as e:
            print(f"API调用失败: {e}")
            return {"choices": [{"message": {"content": ""}}]}

    def extract_tool_call(self, model_output: str) -> Tuple[str, Dict]:
        """从模型输出中提取工具调用信息"""
        tool_call_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        match = re.search(tool_call_pattern, model_output, re.DOTALL)
        
        if match:
            try:
                tool_data = json.loads(match.group(1))
                tool_name = tool_data.get("name", "")
                arguments = tool_data.get("arguments", {})
                return tool_name, arguments
            except json.JSONDecodeError:
                return "", {}
        return "", {}

    def parse_ground_truth(self, assistant_message: str) -> Tuple[str, Dict]:
        """从标注数据中解析真实工具调用"""
        return self.extract_tool_call(assistant_message)

    def evaluate_tool_selection(self, predicted_tool: str, ground_truth_tool: str) -> bool:
        """评估工具选择准确性"""
        return predicted_tool == ground_truth_tool

    def evaluate_parameters(self, predicted_params: Dict, ground_truth_params: Dict, tool_schema: Dict) -> bool:
        """评估参数准确性"""
        if not predicted_params or not ground_truth_params:
            return predicted_params == ground_truth_params
        
        # 检查必需参数
        required_params = tool_schema.get('required', [])
        for param in required_params:
            if param not in predicted_params:
                return False
            # 简单值比较 - 实际中可以更复杂
            if str(predicted_params[param]) != str(ground_truth_params.get(param)):
                return False
        
        return True

    def evaluate_format_correctness(self, model_output: str, has_tool_call: bool) -> bool:
        """评估输出格式正确性"""
        if has_tool_call:
            # 检查工具调用格式
            return bool(re.search(r'<tool_call>\s*{.*?}\s*</tool_call>', model_output, re.DOTALL))
        else:
            # 检查自然回复格式（无tool_call标签）
            return not bool(re.search(r'<tool_call>', model_output))
```

### 3. 批量评估主函数

```python
    def evaluate_dataset(self, dataset, sample_limit=100) -> Dict:
        """在数据集上运行批量评估"""
        results = []
        
        for i, sample in enumerate(dataset):
            if i >= sample_limit:
                break
                
            print(f"处理样本 {i+1}/{min(sample_limit, len(dataset))}")
            
            # 准备输入（只取到user消息）
            input_messages = []
            for msg in sample['messages']:
                if msg['role'] in ['system', 'user']:
                    input_messages.append(msg)
                elif msg['role'] == 'assistant':
                    break  # 停止在第一个assistant消息前
            
            # 调用模型
            model_response = self.call_model_api(input_messages, sample['tools'])
            model_output = model_response['choices'][0]['message'].get('content', '')
            
            # 获取真实标注
            ground_truth_msg = next(
                (msg for msg in sample['messages'] if msg['role'] == 'assistant'), 
                {'content': ''}
            )
            
            # 解析工具调用
            pred_tool, pred_params = self.extract_tool_call(model_output)
            true_tool, true_params = self.parse_ground_truth(ground_truth_msg['content'])
            
            # 获取当前工具的schema
            tool_schema = next(
                (tool['function'] for tool in sample['tools'] if tool['function']['name'] == true_tool),
                {}
            ) if true_tool else {}
            
            # 执行评估
            tool_selection_correct = self.evaluate_tool_selection(pred_tool, true_tool)
            params_correct = self.evaluate_parameters(pred_params, true_params, tool_schema)
            format_correct = self.evaluate_format_correctness(model_output, bool(true_tool))
            
            # 端到端准确性
            end_to_end_correct = tool_selection_correct and params_correct and format_correct
            
            # 更新统计
            self.total_samples += 1
            if tool_selection_correct:
                self.tool_selection_correct += 1
            if params_correct and true_tool:  # 只有需要工具调用时才计参数准确率
                self.parameter_correct += 1
            if format_correct:
                self.format_correct += 1
            if end_to_end_correct:
                self.end_to_end_correct += 1
            
            # 记录结果
            result = {
                'sample_id': i,
                'user_input': next((msg['content'] for msg in sample['messages'] if msg['role'] == 'user'), ''),
                'predicted_tool': pred_tool,
                'true_tool': true_tool,
                'predicted_params': pred_params,
                'true_params': true_params,
                'tool_selection_correct': tool_selection_correct,
                'parameters_correct': params_correct,
                'format_correct': format_correct,
                'end_to_end_correct': end_to_end_correct,
                'model_output': model_output
            }
            results.append(result)
        
        return self._calculate_final_metrics(results)

    def _calculate_final_metrics(self, results: List[Dict]) -> Dict:
        """计算最终评估指标"""
        total_tool_calls = sum(1 for r in results if r['true_tool'])
        
        metrics = {
            'tool_selection_accuracy': self.tool_selection_correct / self.total_samples,
            'parameter_accuracy': self.parameter_correct / total_tool_calls if total_tool_calls > 0 else 0,
            'format_correctness': self.format_correct / self.total_samples,
            'end_to_end_accuracy': self.end_to_end_correct / self.total_samples,
            'total_samples': self.total_samples,
            'total_tool_calls': total_tool_calls
        }
        
        return metrics
```

### 4. 运行评估和结果分析

```python
def run_evaluation():
    # 初始化评估器
    evaluator = ToolCallEvaluator(
        api_url='http://localhost:18008/v1',
        model_id="Qwen3-0.6B"
    )
    
    # 运行评估（限制样本数量以便快速测试）
    metrics = evaluator.evaluate_dataset(dataset, sample_limit=100)
    
    # 打印结果
    print("\n" + "="*50)
    print("工具调用准确性评估结果")
    print("="*50)
    print(f"总样本数: {metrics['total_samples']}")
    print(f"工具调用样本: {metrics['total_tool_calls']}")
    print(f"工具选择准确率: {metrics['tool_selection_accuracy']:.2%}")
    print(f"参数准确率: {metrics['parameter_accuracy']:.2%}")
    print(f"格式正确率: {metrics['format_correctness']:.2%}")
    print(f"端到端准确率: {metrics['end_to_end_accuracy']:.2%}")
    
    return metrics

# 执行评估
results = run_evaluation()
```

### 5. 错误分析工具

```python
def analyze_errors(results: List[Dict]):
    """分析常见错误模式"""
    tool_selection_errors = []
    parameter_errors = []
    format_errors = []
    
    for result in results:
        if not result['tool_selection_correct']:
            tool_selection_errors.append(result)
        elif not result['parameters_correct']:
            parameter_errors.append(result)
        elif not result['format_correct']:
            format_errors.append(result)
    
    print(f"\n错误分析:")
    print(f"工具选择错误: {len(tool_selection_errors)} 个")
    print(f"参数错误: {len(parameter_errors)} 个") 
    print(f"格式错误: {len(format_errors)} 个")
    
    # 显示一些错误示例
    if tool_selection_errors:
        print("\n工具选择错误示例:")
        for error in tool_selection_errors[:3]:
            print(f"输入: {error['user_input']}")
            print(f"预测: {error['predicted_tool']}, 正确: {error['true_tool']}")

# 运行错误分析
analyze_errors(results)
```

## 📊 评估报告生成

```python
def generate_evaluation_report(metrics: Dict, results: List[Dict]):
    """生成详细的评估报告"""
    report = {
        'model_info': {
            'api_url': 'http://localhost:18008/v1',
            'model_id': 'Qwen3-0.6B',
            'evaluation_time': datetime.now().isoformat()
        },
        'summary_metrics': metrics,
        'detailed_results': results[:10],  # 只保留前10个详细结果
        'recommendations': []
    }
    
    # 基于结果给出建议
    if metrics['tool_selection_accuracy'] < 0.8:
        report['recommendations'].append("建议加强工具选择能力的训练")
    if metrics['parameter_accuracy'] < 0.8:
        report['recommendations'].append("建议改进参数提取的准确性")
    
    # 保存报告
    with open('tool_call_evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report
```

这个评估方案能够全面测试您的模型在工具调用任务上的表现，包括工具选择、参数生成、格式正确性等关键维度。您可以根据需要调整样本数量或添加更复杂的评估逻辑。