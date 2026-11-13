#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from swift.llm import sft_main, TrainArguments  # , SwiftSft
from typing import Optional, Union, List

# def sft_main(args: Optional[Union[List[str], TrainArguments]] = None):
#     return SwiftSft(args).main()

if __name__ == '__main__':
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
        dataset=['custom_stop_speaking'],
        # 是否启用通道损失，用于多任务学习或特殊损失函数
        enable_channel_loss=True,
        
        # ==================== 数据集配置 ====================
        # 数据集分割比例，将训练集按此比例分割出验证集
        split_dataset_ratio=0.01,
        # 数据集预处理进程数，控制数据预处理的并行度
        dataset_num_proc=4,
        
        # ==================== 训练配置 ====================
        # 训练轮数，整个数据集训练的遍数
        num_train_epochs=3,
        # 每个设备的训练批次大小，根据GPU内存调整
        per_device_train_batch_size=8,
        # 每个设备的评估批次大小
        per_device_eval_batch_size=5,
        # 学习率，控制模型参数更新的步长
        learning_rate=1e-5,
        # 梯度累积步数，模拟更大批次训练，解决显存不足问题
        gradient_accumulation_steps=16,
        
        # ==================== 日志和保存配置 ====================
        # 保存检查点的步数间隔
        save_steps=100,
        # 最多保存的检查点数量，超出时会删除旧的
        save_total_limit=2,
        # 日志记录步数间隔
        logging_steps=5,
        # 学习率预热比例，在训练开始时线性增加学习率
        warmup_ratio=0.05,
        # 输出目录，保存训练结果和检查点
        output_dir='output',
        
        # ==================== 数据加载配置 ====================
        # 数据加载器的工作进程数
        dataloader_num_workers=4,
        # 输入序列的最大长度，超过会被截断
        max_length=8192,
        
        # ==================== 评估配置 ====================
        # 评估策略：'steps'按步数评估，'epoch'按轮次评估
        eval_strategy='steps',
        # 评估步数间隔，每多少步评估一次
        eval_steps=2,
        # 是否使用评估作用域
        eval_use_evalscope=True,
        # 评估数据集名称列表
        eval_dataset=['gsm8k'],
        # 评估时使用的样本数量限制
        eval_limit=10,
        # 额外的评估参数，用于配置实验跟踪工具
        extra_eval_args={'swanlab': {'run_name': 'qwen3-0.6B-full-test'}},
        
        # ==================== 自定义注册路径 ====================
        # 自定义数据集注册文件的路径
        custom_register_path='examples/custom_dataset/dataset.py',
        
        # ==================== SwanLab配置 ====================
        # SwanLab API令牌，用于身份验证
        swanlab_token='YOUR_SWANLAB_TOKEN',
        # SwanLab项目名称
        swanlab_project='ms-swift',
        # SwanLab工作空间名称
        swanlab_workspace='demo',
        # SwanLab实验名称
        swanlab_exp_name='custom_stop_speaking',
        # SwanLab模式：'cloud'云端模式，'local'本地模式
        swanlab_mode='cloud',
        # 飞书Webhook URL，用于通知
        swanlab_lark_webhook_url='YOUR_LARK_WEBHOOK_URL',
        # 飞书Webhook密钥
        swanlab_lark_secret='YOUR_LARK_SECRET',
        
        # ==================== 其他配置 ====================
        # 运行名称，用于标识本次训练
        run_name='qwen3-0.6B-full-test',
        # 报告到的工具，用于实验跟踪和日志记录
        report_to='swanlab'
    )
    
    # 旧版本的调用方式（已注释）
    # sft = SwiftSft(args)
    # 运行SFT训练
    # sft.main()
    
    # 新版本的调用方式，直接使用sft_main函数
    result = sft_main(args)
    print(result)