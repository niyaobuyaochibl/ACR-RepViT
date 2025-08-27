#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估ACR-RepViT模型
此脚本用于评估不同重参数化策略的性能
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from timm.data import create_dataset, create_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy, AverageMeter
import numpy as np

# 导入模型定义
from model.repvit import repvit_m0_9
from model.acr_repvit import (
    acr_repvit_m0_9,
    acr_repvit_m0_9_decomposed,
    acr_repvit_m0_9_multiscale,
    acr_repvit_m0_9_attention
)

def get_args_parser():
    parser = argparse.ArgumentParser('ACR-RepViT 模型评估', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-path', default='/root/autodl-tmp/data', type=str, help='数据集路径')
    parser.add_argument('--model-path', default='checkpoints/acr_repvit', type=str, help='模型权重路径')
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--device', default='cuda', help='设备')
    parser.add_argument('--pin-mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.add_argument('--measure-latency', action='store_true', help='是否测量推理延迟')
    parser.add_argument('--input-size', default=224, type=int, help='输入图像大小')
    return parser

def main(args):
    print("正在评估ACR-RepViT模型及其变体...")
    device = torch.device(args.device)
    
    # 准备验证集数据加载器
    dataset_val = create_dataset(
        'imagenet-val', root=args.data_path, split='val', is_training=False, 
        batch_size=args.batch_size)
    
    loader_val = create_loader(
        dataset_val,
        input_size=args.input_size,
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=True,
        interpolation='bicubic',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    
    # 定义要评估的模型字典及其权重路径
    models_to_evaluate = {
        'repvit_m0_9_baseline': {
            'model_fn': repvit_m0_9,
            'checkpoint': os.path.join('pretrained', 'repvit_m0_9_distill_300e.pth'),
            'args': {}
        },
        'acr_repvit_decomposed': {
            'model_fn': acr_repvit_m0_9_decomposed,
            'checkpoint': os.path.join(args.model_path, 'decomposed', 'checkpoint-best.pth'),
            'args': {}
        },
        'acr_repvit_multiscale': {
            'model_fn': acr_repvit_m0_9_multiscale,
            'checkpoint': os.path.join(args.model_path, 'multiscale', 'checkpoint-best.pth'),
            'args': {}
        },
        'acr_repvit_attention': {
            'model_fn': acr_repvit_m0_9_attention,
            'checkpoint': os.path.join(args.model_path, 'attention', 'checkpoint-best.pth'),
            'args': {}
        },
        'acr_repvit_progressive': {
            'model_fn': acr_repvit_m0_9,
            'checkpoint': os.path.join(args.model_path, 'progressive', 'checkpoint-best.pth'),
            'args': {}
        }
    }
    
    # 结果表格
    results = []
    header = ["模型", "Top-1 准确率", "Top-5 准确率", "参数量(M)", "重参数化前参数量(M)", "延迟(ms)"]
    results.append(header)
    
    for model_name, config in models_to_evaluate.items():
        print(f"\n评估模型: {model_name}")
        # 检查权重文件是否存在
        if not os.path.isfile(config['checkpoint']) and not model_name == 'repvit_m0_9_baseline':
            print(f"警告: 找不到权重文件 {config['checkpoint']}，跳过此模型")
            results.append([model_name, "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue
            
        # 创建模型
        model_fn = config['model_fn']
        model = model_fn(**config['args'])
        print(f"模型创建: {model_name}")
        
        # 计算模型参数量
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        
        # 加载权重
        try:
            if model_name != 'repvit_m0_9_baseline' or os.path.isfile(config['checkpoint']):
                checkpoint = torch.load(config['checkpoint'], map_location='cpu')
                if 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"加载权重: {config['checkpoint']}")
        except Exception as e:
            print(f"加载权重失败: {e}")
        
        # 重参数化模型
        try:
            orig_params = param_count
            if hasattr(model, 'fuse'):
                model = model.fuse()
                print(f"模型已重参数化")
        except Exception as e:
            print(f"重参数化失败: {e}")
        
        # 模型移至设备
        model = model.to(device)
        model.eval()
        
        # 测量延迟
        latency = "N/A"
        if args.measure_latency:
            latency = measure_inference_latency(model, device, args.input_size)
        
        # 评估精度
        top1, top5 = validate(model, loader_val, device)
        
        # 收集结果
        results.append([
            model_name, 
            f"{top1:.2f}%", 
            f"{top5:.2f}%", 
            f"{param_count:.2f}M", 
            f"{orig_params:.2f}M", 
            f"{latency}" if latency != "N/A" else "N/A"
        ])
    
    # 打印结果表格
    print("\n========== ACR-RepViT 评估结果 ==========")
    # 计算每列的最大宽度
    col_widths = [max(len(row[i]) for row in results) for i in range(len(header))]
    # 打印表头
    print("+" + "+".join(['-' * (width + 2) for width in col_widths]) + "+")
    print("|" + "|".join([f" {results[0][i]:{col_widths[i]}} " for i in range(len(header))]) + "|")
    print("+" + "+".join(['-' * (width + 2) for width in col_widths]) + "+")
    # 打印数据行
    for row in results[1:]:
        print("|" + "|".join([f" {row[i]:{col_widths[i]}} " for i in range(len(header))]) + "|")
    print("+" + "+".join(['-' * (width + 2) for width in col_widths]) + "+")
    
    # 保存结果到文件
    save_dir = os.path.join(args.model_path, 'evaluation_results')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'acr_repvit_evaluation.txt'), 'w') as f:
        for row in results:
            f.write("\t".join(row) + "\n")
    
    print(f"\n结果已保存至: {os.path.join(save_dir, 'acr_repvit_evaluation.txt')}")

def validate(model, data_loader, device):
    """评估模型在数据集上的准确率"""
    criterion = nn.CrossEntropyLoss()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    
    # 切换到评估模式
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)
            
            # 前向传播
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]
            
            # 计算准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            batch_size = images.shape[0]
            top1_meter.update(acc1.item(), batch_size)
            top5_meter.update(acc5.item(), batch_size)
    
    return top1_meter.avg, top5_meter.avg

def measure_inference_latency(model, device, input_size, n_warmup=50, n_measure=100):
    """测量模型的推理延迟"""
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    
    # 预热
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
    
    # 同步GPU
    torch.cuda.synchronize()
    
    # 测量延迟
    latency_list = []
    with torch.no_grad():
        for _ in range(n_measure):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end_time = time.time()
            latency_list.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    return f"{np.mean(latency_list):.2f}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser('ACR-RepViT 模型评估', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args) 