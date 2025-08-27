#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析CBAM注意力机制消融实验变体的参数量、计算复杂度和推理时间
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from thop import profile
from timm import create_model
from timm.models import register_model

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模型
from model.acr_repvit import (
    acr_repvit_m0_9, acr_repvit_m0_9_cbam, 
    acr_repvit_m0_9_ca, acr_repvit_m0_9_sa, 
    acr_repvit_m0_9_multiscale
)

# 加载模型
def load_model(model_name, pretrained=False):
    print(f"Loading model: {model_name}")
    
    # 手动创建模型
    if model_name == 'acr_repvit_m0_9':
        model = acr_repvit_m0_9(pretrained=pretrained)
    elif model_name == 'acr_repvit_m0_9_cbam':
        model = acr_repvit_m0_9_cbam(pretrained=pretrained)
    elif model_name == 'acr_repvit_m0_9_ca':
        model = acr_repvit_m0_9_ca(pretrained=pretrained)
    elif model_name == 'acr_repvit_m0_9_sa':
        model = acr_repvit_m0_9_sa(pretrained=pretrained)
    elif model_name == 'acr_repvit_m0_9_multiscale':
        model = acr_repvit_m0_9_multiscale(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# 计算参数量和计算复杂度
def calc_model_complexity(model, input_size=(1, 3, 224, 224)):
    input = torch.randn(input_size)
    macs, params = profile(model, inputs=(input,))
    return params / 1e6, macs / 1e9  # 转换为M和G

# 测量推理时间
def measure_inference_time(model, input_size=(1, 3, 224, 224), n_repeat=100, warmup=10):
    input = torch.randn(input_size).cuda()
    model = model.cuda()
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input)
    
    # 计时
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_repeat):
            _ = model(input)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) * 1000 / n_repeat  # 转换为毫秒

# 分析模型
def analyze_models(model_names, pretrained=False):
    results = []
    
    for model_name in model_names:
        # 加载模型
        model = load_model(model_name, pretrained)
        
        # 计算参数量和计算复杂度
        params, flops = calc_model_complexity(model)
        
        # 测量推理时间
        infer_time = measure_inference_time(model)
        
        # 保存结果
        results.append({
            'Model': model_name,
            'Parameters (M)': params,
            'FLOPs (G)': flops,
            'Inference Time (ms)': infer_time
        })
        
        print(f"Model: {model_name}, Params: {params:.2f}M, FLOPs: {flops:.2f}G, Inference Time: {infer_time:.2f}ms")
    
    return pd.DataFrame(results)

# 绘制对比图
def plot_comparison(df, save_dir='./visualization_output'):
    os.makedirs(save_dir, exist_ok=True)
    
    # 参数量比较
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Parameters (M)'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}', ha='center')
    plt.title('Model Parameters Comparison')
    plt.ylabel('Parameters (M)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'params_comparison.png'))
    
    # 计算复杂度比较
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['FLOPs (G)'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', ha='center')
    plt.title('Model FLOPs Comparison')
    plt.ylabel('FLOPs (G)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'flops_comparison.png'))
    
    # 推理时间比较
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Inference Time (ms)'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.2f}', ha='center')
    plt.title('Model Inference Time Comparison')
    plt.ylabel('Inference Time (ms)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'infer_time_comparison.png'))
    
    # 保存表格
    df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
    
    print(f"Visualization results saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze CBAM attention mechanism ablation variants')
    parser.add_argument('--save-dir', default='/root/autodl-tmp/RepViT_output/visualization_output', help='Directory to save visualization results')
    args = parser.parse_args()
    
    # 待分析模型列表
    model_names = [
        'acr_repvit_m0_9',           # 原始模型
        'acr_repvit_m0_9_cbam',      # 完整CBAM注意力
        'acr_repvit_m0_9_ca',        # 仅通道注意力
        'acr_repvit_m0_9_sa',        # 仅空间注意力
        'acr_repvit_m0_9_multiscale' # 多尺度卷积
    ]
    
    # 分析模型
    df = analyze_models(model_names)
    
    # 绘制对比图
    plot_comparison(df, args.save_dir) 