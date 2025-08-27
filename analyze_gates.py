#!/usr/bin/env python3
# 门控分析脚本

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from model.simple_dynarepvit import simple_dynarepvit_m0_9

def analyze_gates(checkpoint_path, output_dir="gate_analysis"):
    """分析门控机制的激活分布"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"分析检查点: {checkpoint_path}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        return None
    
    try:
        # 加载模型
        model = simple_dynarepvit_m0_9(pretrained=False)
        
        # 尝试加载检查点
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            return None
        
        # 尝试不同的键来加载状态字典
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_ema' in checkpoint and checkpoint['model_ema'] is not None:
            print("使用EMA模型状态")
            state_dict = checkpoint['model_ema']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 检查是否包含门控参数
        has_gate_params = any('channel_gate' in k for k in state_dict.keys())
        if not has_gate_params:
            print("警告: 此检查点似乎不包含门控参数")
            
        # 加载模型参数
        try:
            model.load_state_dict(state_dict, strict=False)
            print("已成功加载模型权重")
        except Exception as e:
            print(f"警告: 加载模型权重时出现非严重错误: {e}")
            print("尝试继续分析...")
        
        # 设置为评估模式
        model.eval()
        
        # 尝试获取门控状态
        try:
            gates_info = model.get_gates_status()
        except Exception as e:
            print(f"获取门控状态时出错: {e}")
            return None
        
        # 检查门控信息是否有效
        if not gates_info or len(gates_info) == 0:
            print("错误: 未能从模型获取有效的门控信息")
            return None
        
        # 打印总体激活率
        if 'activation_rate' in gates_info:
            print(f"总体通道激活率: {gates_info['activation_rate'] * 100:.2f}%")
            print(f"计算量降低约: {gates_info['compute_saving_estimate'] * 100:.2f}%")
            
            if 'mean_block_active_rate' in gates_info:
                print(f"平均块激活率: {gates_info['mean_block_active_rate'] * 100:.2f}%")
                print(f"最小块激活率: {gates_info['min_block_active_rate'] * 100:.2f}%")
                print(f"最大块激活率: {gates_info['max_block_active_rate'] * 100:.2f}%")
        else:
            print("警告: 未找到激活率信息")
            
        # 保存分析结果到文本文件
        with open(os.path.join(output_dir, "analysis_summary.txt"), "w") as f:
            f.write(f"检查点: {checkpoint_path}\n")
            f.write(f"总体通道激活率: {gates_info.get('activation_rate', 0) * 100:.2f}%\n")
            f.write(f"计算量降低约: {gates_info.get('compute_saving_estimate', 0) * 100:.2f}%\n")
            
            if 'mean_block_active_rate' in gates_info:
                f.write(f"平均块激活率: {gates_info['mean_block_active_rate'] * 100:.2f}%\n")
                f.write(f"最小块激活率: {gates_info['min_block_active_rate'] * 100:.2f}%\n")
                f.write(f"最大块激活率: {gates_info['max_block_active_rate'] * 100:.2f}%\n")
        
        # 绘制每层门控分布
        layer_names = []
        layer_means = []
        
        for name, values in gates_info.items():
            if name.startswith('block_') and not name.endswith('_active_rate'):
                try:
                    layer_names.append(name)
                    mean_val = float(np.mean(values))
                    layer_means.append(mean_val)
                    
                    # 绘制每层的门控分布直方图
                    plt.figure(figsize=(10, 6))
                    plt.hist(values, bins=20, alpha=0.7)
                    plt.axvline(x=mean_val, color='r', linestyle='--', 
                               label=f'平均值: {mean_val:.3f}')
                    plt.title(f'层 {name} 门控分布')
                    plt.xlabel('门控值')
                    plt.ylabel('频率')
                    plt.legend()
                    plt.savefig(f"{output_dir}/{name}_distribution.png")
                    plt.close()
                except Exception as e:
                    print(f"绘制 {name} 的分布图时出错: {e}")
        
        # 只有当有层数据时才绘制
        if layer_names and layer_means:
            try:
                # 绘制各层平均门控值
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(layer_names)), layer_means)
                plt.xticks(range(len(layer_names)), layer_names, rotation=90)
                plt.xlabel('层名称')
                plt.ylabel('平均门控值')
                plt.title('各层平均门控值分布')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/layer_mean_gates.png")
            except Exception as e:
                print(f"绘制平均门控值图时出错: {e}")
        
        # 绘制激活率的层分布
        if any(k.endswith('_active_rate') for k in gates_info.keys()):
            try:
                block_names = []
                active_rates = []
                
                for key, value in gates_info.items():
                    if key.endswith('_active_rate'):
                        block_name = key.replace('_active_rate', '')
                        block_names.append(block_name)
                        active_rates.append(value * 100)  # 转为百分比
                
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(block_names)), active_rates)
                plt.xticks(range(len(block_names)), block_names, rotation=90)
                plt.xlabel('块名称')
                plt.ylabel('激活率 (%)')
                plt.title('各块激活率分布')
                if 'activation_rate' in gates_info:
                    plt.axhline(y=gates_info['activation_rate'] * 100, color='r', linestyle='--',
                               label=f'平均激活率: {gates_info["activation_rate"] * 100:.2f}%')
                    plt.legend()
                plt.tight_layout()
                plt.savefig(f"{output_dir}/block_activation_rates.png")
            except Exception as e:
                print(f"绘制激活率分布图时出错: {e}")
        
        print(f"分析完成，结果保存在: {output_dir}")
        return gates_info
    except Exception as e:
        print(f"分析过程中出现严重错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='gate_analysis',
                       help='Output directory for analysis')
    args = parser.parse_args()
    
    result = analyze_gates(args.checkpoint, args.output)
    if result is None:
        sys.exit(1)  # 如果分析失败则返回错误状态
    else:
        sys.exit(0)  # 分析成功 