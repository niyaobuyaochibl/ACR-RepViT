#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACR-RepViT 模型深入分析脚本
实现注意力可视化、特征空间可视化和参数效率分析
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from PIL import Image
import argparse
import time
from pathlib import Path

# 导入模型定义
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.acr_repvit import acr_repvit_m0_9, acr_repvit_m0_9_cbam, acr_repvit_m0_9_multiscale

class GradCam:
    """简化版Grad-CAM实现"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        self.register_hooks()
        
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activation = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(self, input_tensor, target_category=None):
        # 前向传播
        model_output = self.model(input_tensor)
        
        if target_category is None:
            target_category = torch.argmax(model_output, dim=1)
        
        # 清除现有梯度
        self.model.zero_grad()
        
        # 反向传播
        model_output[0, target_category].backward(retain_graph=True)
        
        # 获取梯度和激活
        gradient = self.gradient[0].cpu().data.numpy()
        activation = self.activation[0].cpu().data.numpy()
        
        # 计算权重
        weights = np.mean(gradient, axis=(1, 2))
        
        # 生成CAM
        cam = np.zeros(activation.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activation[i]
        
        # 应用ReLU
        cam = np.maximum(cam, 0)
        
        # 归一化
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def load_model(model_name, checkpoint_path, num_classes=1000):
    """加载训练好的模型"""
    if model_name == 'acr_repvit_m0_9':
        model = acr_repvit_m0_9(num_classes=num_classes)
    elif model_name == 'acr_repvit_m0_9_cbam':
        model = acr_repvit_m0_9_cbam(num_classes=num_classes)
    elif model_name == 'acr_repvit_m0_9_multiscale':
        model = acr_repvit_m0_9_multiscale(num_classes=num_classes)
    else:
        raise ValueError(f"未知模型名称: {model_name}")
    
    print(f"加载模型: {model_name}")
    if os.path.exists(checkpoint_path):
        print(f"从 {checkpoint_path} 加载权重")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 处理DDP状态字典
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"警告: 检查点 {checkpoint_path} 不存在")
    
    return model

def preprocess_image(img_path, transform=None):
    """预处理图像以进行模型推理"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    return input_tensor, img

def visualize_attention(models, img_path, output_dir, device):
    """可视化不同模型的注意力图"""
    # 准备图像
    input_tensor, img = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)
    img_array = np.array(img.resize((224, 224))) / 255.0
    
    # 为每个模型创建Grad-CAM
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(5 * (len(models) + 1), 5))
    
    # 显示原始图像
    axes[0].imshow(img_array)
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    # 为每个模型生成热图
    for i, (model_name, model) in enumerate(models.items()):
        model.eval()
        
        # 获取目标层 (通常是最后一个卷积层或注意力层)
        if model_name == 'acr_repvit_m0_9':
            target_layer = model.stages[-1][-1].token_mixer
        elif model_name == 'acr_repvit_m0_9_cbam':
            target_layer = model.stages[-1][-1].token_mixer
        elif model_name == 'acr_repvit_m0_9_multiscale':
            target_layer = model.stages[-1][-1].token_mixer
        
        # 创建Grad-CAM
        grad_cam = GradCam(model, target_layer)
        
        # 生成热图
        cam = grad_cam.generate_cam(input_tensor)
        
        # 调整热图大小以匹配原始图像
        cam_resized = transforms.Resize((224, 224))(torch.tensor(cam).unsqueeze(0).unsqueeze(0))
        cam_resized = cam_resized.squeeze().numpy()
        
        # 叠加热图到原始图像
        heatmap = np.uint8(255 * cam_resized)
        heatmap = plt.cm.jet(heatmap)[:, :, :3]
        superimposed_img = heatmap * 0.4 + img_array
        
        # 显示图像
        axes[i+1].imshow(superimposed_img)
        axes[i+1].set_title(f"{model_name}")
        axes[i+1].axis('off')
    
    # 保存图像
    plt.tight_layout()
    img_name = os.path.basename(img_path)
    plt.savefig(os.path.join(output_dir, f"attention_map_{img_name}"), dpi=300)
    plt.close()

def extract_features(model, data_loader, device, max_samples=1000):
    """提取特征向量用于t-SNE可视化"""
    features = []
    labels = []
    model.eval()
    
    with torch.no_grad():
        for images, targets in data_loader:
            if len(features) * images.size(0) >= max_samples:
                break
                
            images = images.to(device)
            
            # 提取特征 (去掉最后的分类头)
            if hasattr(model, 'forward_features'):
                feat = model.forward_features(images)
            else:
                # 假设模型最后一层是分类头
                modules = list(model.children())[:-1]
                temp_model = nn.Sequential(*modules)
                feat = temp_model(images)
            
            # 将特征展平
            feat = feat.view(feat.size(0), -1).cpu().numpy()
            features.append(feat)
            labels.append(targets.numpy())
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels

def plot_tsne(models, data_loader, output_dir, device, perplexity=30, n_iter=3000):
    """生成t-SNE可视化"""
    all_features = {}
    all_labels = None
    
    # 提取所有模型的特征
    for model_name, model in models.items():
        print(f"为{model_name}提取特征...")
        features, labels = extract_features(model, data_loader, device)
        all_features[model_name] = features
        if all_labels is None:
            all_labels = labels
    
    # 选择一些类别进行可视化 (太多类别会导致可视化混乱)
    num_classes = min(10, len(np.unique(all_labels)))
    selected_classes = np.unique(all_labels)[:num_classes]
    mask = np.isin(all_labels, selected_classes)
    
    # 为每个模型应用t-SNE
    for model_name, features in all_features.items():
        print(f"计算{model_name}的t-SNE...")
        
        # 选择数据子集
        subset_features = features[mask]
        subset_labels = all_labels[mask]
        
        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embedded = tsne.fit_transform(subset_features)
        
        # 绘制t-SNE结果
        plt.figure(figsize=(12, 10))
        
        # 使用seaborn为不同类别着色
        palette = sns.color_palette("bright", num_classes)
        scatter = sns.scatterplot(
            x=embedded[:, 0], y=embedded[:, 1], 
            hue=subset_labels, palette=palette, 
            s=50, alpha=0.8, edgecolor='none'
        )
        
        # 添加标题和图例
        plt.title(f"{model_name}特征空间分布", fontsize=16)
        plt.xlabel("t-SNE维度1", fontsize=14)
        plt.ylabel("t-SNE维度2", fontsize=14)
        
        # 移动图例到图外
        plt.legend(title="类别", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"tsne_{model_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

def analyze_parameter_importance(models, test_loader, output_dir, device):
    """分析不同模型的参数数量与性能关系"""
    results = {}
    
    for model_name, model in models.items():
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 评估准确率
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        results[model_name] = {
            'params': total_params,
            'accuracy': accuracy
        }
    
    # 绘制参数量-准确率图
    plt.figure(figsize=(10, 6))
    
    model_names = list(results.keys())
    params = [results[m]['params'] / 1000000 for m in model_names]  # 转换为百万
    accuracies = [results[m]['accuracy'] for m in model_names]
    
    # 绘制散点图
    for i, model_name in enumerate(model_names):
        plt.scatter(params[i], accuracies[i], s=100, label=model_name)
    
    # 连接点以形成Pareto边界
    plt.plot(params, accuracies, 'k--', alpha=0.5)
    
    plt.title("模型参数量 vs 准确率", fontsize=16)
    plt.xlabel("参数量 (百万)", fontsize=14)
    plt.ylabel("Top-1准确率 (%)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "params_vs_accuracy.png"), dpi=300)
    plt.close()
    
    # 打印详细结果
    print("\n参数效率分析结果:")
    print("=" * 50)
    print(f"{'模型名称':<25} {'参数量':<15} {'准确率':<10} {'参数效率':<15}")
    print("-" * 50)
    for model_name in model_names:
        params = results[model_name]['params']
        accuracy = results[model_name]['accuracy']
        efficiency = accuracy / (params / 1000000)  # 每百万参数的准确率
        print(f"{model_name:<25} {params:<15,} {accuracy:<10.2f}% {efficiency:<15.2f}")
    print("=" * 50)
    
    return results

def analyze_inference_time(models, device, output_dir, batch_size=1, input_size=(3, 224, 224), iterations=100):
    """分析不同模型的推理时间"""
    results = {}
    
    # 准备随机输入
    input_tensor = torch.randn(batch_size, *input_size).to(device)
    
    for model_name, model in models.items():
        model.eval()
        model.to(device)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # 计时
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = (end_time - start_time) / iterations * 1000  # 转换为毫秒
        results[model_name] = inference_time
    
    # 绘制推理时间对比图
    plt.figure(figsize=(10, 6))
    
    model_names = list(results.keys())
    times = [results[m] for m in model_names]
    
    # 创建条形图
    bars = plt.bar(model_names, times, color=['blue', 'orange', 'green'])
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} ms', ha='center', va='bottom')
    
    plt.title("不同模型的推理时间对比", fontsize=16)
    plt.ylabel("每张图像推理时间 (ms)", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time.png"), dpi=300)
    plt.close()
    
    # 打印详细结果
    print("\n推理性能分析结果:")
    print("=" * 50)
    print(f"{'模型名称':<25} {'推理时间 (ms)':<20}")
    print("-" * 50)
    for model_name in model_names:
        print(f"{model_name:<25} {results[model_name]:<20.2f}")
    print("=" * 50)
    
    return results

def create_validation_loader(data_path, batch_size=64, num_workers=4):
    """创建ImageNet验证集数据加载器"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        transform=transform
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return val_loader

def main():
    parser = argparse.ArgumentParser(description='ACR-RepViT模型分析工具')
    parser.add_argument('--data-path', type=str, default='/datasets/ImageNet', help='ImageNet数据路径')
    parser.add_argument('--output-dir', type=str, default='./analysis_output', help='分析输出路径')
    parser.add_argument('--batch-size', type=int, default=64, help='批量大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--sample-images', type=str, default='./sample_images', help='示例图像目录')
    parser.add_argument('--original-checkpoint', type=str, help='原始ACR-RepViT模型检查点')
    parser.add_argument('--cbam-checkpoint', type=str, help='CBAM变体模型检查点')
    parser.add_argument('--multiscale-checkpoint', type=str, help='多尺度卷积变体模型检查点')
    parser.add_argument('--attention-only', action='store_true', help='只生成注意力热图')
    args = parser.parse_args()
    
    # 检查并创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建验证数据加载器
    val_loader = create_validation_loader(args.data_path, args.batch_size, args.num_workers)
    
    # 加载模型
    models = {}
    if args.original_checkpoint:
        models['acr_repvit_m0_9'] = load_model('acr_repvit_m0_9', args.original_checkpoint).to(device)
    if args.cbam_checkpoint:
        models['acr_repvit_m0_9_cbam'] = load_model('acr_repvit_m0_9_cbam', args.cbam_checkpoint).to(device)
    if args.multiscale_checkpoint:
        models['acr_repvit_m0_9_multiscale'] = load_model('acr_repvit_m0_9_multiscale', args.multiscale_checkpoint).to(device)
    
    # 检查样本图像目录
    if os.path.exists(args.sample_images):
        sample_imgs = [os.path.join(args.sample_images, f) for f in os.listdir(args.sample_images) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
        if sample_imgs:
            print(f"找到 {len(sample_imgs)} 张样本图像")
            # 生成注意力热图
            for img_path in sample_imgs:
                print(f"处理图像: {img_path}")
                visualize_attention(models, img_path, args.output_dir, device)
        else:
            print(f"在 {args.sample_images} 中没有找到图像")
    
    # 如果指定只生成注意力热图，则到此结束
    if args.attention_only:
        print("仅生成注意力热图完成")
        return
    
    # 生成t-SNE可视化
    plot_tsne(models, val_loader, args.output_dir, device)
    
    # 分析参数效率
    analyze_parameter_importance(models, val_loader, args.output_dir, device)
    
    # 分析推理时间
    analyze_inference_time(models, device, args.output_dir)
    
    print("分析完成")

if __name__ == '__main__':
    main() 