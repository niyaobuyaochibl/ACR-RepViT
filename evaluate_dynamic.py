import os
import torch
import time
import argparse
import numpy as np
from tqdm import tqdm
from timm.models import create_model
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# 设置评估参数
parser = argparse.ArgumentParser(description='Evaluate Dynamic RepViT Model')
parser.add_argument('--data-path', default='/root/autodl-tmp/data', type=str, help='dataset path')
parser.add_argument('--batch-size', default=64, type=int, help='batch size')
parser.add_argument('--input-size', default=224, type=int, help='image input size')
parser.add_argument('--original-model', default='repvit_m0_9', type=str, help='original model name')
parser.add_argument('--dynamic-model', default='simple_dynarepvit_m0_9', type=str, help='dynamic model name')
parser.add_argument('--original-ckpt', default='pretrained/repvit_m0_9_distill_300e.pth', 
                    type=str, help='original model checkpoint')
parser.add_argument('--dynamic-ckpt', default='checkpoints/simple_dynarepvit_m0_9/checkpoint_best.pth',
                    type=str, help='dynamic model checkpoint')
parser.add_argument('--num-samples', default=1000, type=int, help='number of samples for speed test')
parser.add_argument('--visualize', action='store_true', help='visualize gate activations')
parser.add_argument('--device', default='cuda', help='device to use for testing')

def get_transform(input_size):
    # 创建验证集转换
    return transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

def evaluate_accuracy(model, data_loader, device):
    """评估模型准确率"""
    model.eval()
    
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            # 计算Top-1和Top-5准确率
            _, pred = outputs.topk(5, 1, True, True)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred))
            correct_1 += correct[:, 0].sum().item()
            correct_5 += correct.sum().item()
            total += targets.size(0)
    
    top1_acc = 100 * correct_1 / total
    top5_acc = 100 * correct_5 / total
    
    return top1_acc, top5_acc

def benchmark_speed(model, input_size, iterations=100, device='cuda'):
    """测量模型推理速度"""
    model.eval()
    
    # 准备输入
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 测量时间
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations * 1000  # 毫秒
    return avg_time

def analyze_gate_activations(model, data_loader, device, num_samples=100):
    """分析门控激活情况"""
    model.eval()
    
    # 收集门控激活值
    gate_activations = []
    gate_hooks = []
    
    # 注册钩子
    def hook_fn(self, input, output):
        # 收集门控值
        gate_activations.append(output.detach().cpu().numpy())
        return output
    
    for name, module in model.named_modules():
        if 'channel_gate' in name:
            gate_hooks.append(module.register_forward_hook(hook_fn))
    
    # 运行模型收集激活值
    count = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            _ = model(images)
            count += images.size(0)
            if count >= num_samples:
                break
    
    # 移除钩子
    for hook in gate_hooks:
        hook.remove()
    
    # 分析门控激活
    if gate_activations:
        # 计算平均门控值
        avg_activations = []
        for activations in gate_activations:
            # 对批次和空间维度取平均
            avg_act = np.mean(activations, axis=(0, 2, 3))
            avg_activations.append(avg_act)
        
        # 合并所有样本的平均值
        all_avg = np.concatenate(avg_activations, axis=0)
        mean_activations = np.mean(all_avg, axis=0)
        
        # 计算通道开启比例
        channel_sparsity = np.mean(all_avg < 0.1) * 100
        
        return mean_activations, channel_sparsity
    else:
        return None, 0

def main():
    args = parser.parse_args()
    
    # 创建数据集
    val_transform = get_transform(args.input_size)
    val_dataset = ImageFolder(os.path.join(args.data_path, 'val'), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print("\n" + "="*50)
    print("评估动态RepViT模型性能")
    print("="*50)
    
    # Step 1: 加载并评估原始模型
    print("\n1. 评估原始RepViT模型...")
    original_model = create_model(args.original_model, pretrained=False)
    
    original_checkpoint = torch.load(args.original_ckpt, map_location='cpu')
    if 'model' in original_checkpoint:
        original_checkpoint = original_checkpoint['model']
    original_model.load_state_dict(original_checkpoint)
    original_model = original_model.to(args.device)
    
    # 测量原始模型准确率
    original_top1, original_top5 = evaluate_accuracy(original_model, val_loader, args.device)
    print(f"原始模型精度: Top-1={original_top1:.2f}%, Top-5={original_top5:.2f}%")
    
    # 测量原始模型速度
    original_speed = benchmark_speed(original_model, args.input_size, device=args.device)
    print(f"原始模型推理速度: {original_speed:.2f}ms/image")
    
    # Step 2: 加载并评估动态模型
    print("\n2. 评估动态RepViT模型...")
    dynamic_model = create_model(args.dynamic_model, pretrained=False)
    
    dynamic_checkpoint = torch.load(args.dynamic_ckpt, map_location='cpu')
    if 'model' in dynamic_checkpoint:
        dynamic_checkpoint = dynamic_checkpoint['model']
    dynamic_model.load_state_dict(dynamic_checkpoint)
    dynamic_model = dynamic_model.to(args.device)
    
    # 测量动态模型准确率
    dynamic_top1, dynamic_top5 = evaluate_accuracy(dynamic_model, val_loader, args.device)
    print(f"动态模型精度: Top-1={dynamic_top1:.2f}%, Top-5={dynamic_top5:.2f}%")
    
    # 测量动态模型速度
    dynamic_speed = benchmark_speed(dynamic_model, args.input_size, device=args.device)
    print(f"动态模型推理速度: {dynamic_speed:.2f}ms/image")
    
    # Step 3: 分析动态模型门控情况
    if args.visualize:
        print("\n3. 分析动态模型门控激活情况...")
        mean_activations, channel_sparsity = analyze_gate_activations(
            dynamic_model, val_loader, args.device, args.num_samples)
        
        if mean_activations is not None:
            print(f"平均通道开启比例: {100 - channel_sparsity:.2f}%")
            print(f"通道稀疏度: {channel_sparsity:.2f}%")
    
    # 性能对比
    print("\n" + "="*50)
    print("性能对比")
    print("="*50)
    print(f"精度差异: {dynamic_top1 - original_top1:.2f}% (Top-1), {dynamic_top5 - original_top5:.2f}% (Top-5)")
    speed_improvement = (original_speed - dynamic_speed) / original_speed * 100
    print(f"速度提升: {speed_improvement:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 