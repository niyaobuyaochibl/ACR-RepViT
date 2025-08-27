import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from timm.models import create_model
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='评估动态RepViT模型性能')
    parser.add_argument('--data-path', type=str, default='/root/autodl-tmp/data/val', help='验证集路径')
    parser.add_argument('--batch-size', type=int, default=64, help='批量大小')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/simple_dynarepvit_m0_9_final/checkpoint_best.pth', help='模型检查点路径')
    parser.add_argument('--input-size', type=int, default=224, help='输入图像大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 创建模型
    model = create_model('simple_dynarepvit_m0_9', pretrained=False)
    
    # 加载训练好的权重
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    
    # 创建数据变换
    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载验证数据集
    dataset = ImageFolder(args.data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 评估精度
    correct_1 = 0
    correct_5 = 0
    total = 0
    
    # 记录推理时间
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i == 0:  # 预热
                _ = model(images.to(device))
                
            # 计时
            start_time = time.time()
            outputs = model(images.to(device))
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 更新统计信息
            batch_size = images.size(0)
            total_time += (end_time - start_time)
            total_samples += batch_size
            
            # 计算准确率
            _, pred = outputs.topk(5, 1, True, True)
            labels = labels.to(device)
            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()
            
            # 更新统计数据
            correct_1 += correct[:, 0].sum().item()
            correct_5 += correct.sum().item()
            total += batch_size
            
            if i % 10 == 0:
                print(f'Batch: {i}/{len(dataloader)} - Top-1: {correct_1/total*100:.2f}%, Top-5: {correct_5/total*100:.2f}%')
    
    # 计算最终结果
    top1_acc = correct_1 / total * 100
    top5_acc = correct_5 / total * 100
    avg_time = total_time / total_samples * 1000  # 毫秒
    
    print(f'\n最终结果:')
    print(f'Top-1 准确率: {top1_acc:.2f}%')
    print(f'Top-5 准确率: {top5_acc:.2f}%')
    print(f'平均推理时间: {avg_time:.2f}毫秒/样本')
    
    # 获取门控激活率
    if hasattr(model, 'get_gates_status'):
        gates_info = model.get_gates_status()
        activation_rate = gates_info.get('activation_rate', 1.0)
        print(f'平均通道激活率: {activation_rate*100:.2f}%')
        print(f'理论计算量减少: {(1.0-activation_rate)*100:.2f}%')
        
        # 可视化门控分布（可选）
        gate_values = []
        for key, value in gates_info.items():
            if key != 'activation_rate':
                gate_values.extend(value.flatten())
        
        if gate_values:
            plt.figure(figsize=(10, 6))
            plt.hist(gate_values, bins=50, alpha=0.7)
            plt.title('门控值分布')
            plt.xlabel('门控值')
            plt.ylabel('频率')
            plt.savefig('gate_distribution.png')
            print('门控分布可视化已保存为 gate_distribution.png')

if __name__ == '__main__':
    main()
