#!/usr/bin/env python3
# ACR-RepViT训练和评估脚本

import argparse
import datetime
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import model.acr_repvit as acr_repvit

# 分布式训练相关函数
def get_acr_model(model_name, pretrained_path=None, num_classes=1000):
    """
    获取ACR-RepViT模型实例
    
    Args:
        model_name: 模型名称 (acr_repvit_m0_9, acr_repvit_m1_0等)
        pretrained_path: 预训练模型路径
        num_classes: 分类类别数
    
    Returns:
        model: 模型实例
    """
    model = getattr(acr_repvit, model_name)(num_classes=num_classes)
    
    # 如果指定了预训练模型，加载权重
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # 处理不同checkpoint格式
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 处理模型键名不匹配问题
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            
            # 打印加载状态
            print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 层权重")
            
            # 加载权重
            model.load_state_dict(pretrained_dict, strict=False)
        except Exception as e:
            print(f"加载预训练权重时出错: {e}")
    else:
        print("未指定预训练模型或路径不存在，使用随机初始化权重")
    
    return model

def parse_args():
    parser = argparse.ArgumentParser('ACR-RepViT训练和评估脚本', add_help=False)
    # 数据集参数
    parser.add_argument('--dataset', default='imagenet', type=str, help='数据集名称 (imagenet, cifar10, cifar100)')
    parser.add_argument('--data-path', default='/path/to/dataset', type=str, help='数据集路径')
    parser.add_argument('--batch-size', default=128, type=int, help='每个GPU的批次大小')
    parser.add_argument('--num-workers', default=8, type=int, help='数据加载器线程数')
    
    # 模型参数
    parser.add_argument('--model', default='acr_repvit_m0_9', type=str, help='模型名称')
    parser.add_argument('--finetune', default='', help='微调权重路径')
    parser.add_argument('--amp', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='是否使用梯度检查点以节省内存')
    parser.add_argument('--input-size', default=224, type=int, help='输入图像大小')
    parser.add_argument('--num-classes', default=1000, type=int, help='类别数量')
    parser.add_argument('--start-epoch', default=0, type=int, help='开始训练的轮次')
    
    # 训练参数
    parser.add_argument('--epochs', default=300, type=int, help='训练轮次')
    parser.add_argument('--warmup-epochs', default=5, type=int, help='热身轮次')
    parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
    parser.add_argument('--min-lr', default=1e-5, type=float, help='最小学习率')
    parser.add_argument('--weight-decay', default=0.05, type=float, help='权重衰减')
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='标签平滑')
    parser.add_argument('--print-freq', default=10, type=int, help='打印频率')
    parser.add_argument('--save-freq', default=10, type=int, help='保存检查点频率')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--early-stopping', action='store_true', help='启用早停机制')
    parser.add_argument('--patience', default=10, type=int, help='早停耐心值，连续多少个epoch无改进后停止')
    
    # 分布式和硬件参数
    parser.add_argument('--world-size', default=1, type=int, help='分布式训练节点数')
    parser.add_argument('--rank', default=0, type=int, help='分布式训练节点编号')
    parser.add_argument('--dist-url', default='env://', help='分布式URL')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='分布式后端')
    parser.add_argument('--local_rank', default=-1, type=int, help='分布式本地进程编号')
    parser.add_argument('--device', default='cuda', help='设备')
    parser.add_argument('--distributed', action='store_true', help='启用分布式训练')
    
    # 输出参数
    parser.add_argument('--output-dir', default='./output', help='输出目录')
    
    # 测试参数
    parser.add_argument('--test-only', action='store_true', help='仅评估，不训练')
    parser.add_argument('--fuse', action='store_true', help='评估时是否融合模型')
    parser.add_argument('--no-fuse', action='store_true', help='强制不融合模型')
    
    parser.add_argument('--fp16', action='store_true', help='是否使用混合精度训练')
    parser.add_argument('--fp16-scaler', action='store_true', help='使用梯度缩放器防止下溢')
    parser.add_argument('--memory-fraction', type=float, default=0.85, 
                      help='限制CUDA使用的最大内存比例')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2,
                      help='梯度累积步数，用于增大批次大小')
    
    return parser.parse_args()

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('不使用分布式训练')
        args.distributed = False
        return

    args.distributed = True
    
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| 分布式训练初始化 (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    """
    在分布式训练中禁用非主进程的打印
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    """检查分布式训练是否可用且已初始化"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    """获取分布式训练的世界大小"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    """获取当前进程的排名"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

class SmoothedValue(object):
    """
    跟踪一系列值并计算平均值等
    """
    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self):
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

    def synchronize_between_processes(self):
        """分布式训练中同步统计信息"""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

class MetricLogger(object):
    """
    用于记录指标的日志记录器
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """在分布式训练中同步所有进程间的指标"""
        if not is_dist_avail_and_initialized():
            return
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} 总时间: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def train_one_epoch(model, criterion, optimizer, lr_scheduler, 
                   train_loader, device, epoch, args, scaler=None,
                   accumulation_steps=1):
    model.train()
    
    # 统计变量
    loss_sum = 0.0
    correct = 0
    total = 0
    
    # 进度条参数
    num_steps = len(train_loader)
    batch_size = train_loader.batch_size
    samples_per_step = batch_size * accumulation_steps
    
    print(f"Epoch: {epoch}/{args.epochs}, 总批次数: {num_steps}, "
          f"梯度累积步数: {accumulation_steps}, 每步实际样本数: {samples_per_step}")
    
    start_time = time.time()
    
    # 梯度累积迭代
    optimizer.zero_grad()  # 开始前清零梯度
    
    for step, (images, targets) in enumerate(train_loader):
        # 将数据移至GPU
        images, targets = images.to(device), targets.to(device)
        
        # 计算每批次的缩放因子，用于梯度累积
        scaling_factor = 1.0 / accumulation_steps
        
        # 混合精度训练
        if args.fp16:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets) * scaling_factor
            
            if scaler is not None:
                # 使用梯度缩放器
            scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
                    optimizer.zero_grad()
            else:
                # 不使用缩放器的混合精度
                loss.backward()
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
            optimizer.zero_grad()
        else:
            # 标准全精度训练
            outputs = model(images)
            loss = criterion(outputs, targets) * scaling_factor
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # 统计准确率
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 累计损失
        loss_sum += loss.item() * accumulation_steps  # 反向乘以累积步数得到真实损失
        
        # 打印训练进度
        if (step + 1) % 20 == 0 or step == num_steps - 1:
            elapsed = time.time() - start_time
            print(f"批次 [{step+1}/{num_steps}] "
                  f"损失: {loss_sum / (step+1):.4f} "
                  f"准确率: {100 * correct / total:.2f}% "
                  f"用时: {elapsed:.1f}s "
                  f"速度: {(step+1) * batch_size / elapsed:.1f} 样本/秒")
    
    # 确保最后的梯度也被应用（如果批次数不是累积步数的整数倍）
    if (step + 1) % accumulation_steps != 0 and not args.fp16:
            optimizer.step()
        optimizer.zero_grad()
    
    # 更新学习率
    lr_scheduler.step()
    
    # 清理内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 返回平均损失和准确率
    return {
        'loss': loss_sum / num_steps,
        'accuracy': 100 * correct / total
    }

def analyze_model(model, input_size=(1, 3, 224, 224)):
    """分析模型参数量和计算量"""
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 尝试计算FLOPs
    try:
        from thop import profile
        device = next(model.parameters()).device
        input = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(input, ))
        flops_info = f", 计算量: {flops/1e9:.2f}G FLOPs"
    except ImportError:
        print("提示: 安装thop库可以计算FLOPs: pip install thop")
        flops_info = ""
    
    print(f"模型参数量: 总计 {total_params/1e6:.2f}M, 可训练 {trainable_params/1e6:.2f}M{flops_info}")
    return total_params, trainable_params

@torch.no_grad()
def evaluate(model, criterion, data_loader, device, fuse=False):
    """评估模型"""
    # 如果融合，先测试融合前性能再测试融合后性能
    if fuse and hasattr(model, 'fuse_model'):
        # 保存原始模型副本用于对比
        print("测试重参数化融合效果...")
        print("测试融合前性能...")
        before_stats = evaluate(model, criterion, data_loader, device, fuse=False)
        
        print("执行模型重参数化融合...")
        model.eval()
        if hasattr(model, 'module'):
            model.module.fuse_model()
        else:
            model.fuse_model()
        
        print("测试融合后性能...")
        after_stats = evaluate_inner(model, criterion, data_loader, device)
        
        print(f"融合前 vs 融合后: Top-1准确率 {before_stats['acc1']:.2f}% vs {after_stats['acc1']:.2f}%")
        return after_stats
    else:
        return evaluate_inner(model, criterion, data_loader, device)

@torch.no_grad()
def evaluate_inner(model, criterion, data_loader, device):
    """内部评估函数，不处理融合逻辑"""
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # 计算输出
        output = model(images)
        loss = criterion(output, target)

        # 计算准确率
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = target.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(acc5=acc5.item())

    # 同步所有进程的评估指标
    metric_logger.synchronize_between_processes()
    print('* 测试结果: Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} 损失 {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def accuracy(output, target, topk=(1,)):
    """计算前k个准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(model, criterion, data_loader, device, epoch, args):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'测试: [Epoch {epoch}]'
    
    # 统计变量
    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    # 禁用梯度计算以节省内存
    with torch.no_grad():
        for images, target in metric_logger.log_every(data_loader, args.print_freq, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # 混合精度推理
            if args.fp16:
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, target)
            else:
                output = model(images)
                loss = criterion(output, target)
            
            # 计算Top-1和Top-5准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            # 统计
            batch_size = images.shape[0]
            loss_sum += loss.item() * batch_size
            top1_correct += acc1.item() * batch_size / 100
            top5_correct += acc5.item() * batch_size / 100
            total += batch_size
            
            # 记录
            metric_logger.update(loss=loss.item())
            metric_logger.update(acc1=acc1.item())
            metric_logger.update(acc5=acc5.item())
    
    # 清理内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 收集验证结果
    stats = {
        'loss': loss_sum / total,
        'acc1': 100 * top1_correct / total,
        'acc5': 100 * top5_correct / total
    }
    
    return stats

def save_checkpoint(state, save_dir, name):
    """保存检查点"""
    torch.save(state, os.path.join(save_dir, f"{name}.pth"))

def build_transform(is_train, args):
    """构建图像变换"""
    if args.dataset == 'imagenet':
        # ImageNet数据转换
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    elif args.dataset.lower() in ['cifar10', 'cifar100']:
        # CIFAR数据转换
        mean = (0.4914, 0.4822, 0.4465) if args.dataset.lower() == 'cifar10' else (0.5071, 0.4867, 0.4408)
        std = (0.2023, 0.1994, 0.2010) if args.dataset.lower() == 'cifar10' else (0.2675, 0.2565, 0.2761)
        
        if is_train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224),  # 调整为模型输入尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),  # 调整为模型输入尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    return transform

def create_optimizer(args, model):
    """创建优化器"""
    print(f"使用优化器: AdamW, lr={args.lr}, weight_decay={args.weight_decay}")
    
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

def create_scheduler(args, optimizer):
    """创建学习率调度器"""
    print(f"使用学习率调度器: CosineAnnealingLR, T_max={args.epochs}, eta_min={args.min_lr}")
    
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.min_lr
    )

def main():
    args = parse_args()
    
    # 打印基本信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 设置CUDA内存分配比例 - 添加错误处理
    if torch.cuda.is_available() and args.memory_fraction < 1.0:
        try:
            print(f"尝试设置CUDA内存使用上限为GPU总内存的{args.memory_fraction*100:.1f}%")
            # 尝试更安全的方式设置内存限制
            torch.cuda.empty_cache()  # 先清理缓存
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        except RuntimeError as e:
            print(f"设置内存比例时出错: {e}")
            print("继续使用默认内存设置...")
    
    # 启用cuDNN基准模式以提高性能
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 初始化分布式训练
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
        elif args.local_rank != -1:
            args.rank = args.local_rank
            args.world_size = int(os.environ.get("WORLD_SIZE", 1))
            args.gpu = args.local_rank
        else:
            print('不使用分布式训练')
            args.distributed = False
            args.gpu = 0
        
        if args.distributed:
            torch.cuda.set_device(args.gpu)
            args.dist_backend = 'nccl'
            print(f'| 分布式训练初始化 (rank {args.rank}): {args.dist_url}')
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank
            )
            torch.distributed.barrier()
    else:
        args.gpu = 0
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化设备
    if args.gpu is not None:
        print(f"使用GPU: {args.gpu}")
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cpu')
    
    # 创建模型
    print(f"创建模型: {args.model}")
    model = get_acr_model(args.model, args.finetune, args.num_classes)
        
    # 模型参数分析
    analyze_model(model, input_size=(1, 3, args.input_size, args.input_size))
    
    # 将模型移到GPU
    model.to(device)
    
    # 设置分布式数据并行
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # 数据加载
    print("加载数据")
    transform_train = build_transform(is_train=True, args=args)
    transform_test = build_transform(is_train=False, args=args)
    
    # 加载数据集
    if args.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'train'), transform=transform_train)
        test_dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'val'), transform=transform_test)
    
    # 混合精度训练设置
    scaler = None
    if args.fp16:
        if args.fp16_scaler:
            scaler = torch.cuda.amp.GradScaler()
            print("启用带梯度缩放的混合精度训练")
        else:
            print("启用混合精度训练")
    
    # 设置DataLoader
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = SequentialSampler(test_dataset)
    
    # 优化DataLoader的worker数量
    actual_workers = min(args.num_workers, 4)  # 限制worker数量以减少内存使用
    
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=actual_workers,
        pin_memory=True,
        persistent_workers=actual_workers > 0,  # 保持worker进程活跃以减少创建开销
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, sampler=test_sampler,
        batch_size=args.batch_size,  # 验证时使用相同的批次大小
        num_workers=actual_workers,
        pin_memory=True,
        persistent_workers=actual_workers > 0,
        drop_last=False
    )
    
    # 设置优化器
    optimizer = create_optimizer(args, model_without_ddp)
    
    # 设置学习率调度器
    lr_scheduler = create_scheduler(args, optimizer)
    
    # 损失函数
        criterion = nn.CrossEntropyLoss()
    
    # 定期清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 训练初始化
    print(f"开始训练 {args.epochs} 轮...")
    best_acc = 0.0
    start_time = time.time()
    
    # TensorBoard记录器
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    else:
        writer = None
    
    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        # 设置epoch对应的sampler
            if args.distributed:
                train_sampler.set_epoch(epoch)
            
            # 训练一个epoch
            train_stats = train_one_epoch(
            model, criterion, optimizer, lr_scheduler, 
            train_loader, device, epoch, args, scaler,
            accumulation_steps=args.gradient_accumulation_steps
        )
            
        # 验证模型
        test_stats = validate(model, criterion, test_loader, device, epoch, args)
            
        # 更新最佳准确率和保存检查点
        acc1 = test_stats['acc1']
        if acc1 > best_acc:
            best_acc = acc1
            if args.rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_without_ddp.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                }, args.output_dir, 'best')
            print(f"保存最佳检查点，准确率: {best_acc:.2f}%")
        
        # 定期保存检查点
        if args.rank == 0 and (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_without_ddp.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
            }, args.output_dir, f'checkpoint-{epoch:03d}')
            
            # 记录TensorBoard
            if args.rank == 0 and writer is not None:
                writer.add_scalar('train/loss', train_stats['loss'], epoch)
            writer.add_scalar('train/acc1', train_stats['accuracy'], epoch)
                writer.add_scalar('val/loss', test_stats['loss'], epoch)
                writer.add_scalar('val/acc1', test_stats['acc1'], epoch)
                writer.add_scalar('val/acc5', test_stats['acc5'], epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 打印状态
            if args.rank == 0:
            print(f"轮次 {epoch}/{args.epochs-1} 训练损失: {train_stats['loss']:.4f} "
                  f"Top-1: {train_stats['accuracy']:.2f}% "
                      f"测试损失: {test_stats['loss']:.4f} "
                      f"Top-1: {test_stats['acc1']:.2f}% Top-5: {test_stats['acc5']:.2f}%")
                
            # 记录CSV结果
            with open(os.path.join(args.output_dir, 'results.csv'), 'a') as f:
                f.write(f"{epoch},{train_stats['loss']:.4f},{train_stats['accuracy']:.2f},"
                        f"{test_stats['loss']:.4f},{test_stats['acc1']:.2f},{test_stats['acc5']:.2f}\n")
            
    # 训练结束，计算总时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.rank == 0:
        print(f"训练完成，总时间: {total_time_str}")
        print(f"最佳Top-1准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main() 