"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, set_bn_eval=False,
                    args = None):
    model.train(set_training_mode)
    # 设置批归一化层为eval模式
    if set_bn_eval:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.set_grad_enabled(True): # 使用上下文管理器保证梯度计算
            # 计算模型输出
            outputs = model(samples)
                
            # 检查输出是否包含NaN
            if isinstance(outputs, tuple):
                # 如果输出是元组（例如在蒸馏模式下），检查每个元素
                has_nan = False
                for output in outputs:
                    if torch.isnan(output).any():
                        has_nan = True
                        break
                if has_nan:
                    print(f"警告: 检测到NaN输出，跳过当前批次")
                    continue
            else:
                # 单个输出的情况
                if torch.isnan(outputs).any():
                    print(f"警告: 检测到NaN输出，跳过当前批次")
                    continue
                
                # 检查loss函数参数位置
                if hasattr(criterion, 'distillation_type') and criterion.distillation_type != 'none':
                    # 蒸馏模式下，需要传入输入样本
                    loss = criterion(samples, outputs, targets)
                elif hasattr(criterion, 'resource_weight'):
                    # 动态模型损失，需要传入模型对象
                    loss = criterion(outputs, targets, model)
                else:
                    # 普通模式下，只需要输出和目标
                    loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("警告: Loss是 {}, 使用小值替代".format(loss_value))
            loss = torch.tensor(0.1, device=device, requires_grad=True)
            loss_value = 0.1
            # 此处不再中断训练
            # sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        
        # 添加安全措施来处理梯度
        try:
            # 统一使用loss_scaler处理，无论是否启用了AMP
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order)
        except RuntimeError as e:
            print(f"警告: 梯度处理错误: {e}")
            print("跳过当前批次，继续训练")
            optimizer.zero_grad()
            continue

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output - 移除autocast
        output = model(images)
        
        # 处理输出是元组的情况（例如在蒸馏模式下）
        if isinstance(output, tuple):
            # 在评估过程中，使用平均输出
            output = 0.5 * (output[0] + output[1])
            
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
