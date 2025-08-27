"""
Implements the knowledge distillation loss, proposed in deit
"""
import torch
from torch.nn import functional as F


# 添加自定义的SoftTargetCrossEntropy，避免类型错误
class CustomSoftTargetCrossEntropy(torch.nn.Module):
    """
    自定义软目标交叉熵，正确处理mixup产生的软标签
    """
    def forward(self, x, target):
        # 处理输入是元组的情况（对于返回(output, dist_output)的模型）
        if isinstance(x, tuple):
            x = x[0]  # 使用主输出进行损失计算
            
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        # 处理当outputs是元组的情况（蒸馏模式下的分类器返回元组）
        if isinstance(outputs, tuple):
            outputs_x, outputs_dist = outputs
        else:
            outputs_x = outputs
            outputs_dist = None
            
        # 处理动态模型的资源感知损失
        resource_loss = 0.0
        
        # 计算基础分类损失
        base_loss = self.base_criterion(outputs_x, labels)
        
        if self.distillation_type == 'none':
            return base_loss + resource_loss
        
        # 如果使用蒸馏，保持原有逻辑并加入资源损失
        if not self.teacher_model:
            return base_loss + resource_loss
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            
        if self.distillation_type == 'soft':
            T = self.tau
            # 使用outputs_dist进行蒸馏，如果为None则使用outputs_x
            distill_input = outputs_dist if outputs_dist is not None else outputs_x
            # 使用KL散度计算软目标蒸馏损失
            distillation_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(distill_input / T, dim=1),
                torch.nn.functional.softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=False
            ) * (T * T)
        elif self.distillation_type == 'hard':
            # 使用outputs_dist进行蒸馏，如果为None则使用outputs_x
            distill_input = outputs_dist if outputs_dist is not None else outputs_x
            # 硬目标：教师模型的预测类别
            hard_target = teacher_outputs.argmax(dim=1)
            distillation_loss = torch.nn.functional.cross_entropy(distill_input, hard_target)
        
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha + resource_loss
        return loss

# 新增用于动态RepViT的损失函数
class DynaDistillationLoss(torch.nn.Module):
    """
    针对动态RepViT的蒸馏损失，支持资源感知训练
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, resource_weight: float = 0.01):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.resource_weight = resource_weight
        
    def forward(self, inputs, outputs, labels, model=None):
        """
        处理动态模型的输出，包括分类结果和资源损失
        Args:
            inputs: 输入样本
            outputs: 模型输出
            labels: 真实标签
            model: 模型对象，用于获取资源损失
        """
        # 处理当outputs是元组的情况（蒸馏模式下的分类器返回元组）
        if isinstance(outputs, tuple):
            outputs_x, outputs_dist = outputs
        else:
            outputs_x = outputs
            outputs_dist = None
            
        # 计算基础分类损失
        base_loss = self.base_criterion(outputs_x, labels)
        
        # 从模型中获取资源损失（如果存在）
        resource_loss = 0.0
        if model is not None and hasattr(model, 'current_resource_loss'):
            resource_loss = model.current_resource_loss * self.resource_weight
        
        # 如果不使用蒸馏，直接返回基础损失和资源损失
        if self.distillation_type == 'none' or not self.teacher_model:
            return base_loss + resource_loss
        
        # 计算蒸馏损失
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        
        if self.distillation_type == 'soft':
            T = self.tau
            # 使用outputs_dist进行蒸馏，如果为None则使用outputs_x
            distill_input = outputs_dist if outputs_dist is not None else outputs_x
            
            # 使用KL散度计算软目标蒸馏损失
            distillation_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(distill_input / T, dim=1),
                torch.nn.functional.softmax(teacher_outputs / T, dim=1),
                reduction='batchmean',
                log_target=False
            ) * (T * T)
        elif self.distillation_type == 'hard':
            # 使用outputs_dist进行蒸馏，如果为None则使用outputs_x
            distill_input = outputs_dist if outputs_dist is not None else outputs_x
            
            # 硬目标：教师模型的预测类别
            hard_target = teacher_outputs.argmax(dim=1)
            distillation_loss = torch.nn.functional.cross_entropy(distill_input, hard_target)
        
        # 组合所有损失
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha + resource_loss
        return loss

class DynaLoss(torch.nn.Module):
    """
    专为动态模型设计的损失函数，包含资源感知损失
    """
    def __init__(self, base_criterion: torch.nn.Module, resource_weight: float = 0.005):
        super().__init__()
        self.base_criterion = base_criterion
        self.resource_weight = resource_weight
        
    def forward(self, outputs, labels, model=None):
        """
        处理动态模型的输出，包括分类结果和资源损失
        Args:
            outputs: 模型输出
            labels: 真实标签
            model: 模型对象，用于获取资源损失
        """
        # 处理当outputs是元组的情况
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        # 计算基础分类损失
        base_loss = self.base_criterion(outputs, labels)
        
        # 从模型中获取资源损失（如果存在）
        resource_loss = 0.0
        if model is not None and hasattr(model, 'current_resource_loss'):
            resource_loss = model.current_resource_loss * self.resource_weight
            
            # 防止资源损失过大导致训练不稳定
            if resource_loss > 0.1 * base_loss:
                resource_loss = 0.1 * base_loss
        
        # 组合所有损失
        loss = base_loss + resource_loss
        return loss
