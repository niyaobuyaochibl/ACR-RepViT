import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PReLU2d(nn.Module):
    """二维可学习参数ReLU，为每个通道学习不同斜率"""
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1) * 0.25)
        
    def forward(self, x):
        # 安全处理：创建适合当前输入尺寸的权重
        b, c, h, w = x.shape
        
        if c != self.num_channels:
            # 完全重新创建权重张量而不是修改已有参数
            temp_weight = torch.ones(1, c, 1, 1, device=x.device) * 0.25
            positive = torch.maximum(torch.zeros_like(x), x)
            negative = torch.minimum(torch.zeros_like(x), x)
            return positive + temp_weight * negative
        else:
            # 使用预定义的权重
            positive = torch.maximum(torch.zeros_like(x), x)
            negative = torch.minimum(torch.zeros_like(x), x)
            return positive + self.weight * negative
    
    def get_equivalent_relu(self):
        """返回等效的ReLU层，用于模型转换"""
        return nn.ReLU(inplace=True)


class MixedActivation(nn.Module):
    """组合多种激活函数的混合激活"""
    def __init__(self, num_channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1) * 0.5)
        self.beta = nn.Parameter(torch.ones(1, num_channels, 1, 1) * 0.5)
        
    def forward(self, x):
        # 获取输入张量的实际通道数
        b, c, h, w = x.shape
        
        # 处理通道数不匹配的情况
        if self.alpha.shape[1] != c:
            # 创建新参数而不是尝试expand
            alpha = torch.ones(1, c, 1, 1, device=x.device) * 0.5
            beta = torch.ones(1, c, 1, 1, device=x.device) * 0.5
        else:
            alpha = self.alpha
            beta = self.beta
            
        relu_part = F.relu(x)
        gelu_part = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        swish_part = x * torch.sigmoid(x)
        
        return alpha * relu_part + (1 - alpha) * (beta * gelu_part + (1 - beta) * swish_part)
    
    def get_equivalent_relu(self):
        """返回等效的ReLU层，用于模型转换"""
        return nn.ReLU(inplace=True)


class AdaptiveNorm(nn.Module):
    """自适应选择不同归一化方法的层"""
    def __init__(self, num_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.ln = nn.GroupNorm(1, num_channels)  # 等同于LayerNorm
        self.weight = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        # 检查输入通道数是否与预期不符
        if x.shape[1] != self.bn.num_features:
            # 创建新的BN和LN层以适应当前输入
            bn_output = F.batch_norm(
                x, 
                running_mean=None, 
                running_var=None, 
                weight=None,
                bias=None, 
                training=self.training, 
                momentum=0.1, 
                eps=self.bn.eps
            )
            
            # 尝试使用GroupNorm，如果不可行，使用InstanceNorm
            try:
                ln_output = F.group_norm(x, 1, None, None, self.ln.eps)
            except:
                ln_output = F.instance_norm(x, None, None, self.ln.eps)
        else:
            bn_output = self.bn(x)
            ln_output = self.ln(x)
            
        return self.weight * bn_output + (1 - self.weight) * ln_output
    
    def get_equivalent_bn(self):
        """返回等效的BatchNorm层，用于模型转换"""
        return self.bn


# 激活函数工厂，根据名称创建激活函数
def create_activation(name, channels):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'prelu2d':
        return PReLU2d(channels)
    elif name == 'mixed':
        return MixedActivation(channels)
    elif name == 'silu' or name == 'swish':
        return nn.SiLU(inplace=True)
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"未知的激活函数: {name}")


# 归一化层工厂，根据名称创建归一化层
def create_norm(name, channels):
    if name == 'bn':
        return nn.BatchNorm2d(channels)
    elif name == 'ln':
        return nn.GroupNorm(1, channels)  # 等同于LayerNorm
    elif name == 'adaptive':
        return AdaptiveNorm(channels)
    else:
        raise ValueError(f"未知的归一化层: {name}") 