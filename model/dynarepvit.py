import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_
from timm.models import register_model

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self

# 动态通道门控模块
class DynamicChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d_BN(in_channels, max(in_channels//reduction, 4), 1),
            nn.GELU(),
            Conv2d_BN(max(in_channels//reduction, 4), in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gates = self.gate(x)
        return x * gates
    
    @torch.no_grad()
    def fuse(self):
        return self

# 动态深度控制模块
class DynamicDepthController(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.controller = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d_BN(in_channels, 8, 1),
            nn.GELU(),
            Conv2d_BN(8, 1, 1),
            nn.Sigmoid()
        )
        self.threshold = 0.5
        self.temperature = 1.0
        
    def forward(self, x):
        # 训练阶段：软门控
        if self.training:
            decision = self.controller(x)
            # 使用温度系数控制软硬门控的过渡
            gumbel_decision = torch.sigmoid((decision - self.threshold) / self.temperature)
            return x, gumbel_decision
        # 推理阶段：硬门控
        else:
            decision = self.controller(x)
            return x, (decision > self.threshold).float()
        
    @torch.no_grad()
    def fuse(self):
        return self

# 自适应核大小选择
class AdaptiveRepVGGDW(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv3 = Conv2d_BN(dim, dim, 3, 1, 1, groups=dim)
        self.conv5 = Conv2d_BN(dim, dim, 5, 1, 2, groups=dim)
        self.conv1 = torch.nn.Conv2d(dim, dim, 1, 1, 0, groups=dim)
        self.dim = dim
        self.bn = torch.nn.BatchNorm2d(dim)
        
        # 核大小选择预测器
        self.kernel_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2d_BN(dim, 4, 1),
            nn.GELU(),
            Conv2d_BN(4, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 预测核大小权重
        if self.training:
            # 训练时: 同时计算所有分支
            kernel_weights = self.kernel_predictor(x)
            out = kernel_weights[:,0:1,0,0].view(-1,1,1,1) * self.conv3(x) + \
                  kernel_weights[:,1:2,0,0].view(-1,1,1,1) * self.conv5(x) + \
                  self.conv1(x)
            return self.bn(out)
        else:
            # 推理时: 仅计算选定的分支
            kernel_weights = self.kernel_predictor(x)
            batch_size = x.size(0)
            out = torch.zeros_like(x)
            
            # 对每个样本单独处理
            for i in range(batch_size):
                if kernel_weights[i,0] > kernel_weights[i,1]:
                    out[i] = self.conv3(x[i:i+1])
                else:
                    out[i] = self.conv5(x[i:i+1])
                    
            return self.bn(out + self.conv1(x))
    
    @torch.no_grad()
    def fuse(self):
        # 推理优化时仅保留一个分支
        conv = self.conv3.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

# 全局复杂度感知控制器
class ComplexityController(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            Conv2d_BN(3, 16, 3, 2, 1),
            nn.GELU(),
            Conv2d_BN(16, 32, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 3)  # 输出三个复杂度级别的得分
        )
        
    def forward(self, x):
        return F.softmax(self.feature_extractor(x), dim=1)

# 动态RepViT块
class DynaRepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(DynaRepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)
        
        # 动态深度控制器 (仅对恒等映射块使用)
        self.use_depth_control = self.identity
        if self.use_depth_control:
            self.depth_controller = DynamicDepthController(inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            # 使用自适应核大小选择模块
            self.token_mixer = nn.Sequential(
                AdaptiveRepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            # 添加动态通道门控
            self.dynamic_channel_gate = DynamicChannelGate(inp, reduction=16)
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        # 使用动态深度控制
        if self.use_depth_control and self.training:
            x, gate_value = self.depth_controller(x)
            # 软门控：结合跳过和处理的特征
            token_out = self.token_mixer(x)
            if hasattr(self, 'dynamic_channel_gate'):
                token_out = self.dynamic_channel_gate(token_out)
            channel_out = self.channel_mixer(token_out)
            # 线性加权组合
            return gate_value * channel_out + (1 - gate_value) * x
        elif self.use_depth_control and not self.training:
            x, gate_value = self.depth_controller(x)
            # 硬门控：如果低于阈值，直接跳过
            # 使用mean()代替item()，处理批量数据
            if gate_value.mean() < 0.5:
                return x
            else:
                token_out = self.token_mixer(x)
                if hasattr(self, 'dynamic_channel_gate'):
                    token_out = self.dynamic_channel_gate(token_out)
                return self.channel_mixer(token_out)
        else:
            token_out = self.token_mixer(x)
            if hasattr(self, 'dynamic_channel_gate'):
                token_out = self.dynamic_channel_gate(token_out)
            return self.channel_mixer(token_out)

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier

class DynaRepViT(nn.Module):
    def __init__(self, cfgs, num_classes=1000, distillation=False):
        super(DynaRepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # 全局复杂度控制器
        self.complexity_controller = ComplexityController()

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                           Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = DynaRepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(output_channel, num_classes, distillation)

        # 计算资源感知损失的权重
        self.resource_weight = 0.01
        
    def forward(self, x):
        # 预测图像复杂度
        complexity_score = self.complexity_controller(x)
        
        # 根据复杂度分数调整处理策略
        for i, f in enumerate(self.features):
            x = f(x)
            
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        cls_output = self.classifier(x)
        
        if self.training:
            # 计算资源感知损失
            # 复杂度越低，我们希望激活的通道和层越少
            resource_loss = complexity_score[:, 0] * 0.2 + complexity_score[:, 1] * 0.5 + complexity_score[:, 2] * 1.0
            resource_loss = resource_loss.mean() * self.resource_weight
            
            # 存储资源损失作为模型属性，而不是返回它
            self.current_resource_loss = resource_loss
            return cls_output
        else:
            return cls_output

    def update_temperature(self, epoch, max_epoch):
        # 随着训练进行，降低温度系数，使决策更加离散
        temperature = max(0.1, 1.0 - 0.9 * epoch / max_epoch)
        for m in self.modules():
            if isinstance(m, DynamicDepthController):
                m.temperature = temperature

# 定义不同大小的DynaRepViT模型
@register_model
def dynarepvit_m0_9(pretrained=False, num_classes=1000, distillation=False):
    """
    动态版本的RepViT-M0.9
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 384, 0, 1, 2],
        [3,   2, 384, 1, 1, 1],
        [3,   2, 384, 0, 1, 1]
    ]
    return DynaRepViT(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def dynarepvit_m1_1(pretrained=False, num_classes=1000, distillation=False):
    """
    动态版本的RepViT-M1.1
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return DynaRepViT(cfgs, num_classes=num_classes, distillation=distillation) 