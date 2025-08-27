import torch
import torch.nn as nn
from timm.models import register_model
from timm.models.layers import SqueezeExcite
from .repvit import _make_divisible, RepViT, RepVGGDW, Residual, Classfier
from .activations import create_activation, create_norm, AdaptiveNorm

class EnhancedConv2d_BN(torch.nn.Module):
    """增强型Conv2d_BN，支持自定义归一化层"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                groups=1, bn_weight_init=1, norm_layer='bn'):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        
        # 使用自定义归一化层
        self.norm = create_norm(norm_layer, out_channels)
        
        # 初始化参数
        if hasattr(self.norm, 'weight') and self.norm.weight is not None:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
        if hasattr(self.norm, 'bias') and self.norm.bias is not None:
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        return self.norm(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """融合卷积和归一化层，用于推理加速"""
        # 如果归一化层是BatchNorm或其子类
        if hasattr(self.norm, 'get_equivalent_bn'):
            bn = self.norm.get_equivalent_bn()
        elif isinstance(self.norm, nn.BatchNorm2d):
            bn = self.norm
        else:
            # 不支持融合的归一化层，直接返回
            return self
            
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = self.conv.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        
        m = torch.nn.Conv2d(w.size(1) * self.conv.groups, w.size(0), 
                          self.conv.kernel_size, stride=self.conv.stride, 
                          padding=self.conv.padding, dilation=self.conv.dilation, 
                          groups=self.conv.groups, bias=True,
                          device=self.conv.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class EnhancedRepVGGDW(torch.nn.Module):
    """增强型RepVGGDW，支持自定义激活函数和归一化层"""
    def __init__(self, channels, norm_layer='bn'):
        super().__init__()
        self.conv = EnhancedConv2d_BN(channels, channels, 3, 1, 1, groups=channels, norm_layer=norm_layer)
        self.conv1 = torch.nn.Conv2d(channels, channels, 1, 1, 0, groups=channels)
        self.norm = create_norm(norm_layer, channels)
    
    def forward(self, x):
        return self.norm((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse(self):
        """融合所有卷积和归一化层，用于推理加速"""
        try:
            conv = self.conv.fuse()
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

            # 如果归一化层是BatchNorm或其子类
            if hasattr(self.norm, 'get_equivalent_bn'):
                bn = self.norm.get_equivalent_bn()
            elif isinstance(self.norm, nn.BatchNorm2d):
                bn = self.norm
            else:
                # 如果不支持融合的归一化层，返回当前融合结果
                return conv
                
            w = bn.weight / (bn.running_var + bn.eps)**0.5
            w = conv.weight * w[:, None, None, None]
            b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / (bn.running_var + bn.eps)**0.5
            
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(b)
            return conv
        except Exception as e:
            print(f"融合EnhancedRepVGGDW时出错: {e}")
            return self


class EnhancedRepViTBlock(nn.Module):
    """增强型RepViTBlock，支持自定义激活函数和归一化层"""
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs,
                act_layer='relu', norm_layer='bn'):
        super(EnhancedRepViTBlock, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        # 创建激活函数
        self.act = create_activation(act_layer, hidden_dim)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                EnhancedConv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp, norm_layer=norm_layer),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                EnhancedConv2d_BN(inp, oup, kernel_size=1, stride=1, padding=0, norm_layer=norm_layer)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    EnhancedConv2d_BN(oup, 2 * oup, 1, 1, 0, norm_layer=norm_layer),
                    self.act,  # 使用自定义激活函数
                    # pw-linear
                    EnhancedConv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0, norm_layer=norm_layer),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                EnhancedRepVGGDW(inp, norm_layer=norm_layer),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    EnhancedConv2d_BN(inp, hidden_dim, 1, 1, 0, norm_layer=norm_layer),
                    self.act,  # 使用自定义激活函数
                    # pw-linear
                    EnhancedConv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0, norm_layer=norm_layer),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))
        
    @torch.no_grad()
    def fuse(self):
        """融合所有可融合层，用于推理加速"""
        try:
            # 尝试融合token_mixer
            for i, m in enumerate(self.token_mixer):
                if hasattr(m, 'fuse'):
                    self.token_mixer[i] = m.fuse()
                    
            # 尝试融合channel_mixer
            if hasattr(self.channel_mixer, 'fuse'):
                self.channel_mixer = self.channel_mixer.fuse()
                
            return self
        except Exception as e:
            print(f"融合EnhancedRepViTBlock时出错: {e}")
            return self


class EnhancedBN_Linear(torch.nn.Module):
    """增强型BN_Linear，支持自定义归一化层"""
    def __init__(self, in_features, out_features, bias=True, std=0.02, norm_layer='bn'):
        super().__init__()
        if norm_layer == 'bn':
            self.norm = torch.nn.BatchNorm1d(in_features)
        elif norm_layer == 'ln':
            self.norm = torch.nn.LayerNorm(in_features)
        elif norm_layer == 'adaptive':
            # 1D版本的自适应归一化
            self.norm = torch.nn.Sequential(
                torch.nn.BatchNorm1d(in_features),
                torch.nn.LayerNorm(in_features)
            )
            self.norm_weight = nn.Parameter(torch.ones(1) * 0.5)
        else:
            raise ValueError(f"分类器不支持的归一化层: {norm_layer}")
            
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        
        # 初始化参数
        from timm.models.vision_transformer import trunc_normal_
        trunc_normal_(self.linear.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.linear.bias, 0)
            
        self.norm_type = norm_layer

    def forward(self, x):
        if self.norm_type == 'adaptive':
            bn_out = self.norm[0](x)
            ln_out = self.norm[1](x)
            x = self.norm_weight * bn_out + (1 - self.norm_weight) * ln_out
        else:
            x = self.norm(x)
        return self.linear(x)

    @torch.no_grad()
    def fuse(self):
        """融合归一化和线性层，用于推理加速"""
        try:
            if self.norm_type == 'bn':
                bn = self.norm
                w = bn.weight / (bn.running_var + bn.eps)**0.5
                b = bn.bias - self.norm.running_mean * self.norm.weight / (bn.running_var + bn.eps)**0.5
                w = self.linear.weight * w[None, :]
                if self.linear.bias is None:
                    b = b @ self.linear.weight.T
                else:
                    b = (self.linear.weight @ b[:, None]).view(-1) + self.linear.bias
                
                m = torch.nn.Linear(w.size(1), w.size(0), device=self.linear.weight.device)
                m.weight.data.copy_(w)
                m.bias.data.copy_(b)
                return m
            else:
                # 不支持融合的归一化层，直接返回
                return self
        except Exception as e:
            print(f"融合EnhancedBN_Linear时出错: {e}")
            return self


class EnhancedClassfier(nn.Module):
    """增强型分类器，支持自定义归一化层"""
    def __init__(self, dim, num_classes, distillation=True, norm_layer='bn'):
        super().__init__()
        self.classifier = EnhancedBN_Linear(dim, num_classes, norm_layer=norm_layer) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = EnhancedBN_Linear(dim, num_classes, norm_layer=norm_layer) if num_classes > 0 else torch.nn.Identity()

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
        """融合分类器，用于推理加速"""
        try:
            if hasattr(self.classifier, 'fuse'):
                self.classifier = self.classifier.fuse()
            if self.distillation and hasattr(self.classifier_dist, 'fuse'):
                self.classifier_dist = self.classifier_dist.fuse()
            return self
        except Exception as e:
            print(f"融合EnhancedClassfier时出错: {e}")
            return self


class EnhancedRepViT(nn.Module):
    """增强型RepViT模型，支持自定义激活函数和归一化层"""
    def __init__(self, cfgs, num_classes=1000, distillation=False, 
                act_layer='relu', norm_layer='bn'):
        super(EnhancedRepViT, self).__init__()
        # 配置
        self.cfgs = cfgs
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        # 第一层
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(
            EnhancedConv2d_BN(3, input_channel // 2, 3, 2, 1, norm_layer=norm_layer), 
            create_activation(act_layer, input_channel // 2),
            EnhancedConv2d_BN(input_channel // 2, input_channel, 3, 2, 1, norm_layer=norm_layer))
        
        # 创建主干网络
        layers = [patch_embed]
        
        # 创建每个块
        block = EnhancedRepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, 
                              act_layer=act_layer, norm_layer=norm_layer))
            input_channel = output_channel
            
        self.features = nn.Sequential(*layers)
        
        # 分类头
        self.classifier = EnhancedClassfier(output_channel, num_classes, distillation, norm_layer=norm_layer)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # 全局平均池化
        x = torch.mean(x, dim=(2, 3))
        # 分类
        x = self.classifier(x)
        return x
        
    @torch.no_grad()
    def fuse(self):
        """融合所有可融合层，用于推理加速"""
        import copy
        model_fused = copy.deepcopy(self)
        
        # 融合特征提取器
        for i, m in enumerate(model_fused.features):
            if hasattr(m, 'fuse'):
                model_fused.features[i] = m.fuse()
        
        # 融合分类器
        if hasattr(model_fused.classifier, 'fuse'):
            model_fused.classifier = model_fused.classifier.fuse()
            
        return model_fused


@register_model
def enhanced_repvit_m0_9(pretrained=False, pretrained_path=None, num_classes=1000, 
                       act_layer='prelu2d', norm_layer='bn', **kwargs):
    """创建增强型RepViT-M0.9模型"""
    # 配置
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  512, 0, 1, 2],
        [3,   2,  512, 1, 1, 1],
        [3,   2,  512, 0, 1, 1],
    ]
    
    # 读取蒸馏设置
    distillation = kwargs.pop('distillation', False)
    
    # 创建模型
    model = EnhancedRepViT(cfgs, num_classes=num_classes, distillation=distillation,
                         act_layer=act_layer, norm_layer=norm_layer)
    
    # 加载预训练权重
    if pretrained or pretrained_path:
        if pretrained_path is None:
            pretrained_path = "pretrained/repvit_m0_9_distill_300e.pth"
            
        # 加载官方预训练权重
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model' in checkpoint:
            weights = checkpoint['model']
        elif 'state_dict' in checkpoint:
            weights = checkpoint['state_dict']
        else:
            weights = checkpoint
            
        # 创建新的权重字典
        new_weights = {}
        
        # 处理权重名称不匹配的问题
        for k, v in weights.items():
            if k.startswith('features.0.0'):
                # patch_embed 第一层
                new_k = k.replace('features.0.0', 'features.0.0')
                new_weights[new_k] = v
            elif k.startswith('features.0.1'):
                # GELU 激活函数会被替换，先保存以防后续处理
                continue
            elif k.startswith('features.0.2'):
                # patch_embed 第二层
                new_k = k.replace('features.0.2', 'features.0.2')
                new_weights[new_k] = v
            elif k.startswith('features.'):
                # 主干网络层
                new_weights[k] = v
            elif k.startswith('classifier.'):
                # 分类器，需要特殊处理
                if k == 'classifier.l.weight':
                    new_weights['classifier.classifier.linear.weight'] = v
                elif k == 'classifier.l.bias':
                    new_weights['classifier.classifier.linear.bias'] = v
                elif k == 'classifier.bn.weight':
                    new_weights['classifier.classifier.norm.weight'] = v
                elif k == 'classifier.bn.bias':
                    new_weights['classifier.classifier.norm.bias'] = v
                elif k == 'classifier.bn.running_mean':
                    new_weights['classifier.classifier.norm.running_mean'] = v
                elif k == 'classifier.bn.running_var':
                    new_weights['classifier.classifier.norm.running_var'] = v
                elif k == 'classifier.bn.num_batches_tracked':
                    new_weights['classifier.classifier.norm.num_batches_tracked'] = v
                # 处理蒸馏分类器
                elif k == 'classifier_dist.l.weight':
                    new_weights['classifier.classifier_dist.linear.weight'] = v
                elif k == 'classifier_dist.l.bias':
                    new_weights['classifier.classifier_dist.linear.bias'] = v
                elif k == 'classifier_dist.bn.weight':
                    new_weights['classifier.classifier_dist.norm.weight'] = v
                elif k == 'classifier_dist.bn.bias':
                    new_weights['classifier.classifier_dist.norm.bias'] = v
                elif k == 'classifier_dist.bn.running_mean':
                    new_weights['classifier.classifier_dist.norm.running_mean'] = v
                elif k == 'classifier_dist.bn.running_var':
                    new_weights['classifier.classifier_dist.norm.running_var'] = v
                elif k == 'classifier_dist.bn.num_batches_tracked':
                    new_weights['classifier.classifier_dist.norm.num_batches_tracked'] = v
            else:
                # 其他权重直接复制
                new_weights[k] = v
                
        # 加载处理后的权重
        msg = model.load_state_dict(new_weights, strict=False)
        print(f"加载预训练权重: {pretrained_path}")
        print(f"缺失键: {len(msg.missing_keys)}, 未使用键: {len(msg.unexpected_keys)}")
    
    return model


@register_model
def enhanced_repvit_m0_9_mixed(pretrained=False, pretrained_path=None, num_classes=1000, **kwargs):
    """创建使用混合激活函数的增强型RepViT-M0.9模型"""
    return enhanced_repvit_m0_9(pretrained, pretrained_path, num_classes, 
                              act_layer='mixed', norm_layer='bn', **kwargs)


@register_model
def enhanced_repvit_m0_9_adaptive(pretrained=False, pretrained_path=None, num_classes=1000, **kwargs):
    """创建使用自适应归一化的增强型RepViT-M0.9模型"""
    return enhanced_repvit_m0_9(pretrained, pretrained_path, num_classes, 
                              act_layer='prelu2d', norm_layer='adaptive', **kwargs)


@register_model
def enhanced_repvit_m0_9_full(pretrained=False, pretrained_path=None, num_classes=1000, **kwargs):
    """创建同时使用混合激活函数和自适应归一化的增强型RepViT-M0.9模型"""
    return enhanced_repvit_m0_9(pretrained, pretrained_path, num_classes, 
                              act_layer='mixed', norm_layer='adaptive', **kwargs)


@register_model
def enhanced_repvit_m0_9_relu(pretrained=False, pretrained_path=None, num_classes=1000, **kwargs):
    """创建使用简单ReLU激活的增强型RepViT-M0.9模型"""
    return enhanced_repvit_m0_9(pretrained, pretrained_path, num_classes, 
                              act_layer='relu', norm_layer='bn', **kwargs) 