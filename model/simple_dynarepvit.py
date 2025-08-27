import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from timm.models.layers import SqueezeExcite
from .repvit import RepViT, RepViTBlock, _make_divisible, Conv2d_BN, BN_Linear, Classfier
from .fixed_config import fixed_cfgs_m0_9
import numpy as np


class ChannelGate(nn.Module):
    """非常简单的通道门控机制"""
    def __init__(self, channels, reduction=16, min_threshold=0.1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
        # 设置可调节的门控阈值
        self.min_threshold = min_threshold
        
        # 初始化为接近全开状态
        nn.init.zeros_(self.gate[3].weight)  # 将最后的卷积层初始化为零
        nn.init.ones_(self.gate[4].weight)   # BN的缩放系数为1
        nn.init.constant_(self.gate[4].bias, 5.0)  # 大偏置，使sigmoid输出接近1
        
    def forward(self, x):
        y = self.avg_pool(x)
        gate = self.gate(y)
        
        # 安全措施：确保门控值不会低于设定阈值，防止灾难性崩溃
        if self.training:
            gate = gate.clamp(min=self.min_threshold)
            
        # 安全措施：检测并修复NaN值
        if torch.isnan(gate).any():
            gate = torch.ones_like(gate)
        
        return x * gate


class SimpleDynaRepViTBlock(nn.Module):
    """包含动态通道门控的简化块"""
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, 
                 enable_dynamic=True):
        super().__init__()
        self.identity = stride == 1 and inp == oup
        self.enable_dynamic = enable_dynamic and self.identity
        
        # 复用RepViTBlock的实现
        self.repvit_block = RepViTBlock(inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs)
        
        # 只为恒等映射块添加通道门控
        if self.enable_dynamic:
            self.channel_gate = ChannelGate(inp, reduction=8)
            
    def set_gating_threshold(self, threshold):
        """设置门控阈值"""
        if self.enable_dynamic and hasattr(self, 'channel_gate'):
            self.channel_gate.min_threshold = threshold
            
    def forward(self, x):
        # 非恒等映射块，或未启用动态特性
        if not self.identity or not self.enable_dynamic:
            return self.repvit_block(x)
        
        # 恒等映射块 + 动态门控
        output = self.repvit_block(x)
        
        # 应用动态通道门控
        output = self.channel_gate(output)
        
        # 统计门控值
        if hasattr(self.channel_gate, 'gate'):
            with torch.no_grad():
                gate_values = self.channel_gate.gate(self.channel_gate.avg_pool(output))
                if not hasattr(self, 'gate_stats'):
                    self.gate_stats = gate_values.detach().mean(dim=0)
                else:
                    self.gate_stats = self.gate_stats * 0.9 + gate_values.detach().mean(dim=0) * 0.1
                
        return output
    
    @torch.no_grad()
    def fuse(self):
        # 推理时模型合并
        return self.repvit_block.fuse()


class SimpleDynaRepViT(nn.Module):
    """简化的动态RepViT模型"""
    def __init__(self, cfgs, num_classes=1000, distillation=False, enable_dynamic=True):
        super().__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.enable_dynamic = enable_dynamic
        self.distillation = distillation
        self.sparsity_target = 0.2  # 设置期望的门控稀疏度
        self.current_resource_loss = 0.0  # 用于跟踪当前资源损失
        
        # 训练进度跟踪
        self.current_epoch = 0
        self.total_epochs = 30  # 默认总训练轮次

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                           Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        
        # building inverted residual blocks
        block = SimpleDynaRepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            # 使用 _make_divisible 保证通道数是8的倍数
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs, 
                                enable_dynamic=enable_dynamic))
            input_channel = output_channel
            
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(output_channel, num_classes, distillation)
        
        # 初始化权重
        self._init_weights()
                    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def freeze_backbone(self, train_gate_only=True):
        """冻结主干网络，仅训练门控机制"""
        # 记录训练前参数数量
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        for name, param in self.named_parameters():
            if train_gate_only:
                # 只有包含"gate"的参数才训练
                if 'gate' not in name:
                    param.requires_grad = False
                else:
                    print(f"将训练参数: {name}")
            else:
                param.requires_grad = True
        
        # 记录训练后参数数量
        gate_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"冻结主干网络完成! 总参数: {total_params}, 门控参数: {gate_params}, 冻结比例: {100 * (1 - gate_params / total_params):.2f}%")
        return self
        
    def update_epoch_info(self, current_epoch, total_epochs):
        """更新当前训练轮次信息，用于动态调整资源损失权重"""
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
    def _compute_resource_loss(self):
        """计算资源使用损失，鼓励通道门控稀疏性"""
        gate_values = []
        # 收集所有动态块的门控统计信息
        for module in self.modules():
            if isinstance(module, SimpleDynaRepViTBlock) and module.enable_dynamic:
                if hasattr(module, 'gate_stats'):
                    gate_values.append(module.gate_stats.mean())
        
        if not gate_values:
            return torch.tensor(0.0, device=self.features[0][0].c.weight.device)
        
        # 计算平均门控值
        avg_gate = torch.stack(gate_values).mean()
        
        # 目标门控值为(1-稀疏度)，设置较宽松的目标
        target = 1.0 - self.sparsity_target * 0.5  # 减少目标稀疏度为原来的一半
        
        # 简单L1损失
        loss = torch.abs(avg_gate - target) 
        
        # 检测到NaN则返回零损失
        if torch.isnan(loss):
            return torch.tensor(0.0, device=avg_gate.device)
        
        # 根据训练阶段动态调整权重
        progress = min(1.0, self.current_epoch / self.total_epochs)
        
        # 训练初期低权重，后期高权重
        base_weight = 0.0001
        max_weight = 0.001
        dynamic_weight = base_weight + (max_weight - base_weight) * progress
            
        return loss * dynamic_weight
                    
    def forward(self, x):
        for f in self.features:
            x = f(x)
            
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        outputs = self.classifier(x)
        
        # 计算资源损失
        if self.training and self.enable_dynamic:
            self.current_resource_loss = self._compute_resource_loss()
        else:
            self.current_resource_loss = 0.0
        
        # 确保当distillation=False时，直接返回单一输出而非元组
        if isinstance(outputs, tuple) and not self.distillation:
            return outputs[0]
            
        return outputs
        
    def load_pretrained(self, checkpoint_path):
        """从预训练模型加载权重"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 如果是带有model键的字典，获取model部分
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        # 转换RepViT模型中的blocks键到features键
        new_state_dict = {}
        for k, v in checkpoint.items():
            # 重新映射blocks到features
            if k.startswith('blocks.'):
                # blocks.0 -> features.1 (因为features.0是patch_embed)
                block_id = int(k.split('.')[1])
                new_k = k.replace(f'blocks.{block_id}', f'features.{block_id+1}')
                new_state_dict[new_k] = v
            # 合适的keys直接复制
            elif k in ['patch_embed.0.c.weight', 'patch_embed.0.bn.weight', 'patch_embed.0.bn.bias', 
                      'patch_embed.0.bn.running_mean', 'patch_embed.0.bn.running_var', 
                      'patch_embed.0.bn.num_batches_tracked',
                      'patch_embed.2.c.weight', 'patch_embed.2.bn.weight', 'patch_embed.2.bn.bias',
                      'patch_embed.2.bn.running_mean', 'patch_embed.2.bn.running_var',
                      'patch_embed.2.bn.num_batches_tracked',
                      'classifier.0.weight', 'classifier.0.bias']:
                # 这些键直接对应到features.0
                if k.startswith('patch_embed'):
                    parts = k.split('.')
                    # patch_embed.0.* -> features.0.0.*
                    # patch_embed.2.* -> features.0.2.*
                    new_k = f'features.0.{parts[1]}.{".".join(parts[2:])}'
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
                    
        # 加载处理过的权重
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        
        print(f"预训练模型加载完成。缺失键: {len(missing)}, 未使用键: {len(unexpected)}")
        return self

    def get_gates_status(self):
        """返回所有门控的状态统计信息，用于分析"""
        gates_info = {}
        active_channels = 0
        total_channels = 0
        
        try:
            # 收集每个块的门控统计
            for i, module in enumerate(self.modules()):
                if isinstance(module, SimpleDynaRepViTBlock) and hasattr(module, 'enable_dynamic') and module.enable_dynamic:
                    if hasattr(module, 'gate_stats') and module.gate_stats is not None:
                        try:
                            # 转换为numpy数组以便分析
                            if torch.is_tensor(module.gate_stats):
                                gate_vals = module.gate_stats.detach().cpu().numpy().flatten()
                            else:
                                print(f"警告: 块 {i} 的门控统计不是张量")
                                continue
                                
                            # 检查nan和inf
                            if np.isnan(gate_vals).any() or np.isinf(gate_vals).any():
                                print(f"警告: 块 {i} 的门控统计包含NaN或Inf值，将被替换为1")
                                gate_vals = np.ones_like(gate_vals)
                                
                            gates_info[f'block_{i}'] = gate_vals
                            
                            # 计算激活率
                            active = (gate_vals > 0.05).sum()  # 阈值为0.05
                            total = gate_vals.size
                            
                            active_channels += active
                            total_channels += total
                            
                            # 添加每个块的激活率
                            gates_info[f'block_{i}_active_rate'] = float(active) / float(total)
                        except Exception as e:
                            print(f"处理块 {i} 的门控统计时出错: {e}")
                    elif hasattr(module, 'channel_gate'):
                        # 如果没有统计信息，尝试通过前向传播生成
                        try:
                            # 创建一个伪输入
                            device = next(module.parameters()).device
                            dummy_input = torch.ones(1, module.repvit_block.conv.in_channels, 8, 8, device=device)
                            
                            # 保存当前训练状态并切换到评估
                            training = module.training
                            module.eval()
                            
                            # 前向传播
                            with torch.no_grad():
                                _ = module.channel_gate(dummy_input)
                                if hasattr(module.channel_gate, 'gate'):
                                    gate_vals = module.channel_gate.gate(module.channel_gate.avg_pool(dummy_input)).detach().cpu().numpy().flatten()
                                    gates_info[f'block_{i}'] = gate_vals
                                    
                                    # 计算激活率
                                    active = (gate_vals > 0.05).sum()
                                    total = gate_vals.size
                                    
                                    active_channels += active
                                    total_channels += total
                                    
                                    # 添加每个块的激活率
                                    gates_info[f'block_{i}_active_rate'] = float(active) / float(total)
                            
                            # 恢复训练状态
                            module.train(training)
                        except Exception as e:
                            print(f"为块 {i} 生成门控统计时出错: {e}")
            
            # 计算整体激活率和计算量节省估计
            if total_channels > 0:
                activation_rate = active_channels / total_channels
                gates_info['activation_rate'] = activation_rate
                gates_info['compute_saving_estimate'] = 1.0 - activation_rate
                
                # 计算激活率分布统计
                block_rates = [v for k, v in gates_info.items() if k.endswith('_active_rate')]
                if block_rates:
                    gates_info['mean_block_active_rate'] = sum(block_rates) / len(block_rates)
                    gates_info['min_block_active_rate'] = min(block_rates)
                    gates_info['max_block_active_rate'] = max(block_rates)
            else:
                # 如果没有找到有效的通道信息，提供默认值
                gates_info['activation_rate'] = 1.0
                gates_info['compute_saving_estimate'] = 0.0
                gates_info['mean_block_active_rate'] = 1.0
                gates_info['min_block_active_rate'] = 1.0
                gates_info['max_block_active_rate'] = 1.0
                
                print("警告: 未能找到有效的门控统计信息，使用默认值")
        
        except Exception as e:
            import traceback
            print(f"收集门控统计时出现错误: {e}")
            traceback.print_exc()
            
            # 提供默认值
            gates_info['activation_rate'] = 1.0
            gates_info['compute_saving_estimate'] = 0.0
        
        return gates_info
            
    @torch.no_grad()
    def fuse(self):
        """融合模型权重以加速推理"""
        for m in self.modules():
            if hasattr(m, 'fuse'):
                m.fuse()
        return self


@register_model
def simple_dynarepvit_m0_9(pretrained=True, pretrained_path=None, num_classes=1000, **kwargs):
    """
    创建简化动态M0.9模型
    """
    # 使用修复后的配置
    cfgs = fixed_cfgs_m0_9
    
    # 读取蒸馏设置，如果提供了该参数
    distillation = kwargs.pop('distillation', False)
    
    model = SimpleDynaRepViT(cfgs, num_classes=num_classes, distillation=distillation)
    
    # 默认加载预训练模型
    if pretrained or pretrained_path:
        # 从官方预训练模型加载权重
        if pretrained_path is None:
            pretrained_path = "pretrained/repvit_m0_9_distill_300e.pth"
        model.load_pretrained(pretrained_path)
        print(f"简化动态模型已加载预训练权重: {pretrained_path}")
        
    return model


@register_model
def simple_dynarepvit_m1_1(pretrained=True, pretrained_path=None, num_classes=1000, **kwargs):
    """
    创建简化动态M1.1模型
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
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    
    # 读取蒸馏设置，如果提供了该参数
    distillation = kwargs.pop('distillation', False)
    
    model = SimpleDynaRepViT(cfgs, num_classes=num_classes, distillation=distillation)
    
    # 默认加载预训练模型
    if pretrained or pretrained_path:
        # 从官方预训练模型加载权重
        if pretrained_path is None:
            pretrained_path = "pretrained/repvit_m1_1_distill_300e.pth"
        model.load_pretrained(pretrained_path)
        print(f"简化动态模型已加载预训练权重: {pretrained_path}")
        
    return model 