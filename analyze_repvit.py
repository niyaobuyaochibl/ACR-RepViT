import torch
from model.repvit import repvit_m0_9

# 加载原始RepViT-M0.9模型
model = repvit_m0_9(pretrained=False)

# 加载预训练权重
checkpoint = torch.load('pretrained/repvit_m0_9_distill_300e.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)

# 分析模型结构
print("=== RepViT-M0.9 模型通道配置分析 ===")
print(f"总层数: {len(model.features)}")

# 分析每个块的通道配置
for i, block in enumerate(model.features):
    if i == 0:  # patch_embed
        print(f"Layer {i} (Patch Embed): in=3, out={list(block[-1].modules())[1].out_channels}")
        continue
        
    if hasattr(block, 'token_mixer') and hasattr(block, 'channel_mixer'):
        in_ch = 0
        out_ch = 0
        hidden_ch = 0
        stride = 1
        
        # 获取输入通道
        if hasattr(block.token_mixer[0], 'c'):
            in_ch = block.token_mixer[0].c.in_channels
        elif hasattr(block.token_mixer[0], 'conv'):
            in_ch = block.token_mixer[0].conv.in_channels
            
        # 获取输出通道
        if hasattr(block, 'channel_mixer') and hasattr(block.channel_mixer, 'm'):
            if len(block.channel_mixer.m) > 0:
                out_ch = in_ch  # 常规块，输出=输入
                if hasattr(block.token_mixer[0], 'conv') and hasattr(block.token_mixer[2], 'c'):
                    out_ch = block.token_mixer[2].c.out_channels  # 下采样块
                    
                # 获取隐藏通道
                if len(block.channel_mixer.m) > 0:
                    hidden_ch = block.channel_mixer.m[0].c.out_channels
                    
        # 判断是否为下采样块
        if hasattr(block.token_mixer[0], 'c'):
            stride = block.token_mixer[0].c.stride[0]
        elif hasattr(block.token_mixer[0], 'conv'):
            stride = block.token_mixer[0].conv.stride[0]
            
        print(f"Layer {i}: in={in_ch}, out={out_ch}, hidden={hidden_ch}, expansion={hidden_ch/in_ch:.1f}x, stride={stride}")
    else:
        print(f"Layer {i}: 结构不明确")
        
# 显示权重结构示例
print("\n=== 预训练权重结构示例 ===")
for i, (k, v) in enumerate(list(checkpoint['model'].items())[:20]):
    if 'channel_mixer' in k:
        print(f"{k}: {v.shape}")
        
print("\n=== 关键参数检查 ===")
# 检查特定层的通道扩展
for layer_idx in [3, 7, 8, 17, 18, 19]:
    if layer_idx >= len(model.features):
        continue
        
    block = model.features[layer_idx]
    
    # 获取输入通道
    in_ch = None
    if hasattr(block.token_mixer[0], 'c'):
        in_ch = block.token_mixer[0].c.in_channels
    elif hasattr(block.token_mixer[0], 'conv'):
        in_ch = block.token_mixer[0].conv.in_channels
    
    if in_ch is not None and hasattr(block.channel_mixer, 'm') and len(block.channel_mixer.m) > 2:
        hidden_ch = block.channel_mixer.m[0].c.out_channels
        out_ch = block.channel_mixer.m[2].c.out_channels
        print(f"Layer {layer_idx} detailed: input={in_ch}, hidden={hidden_ch}, output={out_ch}, ratio={hidden_ch/in_ch:.1f}")
        
        # 获取权重形状
        prefix = f"features.{layer_idx}.channel_mixer.m"
        key1 = f"{prefix}.0.c.weight"
        key2 = f"{prefix}.2.c.weight"
        if key1 in checkpoint['model'] and key2 in checkpoint['model']:
            shape1 = checkpoint['model'][key1].shape
            shape2 = checkpoint['model'][key2].shape
            print(f"  - {key1}: {shape1}")
            print(f"  - {key2}: {shape2}") 