import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .repvit import Mlp

class MultiScaleConv(nn.Module):
    def __init__(self, dim, kernel_sizes=[3, 5, 7], ratio=[1, 1, 1]):
        super().__init__()
        self.dim = dim
        
        # Calculate channel splits based on ratio
        total_ratio = sum(ratio)
        self.split_dims = [int(dim * r / total_ratio) for r in ratio]
        # Adjust the last split to ensure we use all channels
        self.split_dims[-1] = dim - sum(self.split_dims[:-1])
        
        # Create convolution blocks for each kernel size
        self.conv_blocks = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # Depthwise convolution with different kernel sizes
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(self.split_dims[i], self.split_dims[i], 
                          kernel_size=k, padding=k//2, groups=self.split_dims[i], bias=True),
                nn.Conv2d(self.split_dims[i], self.split_dims[i], kernel_size=1, bias=True),
            ))
    
    def forward(self, x):
        # Split input along channel dimension
        x_splits = torch.split(x, self.split_dims, dim=1)
        out_splits = []
        
        # Apply each convolution block to its corresponding split
        for i, block in enumerate(self.conv_blocks):
            out_splits.append(block(x_splits[i]))
        
        # Concatenate outputs along channel dimension
        out = torch.cat(out_splits, dim=1)
        return out

class BlockWithMultiScale(nn.Module):
    def __init__(
            self, dim, kernel_size=3, mlp_ratio=4., drop=0., drop_path=0., 
            act_layer=nn.GELU, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.mixer = MultiScaleConv(dim=dim)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class ACRRepViTMultiScale(nn.Module):
    def __init__(
            self, in_chans=3, num_classes=1000, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], 
            kernel_size=3, patch_size=4, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
            layer_scale_init_value=1e-5, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.depths = depths
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=True),
            norm_layer(dims[0]),
        )
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for stage_idx in range(len(depths)):
            for i in range(depths[stage_idx]):
                if i == 0 and stage_idx > 0:  # Downsampling
                    self.blocks.append(
                        nn.Sequential(
                            norm_layer(dims[stage_idx-1]),
                            nn.Conv2d(dims[stage_idx-1], dims[stage_idx], kernel_size=2, stride=2, bias=True),
                        )
                    )
                self.blocks.append(
                    BlockWithMultiScale(
                        dim=dims[stage_idx], kernel_size=kernel_size, mlp_ratio=mlp_ratio, 
                        drop=drop_rate, drop_path=drop_path_rate, 
                        act_layer=nn.GELU, layer_scale_init_value=layer_scale_init_value
                    )
                )
        
        # Classification head
        self.norm = norm_layer(dims[-1])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dims[-1], num_classes, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x.flatten(1)

def acr_repvit_m0_9_multiscale(pretrained=False, **kwargs):
    model = ACRRepViTMultiScale(
        depths=[1, 2, 4, 1], dims=[48, 96, 192, 384], 
        mlp_ratio=4, patch_size=4, **kwargs
    )
    return model 