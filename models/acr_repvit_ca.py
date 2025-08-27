import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .repvit import Mlp, RepMixer

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class BlockWithCA(nn.Module):
    def __init__(
            self, dim, kernel_size=3, mlp_ratio=4., drop=0., drop_path=0., 
            act_layer=nn.GELU, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.mixer = RepMixer(dim=dim, kernel_size=kernel_size)
        self.ca = ChannelAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        mixer_out = self.mixer(self.norm1(x))
        ca_out = mixer_out * self.ca(mixer_out)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * ca_out)
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class ACRRepViTCA(nn.Module):
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
                    BlockWithCA(
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

def acr_repvit_m0_9_ca(pretrained=False, **kwargs):
    model = ACRRepViTCA(
        depths=[1, 2, 4, 1], dims=[48, 96, 192, 384], 
        mlp_ratio=4, patch_size=4, **kwargs
    )
    return model 