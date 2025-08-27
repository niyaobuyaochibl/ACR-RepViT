"""
ACR-RepViT Model Implementations

This module contains all model variants used in our paper:
"Optimal Complexity in Lightweight Vision Transformers: A Trade-off Analysis 
between Representational Power and Optimization Efficiency"

Available models:
- repvit_m0_9: Baseline RepViT-M0.9
- acr_repvit_m0_9_cbam: RepViT-M0.9 with CBAM attention
- acr_repvit_m0_9_ca: RepViT-M0.9 with Channel Attention only  
- acr_repvit_m0_9_sa: RepViT-M0.9 with Spatial Attention only
- acr_repvit_m0_9_multiscale: RepViT-M0.9 with Multi-scale convolution
"""

from timm.models import register_model

# Import all model implementations
from .repvit import RepViT
from .acr_repvit_cbam import ACRRepViTCBAM
from .acr_repvit_ca import ACRRepViTCA  
from .acr_repvit_sa import ACRRepViTSA
from .acr_repvit_multiscale import ACRRepViTMultiScale

# Model configurations
model_configs = {
    'repvit_m0_9': {
        'depths': [1, 2, 4, 1],
        'dims': [48, 96, 192, 384],
        'kernel_size': 3,
        'patch_size': 4,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'layer_scale_init_value': 1e-5,
    },
    'acr_repvit_m0_9_cbam': {
        'depths': [1, 2, 4, 1],
        'dims': [48, 96, 192, 384],
        'kernel_size': 3,
        'patch_size': 4,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'layer_scale_init_value': 1e-5,
        'cbam_ratio': 8,
        'cbam_kernel_size': 7,
    },
    'acr_repvit_m0_9_ca': {
        'depths': [1, 2, 4, 1],
        'dims': [48, 96, 192, 384],
        'kernel_size': 3,
        'patch_size': 4,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'layer_scale_init_value': 1e-5,
        'ca_ratio': 8,
    },
    'acr_repvit_m0_9_sa': {
        'depths': [1, 2, 4, 1],
        'dims': [48, 96, 192, 384],
        'kernel_size': 3,
        'patch_size': 4,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'layer_scale_init_value': 1e-5,
        'sa_kernel_size': 7,
    },
    'acr_repvit_m0_9_multiscale': {
        'depths': [1, 2, 4, 1],
        'dims': [48, 96, 192, 384],
        'kernel_size': 3,
        'patch_size': 4,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'layer_scale_init_value': 1e-5,
        'multiscale_kernels': [3, 5],
    },
}

# Register models with timm
@register_model
def repvit_m0_9(pretrained=False, **kwargs):
    """RepViT-M0.9 baseline model"""
    model = RepViT(**model_configs['repvit_m0_9'], **kwargs)
    return model

@register_model  
def acr_repvit_m0_9_cbam(pretrained=False, **kwargs):
    """ACR-RepViT-M0.9 with CBAM attention"""
    model = ACRRepViTCBAM(**model_configs['acr_repvit_m0_9_cbam'], **kwargs)
    return model

@register_model
def acr_repvit_m0_9_ca(pretrained=False, **kwargs):
    """ACR-RepViT-M0.9 with Channel Attention only"""
    model = ACRRepViTCA(**model_configs['acr_repvit_m0_9_ca'], **kwargs)
    return model

@register_model
def acr_repvit_m0_9_sa(pretrained=False, **kwargs):
    """ACR-RepViT-M0.9 with Spatial Attention only"""
    model = ACRRepViTSA(**model_configs['acr_repvit_m0_9_sa'], **kwargs)
    return model

@register_model
def acr_repvit_m0_9_multiscale(pretrained=False, **kwargs):
    """ACR-RepViT-M0.9 with Multi-scale convolution"""
    model = ACRRepViTMultiScale(**model_configs['acr_repvit_m0_9_multiscale'], **kwargs)
    return model

# Export model creation functions
__all__ = [
    'RepViT', 'ACRRepViTCBAM', 'ACRRepViTCA', 'ACRRepViTSA', 'ACRRepViTMultiScale',
    'repvit_m0_9', 'acr_repvit_m0_9_cbam', 'acr_repvit_m0_9_ca', 
    'acr_repvit_m0_9_sa', 'acr_repvit_m0_9_multiscale',
    'model_configs'
]