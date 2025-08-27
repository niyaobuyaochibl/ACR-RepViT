#!/bin/bash
# 简化版ACR-RepViT训练脚本

# 设置数据和预训练模型路径
DATA_PATH="/root/autodl-tmp/cifar10"
PRETRAINED="/root/autodl-tmp/repvit_m0_9_pretrain.pth"

# 创建输出目录
mkdir -p ./output/decomposed
mkdir -p ./output/multiscale
mkdir -p ./output/attention
mkdir -p ./output/combined

# 确认预训练模型是否存在
if [ -f "$PRETRAINED" ]; then
    echo "使用预训练模型: $PRETRAINED"
    PRETRAIN_PARAM="--pretrained-path $PRETRAINED"
else
    echo "未找到预训练模型，将从头开始训练"
    PRETRAIN_PARAM=""
fi

# 确认simple_acr_train.py文件存在
if [ ! -f "simple_acr_train.py" ]; then
    echo "错误: simple_acr_train.py 文件不存在"
    exit 1
fi

# 通用参数设置
COMMON_PARAMS="--batch-size 128 --epochs 200 --lr 1e-3 --dataset cifar10 --data-path $DATA_PATH $PRETRAIN_PARAM"

# 训练分解卷积模型
echo "开始训练分解卷积模型..."
python simple_acr_train.py --model acr_repvit_m0_9_decomposed --output-dir ./output/decomposed $COMMON_PARAMS

# 训练多尺度卷积模型
echo "开始训练多尺度卷积模型..."
python simple_acr_train.py --model acr_repvit_m0_9_multiscale --output-dir ./output/multiscale $COMMON_PARAMS

# 训练注意力增强模型
echo "开始训练注意力增强模型..."
python simple_acr_train.py --model acr_repvit_m0_9_attention --output-dir ./output/attention $COMMON_PARAMS

# 训练组合模型
echo "开始训练组合模型..."
python simple_acr_train.py --model acr_repvit_m0_9_combined --output-dir ./output/combined $COMMON_PARAMS

# 测试重参数化效果
echo "测试重参数化效果..."
python simple_acr_train.py --model acr_repvit_m0_9_combined --output-dir ./output/combined --test-only --fuse
