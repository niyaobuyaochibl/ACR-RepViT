#!/bin/bash

# ACR-RepViT单模型训练脚本
# 用于快速验证一个特定重参数化策略的效果

# 设置路径变量
OUTPUT_DIR="checkpoints/acr_repvit"
DATA_PATH="/root/autodl-tmp/data"  # 数据集路径

# 接收命令行参数指定模型
MODEL=${1:-"acr_repvit_m0_9_decomposed"}  # 默认使用分解卷积模型

# 创建输出目录
mkdir -p $OUTPUT_DIR
OUTPUT_SUBDIR="$OUTPUT_DIR/${MODEL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_SUBDIR

echo "=========================================="
echo "  ACR-RepViT单模型训练 - $MODEL"
echo "  输出目录: $OUTPUT_SUBDIR"
echo "  数据路径: $DATA_PATH"
echo "=========================================="

# 安装兼容的库版本
pip install -q timm==0.5.4

# 执行5个epoch的快速训练测试
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model $MODEL \
    --batch-size 64 \
    --data-path $DATA_PATH \
    --lr 0.001 \
    --min-lr 1e-6 \
    --weight-decay 0.02 \
    --epochs 5 \
    --warmup-epochs 1 \
    --output_dir $OUTPUT_SUBDIR \
    --sched cosine \
    --amp \
    --clip-grad 1.0 \
    --model-ema \
    --distillation-type none \
    | tee $OUTPUT_SUBDIR/train.log

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "✅ 训练成功: $MODEL"
    echo "结果保存在: $OUTPUT_SUBDIR"
    echo ""
    echo "要进行完整训练，请运行:"
    echo "bash train_acr.sh"
else
    echo "❌ 训练失败: $MODEL，请检查日志"
fi 