#!/bin/bash

# 设置路径
OUTPUT_DIR="checkpoints/enhanced_repvit"
DATA_PATH="/root/autodl-tmp/data"
RESULTS_FILE="$OUTPUT_DIR/evaluation_results.txt"

# 检查目录
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据路径不存在: $DATA_PATH"
    exit 1
fi

# 创建或清空结果文件
echo "增强型RepViT模型评估结果" > $RESULTS_FILE
echo "============================" >> $RESULTS_FILE
echo "日期: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# 评估基线模型
echo "评估基线模型: repvit_m0_9..."
echo "基线模型: repvit_m0_9" >> $RESULTS_FILE
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model repvit_m0_9 \
    --batch-size 128 \
    --data-path $DATA_PATH \
    --eval \
    --resume pretrained/repvit_m0_9_distill_300e.pth \
    2>&1 | tee temp_eval.log

# 提取基线模型的结果
BASELINE_ACC=$(grep "Accuracy:" temp_eval.log | tail -1 | awk '{print $2}')
echo "准确率: $BASELINE_ACC" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# 模型变体
VARIANTS=(
    "enhanced_repvit_m0_9"
    "enhanced_repvit_m0_9_mixed"
    "enhanced_repvit_m0_9_adaptive" 
    "enhanced_repvit_m0_9_full"
)

# 评估每个变体
for variant in "${VARIANTS[@]}"; do
    MODEL_DIR="$OUTPUT_DIR/$variant"
    
    # 检查模型目录是否存在
    if [ ! -d "$MODEL_DIR" ]; then
        echo "警告: 模型目录不存在: $MODEL_DIR, 跳过"
        echo "$variant: 未训练" >> $RESULTS_FILE
        echo "" >> $RESULTS_FILE
        continue
    fi
    
    # 检查是否有checkpoint_best.pth
    if [ -f "$MODEL_DIR/checkpoint_best.pth" ]; then
        CKPT="$MODEL_DIR/checkpoint_best.pth"
    else
        # 查找最新的检查点
        CKPT=$(ls -t $MODEL_DIR/checkpoint_*.pth 2>/dev/null | head -1)
        if [ -z "$CKPT" ]; then
            echo "警告: 在 $MODEL_DIR 中未找到检查点, 跳过"
            echo "$variant: 未找到检查点" >> $RESULTS_FILE
            echo "" >> $RESULTS_FILE
            continue
        fi
    fi
    
    echo "评估模型: $variant 使用检查点: $(basename $CKPT)..."
    echo "$variant ($(basename $CKPT))" >> $RESULTS_FILE
    
    # 评估模型
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --model $variant \
        --batch-size 128 \
        --data-path $DATA_PATH \
        --eval \
        --resume $CKPT \
        2>&1 | tee temp_eval.log
    
    # 提取结果
    ACC=$(grep "Accuracy:" temp_eval.log | tail -1 | awk '{print $2}')
    echo "准确率: $ACC" >> $RESULTS_FILE
    
    # 计算与基线的差异
    if [ ! -z "$BASELINE_ACC" ] && [ ! -z "$ACC" ]; then
        DIFF=$(echo "$ACC - $BASELINE_ACC" | bc)
        echo "相对基线: $DIFF" >> $RESULTS_FILE
    fi
    
    # 计算和打印参数数量
    echo "计算模型参数数量..."
    python -c "
import torch
import timm
try:
    model = timm.create_model('$variant', pretrained=False)
    params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {params:,}')
    with open('$RESULTS_FILE', 'a') as f:
        f.write(f'参数量: {params:,}\\n')
except Exception as e:
    print(f'计算参数量时出错: {e}')
    with open('$RESULTS_FILE', 'a') as f:
        f.write('参数量: 未知\\n')
"
    
    echo "" >> $RESULTS_FILE
done

# 清理临时文件
rm -f temp_eval.log

echo "评估完成! 结果保存在: $RESULTS_FILE"
cat $RESULTS_FILE 