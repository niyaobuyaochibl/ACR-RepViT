#!/bin/bash

# ACR-RepViT 评估脚本 - 使用官方评估方法
# 用于评估不同重参数化策略的性能

# 基础设置
DATA_PATH="/root/autodl-tmp/data"  # ImageNet数据集路径
MODEL_PATH="checkpoints/acr_repvit"  # 模型权重路径
PRETRAINED_PATH="pretrained/repvit_m0_9_distill_300e.pth"  # 官方预训练权重路径

# 确保输出目录存在
mkdir -p "${MODEL_PATH}/evaluation_results"

# 评估函数
evaluate_model() {
    local model_name=$1
    local checkpoint_path=$2
    
    echo "========== 评估模型: $model_name =========="
    
    # 使用官方的评估命令，添加--fuse参数以使用重参数化
    python main.py \
        --eval \
        --model $model_name \
        --resume $checkpoint_path \
        --data-path $DATA_PATH \
        --num_workers 8 \
        --fuse \
        | tee "${MODEL_PATH}/evaluation_results/${model_name}_eval.log"
        
    # 提取并保存结果
    grep -A 1 "* Acc@1" "${MODEL_PATH}/evaluation_results/${model_name}_eval.log" > "${MODEL_PATH}/evaluation_results/${model_name}_summary.txt"
}

# 1. 评估官方基线模型 RepViT-M0.9
evaluate_model "repvit_m0_9" "${PRETRAINED_PATH}"

# 2. 评估分解卷积重参数化变体
if [ -f "${MODEL_PATH}/decomposed/checkpoint-best.pth" ]; then
    evaluate_model "acr_repvit_m0_9_decomposed" "${MODEL_PATH}/decomposed/checkpoint-best.pth"
else
    echo "找不到分解卷积重参数化模型权重，跳过评估"
fi

# 3. 评估多尺度重参数化变体
if [ -f "${MODEL_PATH}/multiscale/checkpoint-best.pth" ]; then
    evaluate_model "acr_repvit_m0_9_multiscale" "${MODEL_PATH}/multiscale/checkpoint-best.pth"
else
    echo "找不到多尺度重参数化模型权重，跳过评估"
fi

# 4. 评估注意力增强重参数化变体
if [ -f "${MODEL_PATH}/attention/checkpoint-best.pth" ]; then
    evaluate_model "acr_repvit_m0_9_attention" "${MODEL_PATH}/attention/checkpoint-best.pth"
else
    echo "找不到注意力增强重参数化模型权重，跳过评估"
fi

# 5. 评估渐进式结构搜索变体
if [ -f "${MODEL_PATH}/progressive/checkpoint-best.pth" ]; then
    evaluate_model "acr_repvit_m0_9" "${MODEL_PATH}/progressive/checkpoint-best.pth"
else
    echo "找不到渐进式结构搜索模型权重，跳过评估"
fi

# 生成汇总报告
echo "========== ACR-RepViT评估结果汇总 ==========" > "${MODEL_PATH}/evaluation_results/summary.txt"
echo "模型                             Top-1准确率     Top-5准确率" >> "${MODEL_PATH}/evaluation_results/summary.txt"
echo "---------------------------------------------------------------" >> "${MODEL_PATH}/evaluation_results/summary.txt"

# 从各个日志中提取结果
for model in "repvit_m0_9" "acr_repvit_m0_9_decomposed" "acr_repvit_m0_9_multiscale" "acr_repvit_m0_9_attention" "acr_repvit_m0_9"; do
    if [ -f "${MODEL_PATH}/evaluation_results/${model}_summary.txt" ]; then
        acc=$(grep "Acc@1" "${MODEL_PATH}/evaluation_results/${model}_summary.txt" | awk '{print $3}')
        acc5=$(grep "Acc@5" "${MODEL_PATH}/evaluation_results/${model}_summary.txt" | awk '{print $3}')
        printf "%-35s %-15s %-15s\n" "$model" "$acc" "$acc5" >> "${MODEL_PATH}/evaluation_results/summary.txt"
    fi
done

echo "评估完成! 结果保存在: ${MODEL_PATH}/evaluation_results/summary.txt"
cat "${MODEL_PATH}/evaluation_results/summary.txt" 