#!/bin/bash

# ACR-RepViT: Training Script for All Variants
# This script reproduces all experimental results from the paper

set -e

# Configuration
DATA_PATH=${1:-"/path/to/imagenet"}
OUTPUT_DIR=${2:-"./output"}
BATCH_SIZE=${3:-256}
EPOCHS=${4:-270}

# Create output directories
mkdir -p $OUTPUT_DIR/{baseline,cbam,ca,sa,multiscale}

echo "Starting comprehensive training for ACR-RepViT variants..."
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"

# Function to train a model variant using main.py (for CA/SA)
train_model_main() {
    local model_name=$1
    local output_subdir=$2
    local port=$3
    
    echo "Training $model_name using main.py..."
    python -m torch.distributed.launch --nproc_per_node=1 --master_port $port --use_env main.py \
        --model $model_name \
        --data-path $DATA_PATH \
        --output_dir $OUTPUT_DIR/$output_subdir \
        --batch-size $BATCH_SIZE \
        --lr 0.005 \
        --weight-decay 0.05 \
        --warmup-epochs 5 \
        --warmup-lr 1e-5 \
        --epochs $EPOCHS \
        --sched cosine \
        --dist-eval
    
    echo "$model_name training completed!"
}

# Function to train a model variant using acr_train.py (for CBAM/MultiScale)
train_model_acr() {
    local model_name=$1
    local output_subdir=$2
    
    echo "Training $model_name using acr_train.py..."
    python acr_train.py \
        --model $model_name \
        --batch-size $BATCH_SIZE \
        --data-path $DATA_PATH \
        --output-dir $OUTPUT_DIR/$output_subdir \
        --epochs $EPOCHS \
        --warmup-epochs 20 \
        --lr 0.001 \
        --weight-decay 0.05 \
        --early-stopping \
        --patience 15
    
    echo "$model_name training completed!"
}

# 1. Train RepViT-M0.9 Baseline (using original training script)
echo "=== Training RepViT-M0.9 Baseline ==="
python main.py \
    --model repvit_m0_9 \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR/baseline \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr 0.001 \
    --weight-decay 0.05 \
    --drop-path 0.1 \
    --mixup 0.2 \
    --cutmix 1.0 \
    --smoothing 0.1 \
    --warmup-epochs 5 \
    --dist-eval

# 2. Train ACR-RepViT Variants
echo "=== Training ACR-RepViT Variants ==="

# Channel Attention only (using main.py)
train_model_main "acr_repvit_m0_9_ca" "ca" "12346"

# Spatial Attention only (using main.py)
train_model_main "acr_repvit_m0_9_sa" "sa" "12347"

# CBAM variant (using acr_train.py)
train_model_acr "acr_repvit_m0_9_cbam" "cbam"

# Multi-scale variant (using acr_train.py)
train_model_acr "acr_repvit_m0_9_multiscale" "multiscale"

echo "=== All Training Completed! ==="
echo "Results saved to: $OUTPUT_DIR"
echo "To evaluate all models, run: bash scripts/evaluate_all.sh $OUTPUT_DIR"
