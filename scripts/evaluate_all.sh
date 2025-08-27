#!/bin/bash

# ACR-RepViT: Comprehensive Evaluation Script
# Evaluates all model variants and generates comparison results

set -e

# Configuration
DATA_PATH=${1:-"/path/to/imagenet"}
MODEL_DIR=${2:-"./output"}
RESULTS_DIR=${3:-"./results"}

# Create results directory
mkdir -p $RESULTS_DIR

echo "Starting comprehensive evaluation of ACR-RepViT variants..."
echo "Data path: $DATA_PATH"
echo "Model directory: $MODEL_DIR"
echo "Results directory: $RESULTS_DIR"

# Function to evaluate a model
evaluate_model() {
    local model_name=$1
    local model_path=$2
    local script_name=${3:-"main.py"}
    local output_file=$4
    
    echo "Evaluating $model_name..."
    
    python $script_name \
        --eval \
        --model $model_name \
        --resume $model_path \
        --data-path $DATA_PATH \
        --batch-size 256 \
        --dist-eval \
        > $output_file 2>&1
    
    # Extract accuracy from log
    local top1_acc=$(grep "Acc@1" $output_file | tail -1 | awk '{print $3}' | sed 's/,//')
    local top5_acc=$(grep "Acc@5" $output_file | tail -1 | awk '{print $5}')
    
    echo "$model_name: Top-1: $top1_acc%, Top-5: $top5_acc%"
    echo "$model_name,$top1_acc,$top5_acc" >> $RESULTS_DIR/accuracy_summary.csv
}

# Initialize results file
echo "Model,Top1_Accuracy,Top5_Accuracy" > $RESULTS_DIR/accuracy_summary.csv

# Evaluate baseline RepViT-M0.9
echo "=== Evaluating RepViT-M0.9 Baseline ==="
evaluate_model "repvit_m0_9" \
    "$MODEL_DIR/baseline/checkpoint.pth" \
    "main.py" \
    "$RESULTS_DIR/repvit_m0_9_eval.log"

# Evaluate ACR-RepViT variants
echo "=== Evaluating ACR-RepViT Variants ==="

# CBAM variant
evaluate_model "acr_repvit_m0_9_cbam" \
    "$MODEL_DIR/cbam/checkpoint.pth" \
    "acr_train.py" \
    "$RESULTS_DIR/acr_repvit_cbam_eval.log"

# Channel Attention
evaluate_model "acr_repvit_m0_9_ca" \
    "$MODEL_DIR/ca/checkpoint.pth" \
    "acr_train.py" \
    "$RESULTS_DIR/acr_repvit_ca_eval.log"

# Spatial Attention
evaluate_model "acr_repvit_m0_9_sa" \
    "$MODEL_DIR/sa/checkpoint.pth" \
    "acr_train.py" \
    "$RESULTS_DIR/acr_repvit_sa_eval.log"

# Multi-scale
evaluate_model "acr_repvit_m0_9_multiscale" \
    "$MODEL_DIR/multiscale/checkpoint.pth" \
    "acr_train.py" \
    "$RESULTS_DIR/acr_repvit_multiscale_eval.log"

echo "=== Evaluation Completed! ==="
echo "Results summary:"
cat $RESULTS_DIR/accuracy_summary.csv

# Generate detailed comparison table
echo "=== Generating Detailed Analysis ==="
python analyze_results.py \
    --results-dir $RESULTS_DIR \
    --model-dir $MODEL_DIR \
    --output $RESULTS_DIR/detailed_comparison.csv

echo "Detailed results saved to: $RESULTS_DIR/detailed_comparison.csv"
echo "To generate visualizations, run: bash scripts/generate_visualizations.sh $RESULTS_DIR"

