#!/bin/bash

# ACR-RepViT MultiScale Training Script
# Based on the actual training configuration used in the paper

# Configuration
DATA_PATH=${1:-"/path/to/imagenet"}
OUTPUT_DIR=${2:-"./output/imagenet_acr_m0_9_multiscale"}
BATCH_SIZE=${3:-256}
EPOCHS=${4:-270}

echo "Training ACR-RepViT-M0.9-MultiScale (Multi-scale convolution)..."
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Train using acr_train.py (as per original script)
python acr_train.py \
  --model acr_repvit_m0_9_multiscale \
  --batch-size $BATCH_SIZE \
  --data-path $DATA_PATH \
  --output-dir $OUTPUT_DIR \
  --epochs $EPOCHS \
  --warmup-epochs 20 \
  --lr 0.001 \
  --weight-decay 0.05 \
  --early-stopping \
  --patience 15

echo "ACR-RepViT-M0.9-MultiScale training completed!"
echo "Results saved to: $OUTPUT_DIR"

