#!/bin/bash

# ACR-RepViT Channel Attention Training Script
# Based on the actual training configuration used in the paper

# Configuration
DATA_PATH=${1:-"/path/to/imagenet"}
OUTPUT_DIR=${2:-"./output/imagenet_acr_ca_m0_9"}
BATCH_SIZE=${3:-256}
EPOCHS=${4:-270}

echo "Training ACR-RepViT-M0.9-CA (Channel Attention only)..."
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Train using main.py with distributed launch (as per original script)
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 --use_env main.py \
  --model acr_repvit_m0_9_ca \
  --data-path $DATA_PATH \
  --output_dir $OUTPUT_DIR \
  --batch-size $BATCH_SIZE \
  --lr 0.005 \
  --weight-decay 0.05 \
  --warmup-epochs 5 \
  --warmup-lr 1e-5 \
  --epochs $EPOCHS \
  --sched cosine \
  --dist-eval

echo "ACR-RepViT-M0.9-CA training completed!"
echo "Results saved to: $OUTPUT_DIR"

