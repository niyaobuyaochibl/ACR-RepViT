#!/bin/bash

# 训练ACR-RepViT-M0.9 Combined模型（多尺度+CBAM）
python main.py \
  --model acr_repvit_m0_9_combined \
  --data-path /root/autodl-tmp/data \
  --data-set IMNET \
  --batch-size 128 \
  --epochs 200 \
  --lr 0.002 \
  --pretrained \
  --pretrained_path /root/RepViT/pretrained/repvit_m0_9_distill_300e.pth \
  --output_dir ./output/imagenet_acr_combined_m0_9 \
  --amp \
  --warmup-epochs 10 \
  --mixup 0.5 \
  --weight-decay 0.05 