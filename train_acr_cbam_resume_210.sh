#!/bin/bash

# 从第210轮checkpoint继续训练ACR-RepViT-M0.9 CBAM模型
python main.py \
  --model acr_repvit_m0_9_cbam \
  --data-path /root/autodl-tmp/data \
  --data-set IMNET \
  --batch-size 128 \
  --epochs 250 \
  --lr 0.0005 \
  --resume ./output/imagenet_acr_cbam_m0_9/acr_repvit_m0_9_cbam/2025_04_18_22_38_52/checkpoint_210.pth \
  --output_dir ./output/imagenet_acr_cbam_m0_9 \
  --amp \
  --warmup-epochs 0 \
  --mixup 0.5 \
  --weight-decay 0.05 \
  --pretrained_path /root/RepViT/pretrained/repvit_m0_9_distill_300e.pth 