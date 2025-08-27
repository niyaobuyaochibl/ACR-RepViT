#!/bin/bash

# 使用官方训练命令，只是改为使用ACR模型变体
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12349 --use_env main.py \
  --model acr_repvit_m0_9_ms \
  --data-path /root/autodl-tmp/data \
  --output_dir /root/autodl-tmp/RepViT_output/imagenet_acr_ms_m0_9 \
  --batch-size 256 \
  --lr 0.001 \
  --weight-decay 0.05 \
  --warmup-epochs 20 \
  --epochs 270 \
  --dist-eval 