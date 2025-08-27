#!/bin/bash

# 使用官方训练命令，只是改为使用ACR模型变体
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12347 --use_env main.py \
  --model acr_repvit_m0_9_sa \
  --data-path /root/autodl-tmp/data \
  --output_dir /root/autodl-tmp/RepViT_output/imagenet_acr_sa_m0_9 \
  --batch-size 256 \
  --lr 0.005 \
  --weight-decay 0.05 \
  --warmup-epochs 5 \
  --warmup-lr 1e-5 \
  --epochs 270 \
  --sched cosine \
  --dist-eval 