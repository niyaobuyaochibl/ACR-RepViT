#!/bin/bash

python acr_train.py \
  --model acr_repvit_m0_9_multiscale \
  --batch-size 256 \
  --data-path /root/autodl-tmp/data \
  --output-dir ./output/imagenet_acr_m0_9_multiscale \
  --epochs 270 \
  --warmup-epochs 20 \
  --lr 0.001 \
  --weight-decay 0.05 \
  --early-stopping \
  --patience 15
