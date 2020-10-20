#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2
export DATASET=/data/dataset/imagenet/ilsvrc2012/torchvision

python nncf_validate.py \
    --model efficientnet_b0 \
    -b 128 \
    -j 6 \
    --num-gpu 1 \
    --img-size 224 \
    --crop-pct 0.875 \
    --interpolation bicubic \
    --nncf_config efficientnet_b0_imagenet_int8.json \
    --log-dir efficientnet_b0_int8 \
    ${DATASET}
