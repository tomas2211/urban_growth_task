#!/bin/bash

NAME='baseline'

python train_net.py \
    --dropout 0.2 \
    --color_augs \
    --data_folder 'data' \
    --out_folder "models/$NAME" \
    --lr 1e-4 \
    --lr_milestones 100000 \
    --lr_gamma 1 \
    --wd 1e-4 \
    --iter_per_epoch 1000 \
    --dat_muliplier 1 \
    --batch_size 10 \
    --crop_size 50 50 \
    --epochs 50 \
    --clip 1 \
    --save_freq 1 \
    --device 'cuda' \
    --class_weight 1 1 1
