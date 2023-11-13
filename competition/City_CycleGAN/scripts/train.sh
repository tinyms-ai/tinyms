#!/bin/bash
filename=$(basename "$0");exp_id="${filename%.*}"
CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_id "$exp_id" \
--mode train \
--dataset_path /path/to/dataset \
--batch_size 1