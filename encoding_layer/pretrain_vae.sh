#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset_root="/code/map_data" --model_name="MAP_VAE" --save_dir="/code/encoder_checkpoint" -dm