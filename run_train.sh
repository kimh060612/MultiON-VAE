#!/bin/bash

export EGL_DEVICE_ID=1
# Exploration Training
# CUDA_VISIBLE_DEVICES=0 python3 -m habitat_baselines.run --exp-config habitat_baselines/config/multinav/ppo_exp_multinav.yaml --agent-type exp-multinav --run-type train
# Exploration Attention Training
# CUDA_VISIBLE_DEVICES=0 
CUDA_VISIBLE_DEVICES=0,1 python3 -m habitat_baselines.run --exp-config habitat_baselines/config/multinav/ppo_traj_multinav.yaml --agent-type traj-multinav --run-type train