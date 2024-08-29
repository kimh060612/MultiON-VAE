#!/bin/bash

export MAGNUM_LOG=verbose
export MAGNUM_GPU_VALIDATION=ON
export EGL_DEVICE_ID=1
# Exploration Training
# CUDA_VISIBLE_DEVICES=0 python3 -m habitat_baselines.run --exp-config habitat_baselines/config/multinav/ppo_exp_multinav.yaml --agent-type exp-multinav --run-type train
# Exploration Attention Training
CUDA_VISIBLE_DEVICES=0 python3 -m habitat_baselines.run --exp-config habitat_baselines/config/multinav/ppo_ea_multinav.yaml --agent-type ea-multinav --run-type train