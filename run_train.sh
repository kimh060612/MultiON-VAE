#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 -m habitat_baselines.run --exp-config habitat_baselines/config/multinav/ppo_exp_multinav.yaml --agent-type exp-multinav --run-type train