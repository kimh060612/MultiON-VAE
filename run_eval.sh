#!/bin/bash

export EGL_DEVICE_ID=1
python3 habitat_baselines/run.py --exp-config habitat_baselines/config/multinav/ppo_multinav.yaml --agent-type obj-recog --run-type eval