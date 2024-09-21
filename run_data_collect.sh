#!/bin/bash

# export EGL_DEVICE_ID=0
# export BUILD_GUI_VIEWERS=0
python3 habitat_baselines/run.py --exp-config habitat_baselines/config/multinav/ppo_multinav.yaml --agent-type obj-recog --run-type eval