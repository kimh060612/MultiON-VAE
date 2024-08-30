#!/bin/bash

pip install -r requirements.txt && pip install --no-index --upgrade --no-deps --force-reinstall torch_scatter -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install pandas
pip install multiprocess
# ln -s /dev/nvidia3 /dev/nvidia0
# ln -s /dev/nvidia5 /dev/nvidia1