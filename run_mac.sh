#!/usr/bin/env bash

kill -9 $(ps aux | grep 'afy/cam_fomm.py' | awk '{print $2}') 2> /dev/null

source scripts/settings.sh

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

CONFIG=articulated/config/vox256.yaml
CKPT=vox256.pth

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/fomm

python afy/cam_fomm.py --config "$CONFIG" --checkpoint "$CKPT" --no-pad $@
