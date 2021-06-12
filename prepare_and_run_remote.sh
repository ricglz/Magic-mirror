#!/usr/bin/env bash

bash scripts/install.sh --no-vcam
if [! -d 'vox256.pth']; then
  bash scripts/download_data.sh
fi
bash run.sh --is-worker

echo 'run_mac.sh --is-client --in-addr tcp://server_address:5557 --out-addr tcp://server_address:5558'
