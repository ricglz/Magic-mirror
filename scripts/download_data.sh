#!/usr/bin/env bash

filename=vox-adv-cpk.pth.tar

curl -O https://openavatarify.s3.amazonaws.com/weights/$filename

echo "Expected checksum: 8a45a24037871c045fbb8a6a8aa95ebc"

if [ "$(uname)" == "Darwin" ]; then
  sum=`md5 ${filename}`
else
  sum=`md5sum ${filename}`
fi
echo "Found checksum:    $sum"

filename=vox256.pth

curl -L -O https://github.com/snap-research/articulated-animation/raw/main/checkpoints/$filename

if [ "$(uname)" == "Darwin" ]; then
  sum=`md5 ${filename}`
else
  sum=`md5sum ${filename}`
fi
echo "Found checksum:    $sum"
