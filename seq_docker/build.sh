#!/usr/bin/env bash

# Check args
if [ "$#" -ne 1 ]; then
  echo "usage: ./build.sh IMAGE_NAME"
  return 1
fi

# Get this script's path
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

# Need to download the .deb to install the Nvidia GPG Repository Key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb

# Build the docker image
docker build\
  --build-arg user=$USER\
  --build-arg uid=$UID\
  --build-arg home=$HOME\
  --build-arg workspace="/workspace/" \
  --build-arg shell=$SHELL\
  -t $1 \
  -f Dockerfile .