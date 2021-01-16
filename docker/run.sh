#! /bin/bash

if [ -z $1 ] ; then
    GPU=all
else
    GPU=$1
fi

mkdir -p .fake_home

docker run -it --rm \
    --gpus '"device='$GPU'"' \
    --name 'mufcut' \
    --hostname $(hostname) \
    -u $(id -u):$(id -g) \
    -e HOME \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    -v $(pwd)/.fake_home:$HOME \
    -v $(pwd):/mufcut \
    -w /mufcut \
    mufcut:1.0.0