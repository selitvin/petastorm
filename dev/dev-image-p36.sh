#!/bin/bash -eu

docker build -t petastorm-dev-image-p36 .
docker run -it petastorm-dev-image-p36