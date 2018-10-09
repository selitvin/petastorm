#!/usr/bin/env bash

cd $HOME
git clone https://github.com/uber/petastorm
python3.6 -m venv .env
source .env/bin/activate
cd petastorm
pip install -e .[tf,docs,test,opencv,torch]
