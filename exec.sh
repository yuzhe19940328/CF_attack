#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc  gs-container_g1.24h
#$ -ac d=nvcr-pytorch-2104

/usr/local/bin/nvidia_entrypoint.sh
. /fefs/opt/dgx/env_set/nvcr-pytorch-2104.sh

export PATH="${HOME}/.raiden/nvcr-pytorch-2104/bin:$PATH"

export LD_LIBRARY_PATH="${HOME}/.raiden//nvcr-pytorch-2104/lib:$LD_LIBRARY_PATH"
export LDFLAGS=-L/usr/local/nvidia/lib64
export PYTHONPATH="${HOME}/.raiden /nvcr-pytorch-2104/lib/python3.8/site-packages"
export PYTHONUSERBASE="${HOME}/.raiden/nvcr-pytorch-2104"
export PREFIX="${HOME}/.raiden/nvcr-pytorch-2104"


export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL
export OMP_NUM_THREADS=8

#python3 finetune.py
#python3 fine_tune_bd.py
#python3 eval_bd.py
#python3 forgetting.py
#python3 evaluate_tv_forgetting.py

python3 test.py