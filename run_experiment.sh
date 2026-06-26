#!/bin/bash
# Run initial validation experiment
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"

# 禁用 xet 协议下载（代理不兼容），强制传统 HTTP 下载
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1

cd /root/shared-nvme/my_paper_project
python -u initial_validation.py --n-queries 50 --scheme both
