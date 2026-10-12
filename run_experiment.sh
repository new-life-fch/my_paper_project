#!/bin/bash
# Run initial validation experiment
# ⚠️ 代理凭据绝不写进仓库（本文件曾硬编码 user:pass 并已随公开仓库泄露，
#    2026-10-12 移除）。请在调用前从环境注入：
#      export https_proxy="http://<user>:<pass>@<proxy-host>:<port>"
if [ -z "${https_proxy:-}" ]; then
  echo "!! 未设置 https_proxy —— 需要联网下载时会失败。" >&2
  echo "   请先: export https_proxy=\"http://<user>:<pass>@<proxy-host>:<port>\"" >&2
  exit 1
fi
export http_proxy="$https_proxy"
export no_proxy="${no_proxy:-127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn}"

# 禁用 xet 协议下载（代理不兼容），强制传统 HTTP 下载
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1

# 在本仓库根目录运行（原先硬编码 /root/shared-nvme/my_paper_project，
# 那是拆 worktree 之前的路径，换机器必挂）
cd "$(dirname "$0")"
python -u initial_validation.py --n-queries 50 --scheme both
