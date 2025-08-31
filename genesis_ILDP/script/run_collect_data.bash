#!/bin/bash

echo "正在连接服务器并设置端口转发..."
echo "端口转发设置："
echo "  服务器 7100 -> 本地 7100 (环境服务器)"
echo "  服务器 6000 -> 本地 6000 (数据接收)"

ssh -p 9005 -L 7100:localhost:7100 -L 6000:localhost:6000 jiaoda@115.154.239.177