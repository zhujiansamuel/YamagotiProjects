#!/bin/bash
# 1. 启动手动管理的容器 (之前检查显示为 'no' 重启策略的)
echo "Starting standalone containers..."
docker start ypa_redis ypa_clickhouse local-pg 2>/dev/null

# 2. 启动项目主 Compose 服务
echo "Starting project compose services..."
cd /home/samuelzhu/YamagotiProjects
docker compose up -d

# 3. 确保系统级服务已启动
echo "Ensuring systemd services are running..."
sudo systemctl start caddy ollama

# 4. 打印状态汇总
echo "--------------------------------------"
echo "Recovery Complete. Current Status:"
docker ps --format "table {{.Names}}	{{.Status}}"
echo "--------------------------------------"
