# ELK Container Mapping 定时更新

## 概述

ELK 日志系统通过 `container-mapping.yml` 将 Docker 容器短 ID 映射为可读的容器名称。
当容器重建后 ID 会变化，如果映射文件未更新，Logstash 的 `translate` filter 会将容器名标记为 `"unknown"`。

为此配置了 crontab 定时任务，每小时自动更新映射文件。

## Crontab 配置

```
0 * * * * /home/samuelzhu/ELK/logstash/update-container-mapping.sh >> /home/samuelzhu/ELK/logstash/mapping-update.log 2>&1
```

- **执行频率**: 每小时整点
- **脚本路径**: `/home/samuelzhu/ELK/logstash/update-container-mapping.sh`
- **执行日志**: `/home/samuelzhu/ELK/logstash/mapping-update.log`
- **所属用户**: `samuelzhu`

## 工作原理

1. 脚本通过 `docker ps` 获取所有运行中容器的短 ID 和名称
2. 生成 YAML 格式的映射文件写入 `/home/samuelzhu/ELK/logstash/data/container-mapping.yml`
3. Logstash 的 `translate` filter 配置了 `refresh_interval => 60`，会在 60 秒内自动加载新映射

## 管理命令

```bash
# 查看当前 crontab
crontab -l

# 手动执行更新
bash /home/samuelzhu/ELK/logstash/update-container-mapping.sh

# 查看执行日志
tail -f /home/samuelzhu/ELK/logstash/mapping-update.log

# 查看当前映射内容
cat /home/samuelzhu/ELK/logstash/data/container-mapping.yml
```

## 注意事项

- Logstash 的 `refresh_interval` 为 60 秒，映射更新后最多 1 分钟生效，无需重启 Logstash
- 如需立即生效可手动重启：`docker restart logstash`
- 脚本仅映射当前运行中的容器，已停止的容器不会出现在映射中
