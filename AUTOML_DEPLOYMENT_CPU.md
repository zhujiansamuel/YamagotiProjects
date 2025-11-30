# AutoML Pipeline 部署指南 - CPU 模式

本文档说明如何在**没有 GPU** 的环境下部署和运行 AutoML 三阶段因果分析 pipeline。

---

## 📋 快速部署步骤

### 1. 检查文件

确保以下文件存在：
```bash
ls -la docker-compose-automl-cpu.yml
ls -la Dockerfile.cpu
ls -la requirements-cpu.txt
```

### 2. 构建镜像

```bash
# 使用 CPU 专用配置构建镜像
docker-compose -f docker-compose-automl-cpu.yml build

# 查看构建的镜像
docker images | grep apple-web-cpu
```

**预期输出**:
```
apple-web-cpu       latest    abc123def456   2 minutes ago   1.2GB
```

### 3. 启动服务

```bash
# 启动所有服务
docker-compose -f docker-compose-automl-cpu.yml up -d

# 查看运行中的容器
docker-compose -f docker-compose-automl-cpu.yml ps
```

### 4. 验证 AutoML Workers

检查三个 AutoML worker 是否正常运行：

```bash
# 查看所有 AutoML worker 状态
docker ps | grep automl

# 查看 worker 日志
docker-compose -f docker-compose-automl-cpu.yml logs -f worker_automl_preprocessing
docker-compose -f docker-compose-automl-cpu.yml logs -f worker_automl_cause_effect
docker-compose -f docker-compose-automl-cpu.yml logs -f worker_automl_impact
```

**期望看到的日志**:
```
✗ GPU not available, using CPU for AutoML pipeline
[INFO/MainProcess] Connected to redis://redis:6379/0
[INFO/MainProcess] celery@... ready.
```

### 5. 运行数据库迁移

```bash
# 进入 web 容器
docker-compose -f docker-compose-automl-cpu.yml exec web bash

# 运行迁移
python manage.py makemigrations AppleStockChecker
python manage.py migrate

# 退出容器
exit
```

---

## 🧪 测试 AutoML Pipeline

### 方式 1: 通过 Django Shell 创建任务

```bash
# 进入 web 容器
docker-compose -f docker-compose-automl-cpu.yml exec web python manage.py shell

# 在 shell 中运行
from AppleStockChecker.tasks.automl_tasks import schedule_automl_jobs
result = schedule_automl_jobs.delay()
print(f"Task ID: {result.id}")
```

### 方式 2: 通过 API 创建任务

```bash
# 创建一个 AutoML Job
curl -X POST http://localhost/automl/jobs/create/ \
  -H "Content-Type: application/json" \
  -d '{"iphone_id": 1, "days": 7}'

# 查看 Job 状态
curl http://localhost/automl/jobs/status/
```

### 方式 3: 访问 AutoML 页面

打开浏览器访问：
```
http://localhost/automl/
```

---

## 📊 监控任务执行

### 查看 Flower (Celery 监控面板)

```
http://localhost:5555/flower
```

在 Flower 中可以看到：
- 三个 AutoML 队列：`automl_preprocessing`, `automl_cause_effect`, `automl_impact`
- 每个队列的任务执行情况
- Worker 状态和资源使用

### 查看实时日志

```bash
# Stage 1: Preprocessing
docker logs -f apple-worker-automl-preprocessing

# Stage 2: VAR Modeling
docker logs -f apple-worker-automl-cause-effect

# Stage 3: Impact Quantification
docker logs -f apple-worker-automl-impact
```

**成功执行的日志示例**:
```
[Job 1] Starting preprocessing...
✗ GPU not available, using CPU for AutoML pipeline
[Job 1] Found 1500 PSTA records
[Job 1] Preprocessing complete, created 450 series
[Job 1] Triggering VAR stage...

[Job 1] Starting VAR modeling...
[Job 1] Panel shape: (120, 5) (T=120, S=5)
[Job 1] VAR fitted: lag_order=2, AIC=450.23
[Job 1] Triggering Impact stage...

[Job 1] Starting Impact quantification (Granger)...
[Job 1] Running Granger tests for 5 shops (maxlag=2)
[Job 1] Impact complete: 20 tests, 8 significant edges
```

---

## 🔍 故障排查

### 问题 1: cupy 安装失败

如果看到 `cupy` 相关错误，检查是否使用了正确的配置文件：

```bash
# ✅ 正确 - 使用 CPU 配置
docker-compose -f docker-compose-automl-cpu.yml up -d

# ❌ 错误 - 使用了 GPU 配置
docker-compose up -d  # 这会使用默认的 docker-compose.yml
```

### 问题 2: Worker 启动失败

```bash
# 查看 worker 日志
docker-compose -f docker-compose-automl-cpu.yml logs worker_automl_preprocessing

# 重启 worker
docker-compose -f docker-compose-automl-cpu.yml restart worker_automl_preprocessing
```

### 问题 3: 数据库连接错误

```bash
# 检查 PostgreSQL 和 PgBouncer 状态
docker-compose -f docker-compose-automl-cpu.yml ps db pgbouncer

# 重启数据库服务
docker-compose -f docker-compose-automl-cpu.yml restart db pgbouncer
```

### 问题 4: Redis 连接错误

```bash
# 检查 Redis 状态
docker-compose -f docker-compose-automl-cpu.yml exec redis redis-cli ping

# 应该返回: PONG
```

---

## 📈 性能优化（CPU 模式）

由于使用 CPU 运算，可以调整以下参数提高性能：

### 增加 Worker 并发数

编辑 `docker-compose-automl-cpu.yml`:

```yaml
# 默认每个 worker 2 个并发
command: ["celery", "-A", "YamagotiProjects", "worker", "-Q", "automl_preprocessing", "-l", "info", "-c", "2"]

# 增加到 4 个并发（如果 CPU 核心足够）
command: ["celery", "-A", "YamagotiProjects", "worker", "-Q", "automl_preprocessing", "-l", "info", "-c", "4"]
```

### 减少任务复杂度

在测试阶段，可以减少数据量：

```python
# 在 API 调用时指定更短的时间窗口
curl -X POST http://localhost/automl/jobs/create/ \
  -H "Content-Type: application/json" \
  -d '{"iphone_id": 1, "days": 3}'  # 从 7 天减少到 3 天
```

---

## 🚀 后续升级到 GPU

当有 GPU 资源时，可以切换到 GPU 版本：

```bash
# 1. 停止 CPU 版本
docker-compose -f docker-compose-automl-cpu.yml down

# 2. 安装 NVIDIA Container Toolkit
# (参考官方文档)

# 3. 使用 GPU 版本
docker-compose -f docker-compose-with-automl-workers.yml up -d --build
```

代码会**自动检测** GPU 并启用加速，无需修改代码！

---

## 📝 重要提示

1. **CPU 模式性能**: CPU 模式下，大规模数据处理会比较慢，建议：
   - 从小数据集开始测试
   - 逐步增加数据量
   - 监控服务器资源使用

2. **自动降级**: 即使使用 GPU 配置，如果 GPU 不可用，代码也会自动降级到 CPU，不会崩溃

3. **日志监控**: 始终查看 worker 日志，确认是否正确使用 CPU 模式：
   ```
   ✗ GPU not available, using CPU for AutoML pipeline
   ```

4. **数据库备份**: 在运行 AutoML 任务前，建议备份数据库

---

## 📞 获取帮助

- 查看日志: `docker-compose -f docker-compose-automl-cpu.yml logs`
- 查看特定服务: `docker-compose -f docker-compose-automl-cpu.yml logs worker_automl_preprocessing`
- 进入容器调试: `docker-compose -f docker-compose-automl-cpu.yml exec web bash`

---

**祝部署顺利！** 🎉
