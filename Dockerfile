# Dockerfile for YamagotiProjects (AppleStockChecker)
# 默认使用 CPU 版依赖；GPU 环境可构建时覆盖: --build-arg REQUIREMENTS=requirements.txt
ARG REQUIREMENTS=requirements-cpu.txt
FROM python:3.11-slim
ARG REQUIREMENTS=requirements-cpu.txt

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements-cpu.txt .
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r $REQUIREMENTS

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p /app/staticfiles /app/media

# 默认命令（可被 docker-compose 覆盖）
CMD ["gunicorn", "YamagotiProjects.wsgi:application", "--bind", "0.0.0.0:8000"]
