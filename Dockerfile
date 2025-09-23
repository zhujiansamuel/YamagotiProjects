FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# 复制源码并安装依赖
COPY src/ /app/
RUN pip install --upgrade pip && pip install -r requirements/base.txt

# 收集静态
RUN python manage.py collectstatic --noinput
