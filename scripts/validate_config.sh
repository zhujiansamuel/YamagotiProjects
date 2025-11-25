#!/bin/bash
# 验证配置文件

set -e

echo "🔍 验证配置文件..."
echo ""

# 检查必要文件存在
echo "📂 检查必要文件..."
files=(
    "docker-compose.yml"
    "Dockerfile"
    ".env.example"
    "YamagotiProjects/settings.py"
    "YamagotiProjects/celery.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (不存在)"
    fi
done

echo ""

# 检查 Python 语法
echo "🐍 检查 Python 语法..."
python_files=(
    "YamagotiProjects/settings.py"
    "YamagotiProjects/celery.py"
)

for file in "${python_files[@]}"; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (语法错误)"
    fi
done

echo ""

# 检查 YAML 语法
echo "📋 检查 YAML 语法..."
if python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))" 2>/dev/null; then
    echo "  ✅ docker-compose.yml"
else
    echo "  ❌ docker-compose.yml (语法错误)"
fi

echo ""

# 检查脚本文件权限
echo "🔧 检查脚本文件权限..."
for script in scripts/*.sh; do
    if [ -x "$script" ]; then
        echo "  ✅ $script (可执行)"
    else
        echo "  ⚠️  $script (不可执行，运行 chmod +x $script)"
    fi
done

echo ""

# 检查环境变量
echo "⚙️  检查环境变量..."
if [ -f ".env" ]; then
    echo "  ✅ .env 文件存在"

    # 检查关键环境变量
    if grep -q "USE_PGBOUNCER" .env 2>/dev/null; then
        echo "  ✅ USE_PGBOUNCER 已配置"
    else
        echo "  ⚠️  USE_PGBOUNCER 未配置（使用默认值）"
    fi
else
    echo "  ⚠️  .env 文件不存在（建议从 .env.example 复制）"
    echo "     运行: cp .env.example .env"
fi

echo ""

# 检查文档
echo "📚 检查文档..."
docs=(
    "docs/PGBOUNCER_SETUP.md"
    "docs/CELERY_QUEUES.md"
    "docs/README_CHANGES.md"
)

for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo "  ✅ $doc"
    else
        echo "  ❌ $doc (不存在)"
    fi
done

echo ""
echo "✅ 配置验证完成！"
echo ""
echo "💡 下一步："
echo "  1. 复制环境变量模板: cp .env.example .env"
echo "  2. 编辑 .env 文件，配置密码等敏感信息"
echo "  3. 启动开发环境: ./scripts/dev_up.sh"
echo "  4. 或启动生产环境: ./scripts/prod_up.sh"
echo ""
echo "📖 详细文档："
echo "  - PgBouncer 配置: docs/PGBOUNCER_SETUP.md"
echo "  - Celery 队列配置: docs/CELERY_QUEUES.md"
echo "  - 变更说明: docs/README_CHANGES.md"
