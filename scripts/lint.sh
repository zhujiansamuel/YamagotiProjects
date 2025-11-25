#!/bin/bash
# 代码质量检查

set -e

echo "🔍 代码质量检查..."

# 如果安装了 ruff
if command -v ruff &> /dev/null; then
    echo "📝 Running ruff..."
    ruff check .
fi

# 如果安装了 black
if command -v black &> /dev/null; then
    echo "📝 Running black..."
    black --check .
fi

# 如果安装了 mypy
if command -v mypy &> /dev/null; then
    echo "📝 Running mypy..."
    mypy . --ignore-missing-imports
fi

echo "✅ 代码质量检查完成！"
