#!/bin/bash
# -*- coding: utf-8 -*-
# 快速数据库权限诊断脚本（无需交互）

set -e

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_DIR"

echo "======================================"
echo "数据库权限快速诊断"
echo "======================================"
echo ""
echo "项目目录: $PROJECT_DIR"
echo ""

# 检测 Python 命令
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ 错误: 未找到 Python 命令"
    echo "请安装 Python 3"
    exit 1
fi

echo "使用 Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""
echo "======================================"
echo "开始诊断..."
echo "======================================"
echo ""

# 运行诊断脚本
$PYTHON_CMD scripts/diagnose_db_permissions.py

echo ""
echo "======================================"
echo "诊断完成"
echo "======================================"
echo ""
echo "如果发现权限问题，可以运行以下命令修复："
echo "  ./scripts/fix_permissions.sh"
echo ""
echo "或者手动修复（需要 postgres 超级用户）："
echo "  psql -U postgres -d applestockchecker_dev -f scripts/fix_db_permissions.sql"
