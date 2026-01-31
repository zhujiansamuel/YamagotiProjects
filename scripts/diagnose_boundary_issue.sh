#!/bin/bash
# 边界判断问题诊断脚本的 Shell 包装器

set -e

if [ -f "/.dockerenv" ]; then
    python scripts/diagnose_boundary_issue.py "$@"
elif docker compose ps web &>/dev/null 2>&1; then
    docker compose exec web python scripts/diagnose_boundary_issue.py "$@"
else
    python scripts/diagnose_boundary_issue.py "$@"
fi
