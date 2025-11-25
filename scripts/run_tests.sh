#!/bin/bash
# 运行测试

set -e

echo "🧪 运行测试..."

# 运行 pytest
pytest -v --tb=short

echo "✅ 测试完成！"
