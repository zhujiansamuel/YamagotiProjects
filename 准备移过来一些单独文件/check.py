# 进入 shell：python manage.py shell
from django.db import connection
from django.apps import apps

# 你的 app_label（从 admin URL 也能看出是 AppleStockChecker）
APP_LABEL = "AppleStockChecker"

# 统计
missing = []
rename_sql = []

with connection.cursor() as c:
    for model in apps.get_app_config(APP_LABEL).get_models():
        expected = model._meta.db_table  # Django 期望的表名（你现在就是 "AppleStockChecker_..." 这种）
        # 1) 检查是否存在“完全同名（大小写敏感）”的表
        c.execute("""
            SELECT to_regclass(%s)
        """, [expected])
        exists_exact = c.fetchone()[0]  # None = 不存在

        if exists_exact:
            continue  # 这张表没问题

        # 2) 猜测“全小写版本”的表名（pg 默认）
        lower_guess = expected.lower()

        # 是否存在全小写
        c.execute("""
            SELECT to_regclass(%s)
        """, [lower_guess])
        exists_lower = c.fetchone()[0]

        if exists_lower:
            # 生成重命名 SQL：把小写表名 → 大小写敏感的 Django 期望名
            rename_sql.append(f'ALTER TABLE {lower_guess} RENAME TO "{expected}";')
        else:
            # 也可能是其它大小写/命名，需要你手动查一下
            missing.append((expected, lower_guess))

print("=== 需要执行的重命名 SQL ===")
print("\n".join(rename_sql) or "-- 无需重命名")

if missing:
    print("\n=== 未找到（既非精确名也非全小写），请手工确认这些表 ===")
    for exp, low in missing:
        print(f"期望: {exp} | 小写猜测: {low}")