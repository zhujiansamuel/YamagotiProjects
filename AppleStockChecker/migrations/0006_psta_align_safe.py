from django.db import migrations

PSTA = '"public"."AppleStockChecker_purchasingshoptimeanalysis"'

SQL = f"""
-- 供实时流水线使用的辅助列（占位/样本数/新鲜度/最终化）
ALTER TABLE {PSTA} ADD COLUMN IF NOT EXISTS is_placeholder BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE {PSTA} ADD COLUMN IF NOT EXISTS sample_cnt   INT     NOT NULL DEFAULT 0;
ALTER TABLE {PSTA} ADD COLUMN IF NOT EXISTS src_max_ts   TIMESTAMPTZ NULL;
ALTER TABLE {PSTA} ADD COLUMN IF NOT EXISTS is_final     BOOLEAN NOT NULL DEFAULT FALSE;

-- 常用索引（幂等）
CREATE INDEX IF NOT EXISTS apple_psta_idx_bucket      ON {PSTA} ("Timestamp_Time");
CREATE INDEX IF NOT EXISTS apple_psta_idx_shop_bucket ON {PSTA} ("shop_id", "Timestamp_Time");

-- 如果你希望把“唯一约束”扩展为 (shop_id, iphone_id, "Timestamp_Time")，你已经有：
-- CONSTRAINT uniq_shop_iphone_Timestamp_Time UNIQUE ("shop_id","iphone_id","Timestamp_Time")
-- 若名称不同，可按实际名称做 DROP 再 ADD，示例（请按实际旧名替换 <old_name>，否则保持注释）：
-- DO $$
-- BEGIN
--   IF NOT EXISTS (
--     SELECT 1 FROM pg_constraint
--     WHERE conrelid = {PSTA}::regclass AND contype='u' AND conname='uniq_shop_iphone_Timestamp_Time'
--   ) THEN
--     ALTER TABLE {PSTA} DROP CONSTRAINT IF EXISTS "<old_name>";
--     ALTER TABLE {PSTA} ADD CONSTRAINT "uniq_shop_iphone_Timestamp_Time"
--       UNIQUE ("shop_id","iphone_id","Timestamp_Time");
--   END IF;
-- END $$;
"""

class Migration(migrations.Migration):
    dependencies = [
        ('AppleStockChecker', '0005_cohort_modelartifact_overallbar_featuresnapshot_and_more'),
    ]
    operations = [
        migrations.RunSQL(SQL),
    ]