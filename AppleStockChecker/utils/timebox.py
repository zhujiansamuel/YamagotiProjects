# AppleStockChecker/utils/timebox.py
from django.utils import timezone
from datetime import timedelta

def nearest_past_minute_aware(tz=None):
    """
    返回当前时刻在指定时区的“最近的过去整分钟”（秒=0, 微秒=0）的 aware datetime。
    - tz 为空时使用 Django 当前时区（settings.TIME_ZONE，一般是 Asia/Tokyo）
    - 如果刚好在整分钟上，就返回“当前这一分钟”（不再往前减 60s）
    """
    tz = tz or timezone.get_current_timezone()
    now = timezone.now().astimezone(tz)
    # 直接“截断到分钟”
    floored = now.replace(second=0, microsecond=0)
    # 如果你希望“严格过去”（即 00 秒时也回退 1 分钟），改成：
    if now.second == 0 and now.microsecond == 0:
        floored -= timedelta(minutes=1)
    return floored

def nearest_past_minute_iso(tz=None) -> str:
    """ISO8601（秒精度）字符串"""
    return nearest_past_minute_aware(tz).isoformat(timespec="seconds")
