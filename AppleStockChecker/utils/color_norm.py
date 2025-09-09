# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from typing import Dict, List, Tuple

# 视作“全色”的关键词（大小写/全半角不敏感）
ALL_COLOR_TOKENS = {
    "全色", "全カラー", "全部", "任意", "不限", "任何", "ALL", "All", "all", "any", "any color",
    "任选", "カラー問わず", "すべて", "全系", "全机型色"
}

# 规范色名（你可以按需要扩充或改为你库里实际的规范写法）
CANON_COLORS = [
    "Black", "White", "Blue", "Green", "Pink", "Yellow", "Red",
    "Gold", "Silver", "Gray", "Purple", "Starlight", "Midnight",
    "Graphite", "Space Gray",
    "Natural Titanium", "Black Titanium", "White Titanium", "Blue Titanium", "Desert Titanium",
]

# 同义词词表（全部小写比较，匹配采用“包含即可”策略，避免完全相等的苛刻匹配）
COLOR_SYNONYMS: Dict[str, List[str]] = {
    # 基础色
    "Black":   ["black", "ブラック", "黒", "黑", "黑色", "曜石黑"],
    "White":   ["white", "ホワイト", "白", "白色"],
    "Blue":    ["blue", "ブルー", "青", "蓝", "蓝色", "青色", "遠峰藍", "远峰蓝", "天蓝"],
    "Green":   ["green", "グリーン", "緑", "绿", "绿色", "緑色"],
    "Pink":    ["pink", "ピンク", "粉", "粉色", "玫瑰粉"],
    "Yellow":  ["yellow", "イエロー", "黄", "黄色"],
    "Red":     ["red", "レッド", "紅", "红", "红色", "product red", "(product)red", "プロダクトレッド", "特别版红"],
    "Gold":    ["gold", "ゴールド", "金", "金色", "香槟金", "香檳金"],
    "Silver":  ["silver", "シルバー", "銀", "银", "銀色", "银色"],
    "Gray":    ["gray", "grey", "グレー", "グレイ", "灰", "灰色"],
    "Purple":  ["purple", "パープル", "紫", "紫色", "ディープパープル", "深紫"],
    "Graphite":["graphite", "グラファイト", "石墨", "石墨色"],
    "Space Gray": ["space gray", "スペースグレイ", "スペースグレー", "深空灰", "深空灰色"],

    # iPhone 13/14 的“星光/午夜”
    "Starlight": ["starlight", "スターライト", "星光", "星光色"],
    "Midnight":  ["midnight", "ミッドナイト", "午夜", "午夜色", "暗夜色"],

    # 15 Pro / 16 Pro 钛金属家族
    "Natural Titanium": ["natural titanium", "ナチュラルチタニウム", "ナチュラル", "自然钛", "自然色", "自然鈦"],
    "Black Titanium":   ["black titanium", "ブラックチタニウム", "黑钛", "黑色钛"],
    "White Titanium":   ["white titanium", "ホワイトチタニウム", "白钛", "白色钛"],
    "Blue Titanium":    ["blue titanium",  "ブルーチタニウム", "蓝钛", "蓝色钛", "藍鈦"],
    # 16 Pro 新色一类（沙/沙岩/沙漠）
    "Desert Titanium":  ["desert titanium", "デザートチタニウム", "サンド", "砂色", "沙色", "沙岩色", "沙漠色"],
}

# 预编译：把所有同义词做成小写集合
LOWER_SYNONYMS = {
    canon: {s.lower() for s in syns} | {canon.lower()} for canon, syns in COLOR_SYNONYMS.items()
}

def _norm_text(s: str) -> str:
    if not s: return ""
    t = str(s)
    t = t.replace("（", "(").replace("）", ")")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def is_all_color(s: str) -> bool:
    if not s: return True  # 无颜色信息视为“全色”
    t = _norm_text(s).lower()
    for token in ALL_COLOR_TOKENS:
        if token.lower() in t:
            return True
    return False

def normalize_color(s: str) -> Tuple[str, bool]:
    """
    返回 (规范色名, is_all)；若无法识别，规范色名返回空串。
    规则：先判断“全色”，否则在同义词中找“包含”命中。
    """
    if is_all_color(s):
        return ("", True)
    t = _norm_text(s).lower()
    for canon, syns in LOWER_SYNONYMS.items():
        if any(ss in t for ss in syns):
            return (canon, False)
    return ("", False)

def synonyms_for_query(canon: str) -> List[str]:
    """给定规范色名，返回用于 DB 查询的候选写法（包含规范名本身）"""
    if not canon: return []
    return sorted(LOWER_SYNONYMS.get(canon, {canon.lower()}))
