"""
Tests for detect_color_only_filter() (shared in cleaner_tools).

Covers three detection modes:
  1. Bare color names (裸色名)
  2. のみ suffix
  3. Parenthetical delta pattern (括号)
  + apply_color_only_stacking() for 全色 delta stacking
"""
from __future__ import annotations

import re

import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DATABASES={},
        INSTALLED_APPS=["django.contrib.contenttypes"],
        DEFAULT_AUTO_FIELD="django.db.models.BigAutoField",
    )
    django.setup()

import pytest

from AppleStockChecker.utils.external_ingest.cleaner_tools import (
    detect_color_only_filter,
    apply_color_only_stacking,
    _label_matches_color_unified,
)

# ── Shop-specific functions (use shop4's as representative) ──────────

SPLIT_RE = re.compile(r"[／/、，]|(?:\s*;\s*)|\n")


def _normalize_label(lbl: str) -> str:
    if not lbl:
        return ""
    s = re.sub(r"[\s\u3000\xa0]+", "", str(lbl))
    s = re.sub(r"(カラー|色)$", "", s)
    return s.strip()


_BAD_WORDS = ("利用制限", "保証", "郵送", "持ち込み", "開始", "未満", "減額", "SIM", "制限")


def _is_plausible(label: str) -> bool:
    label = _normalize_label(label)
    if not label or label in ("全色", "ALL"):
        return False
    if label.startswith(("△", "▲")) or re.search(r"\d", label):
        return False
    if len(label) > 16 or any(w in label for w in _BAD_WORDS):
        return False
    return True


# ── Fixtures ──────────────────────────────────────────────────────────

COLOR_MAP = {
    "Black Titanium": ("MXX01J/A", "ブラックチタニウム"),
    "White Titanium": ("MXX02J/A", "ホワイトチタニウム"),
    "Natural Titanium": ("MXX03J/A", "ナチュラルチタニウム"),
    "Desert Titanium": ("MXX04J/A", "デザートチタニウム"),
    "Green Titanium": ("MXX05J/A", "グリーンチタニウム"),
    "Cosmic Orange": ("MXX06J/A", "コズミックオレンジ"),
    "Silver": ("MXX07J/A", "シルバー"),
    "Deep Blue": ("MXX08J/A", "ディープブルー"),
}

MATCHER = _label_matches_color_unified

# Common kwargs for detect_color_only_filter
KW = dict(
    split_tokens_re=SPLIT_RE,
    normalize_label_func=_normalize_label,
    is_plausible_label_func=_is_plausible,
)


def _detect(text):
    return detect_color_only_filter(text, COLOR_MAP, MATCHER, **KW)


# ── 1. Bare color names ──────────────────────────────────────────────

class TestBareColorNames:
    def test_single_bare_color(self):
        mode, specs = _detect("コズミックオレンジ")
        assert mode is True
        assert len(specs) == 1
        assert specs[0] == ("コズミックオレンジ", 0, False)

    def test_multiple_bare_colors_slash(self):
        mode, specs = _detect("シルバー/ディープブルー")
        assert mode is True
        assert len(specs) == 2
        labels = {s[0] for s in specs}
        assert "シルバー" in labels
        assert "ディープブルー" in labels

    def test_not_bare_color_with_delta(self):
        mode, specs = _detect("ブラック-1,000円")
        assert mode is False
        assert specs == []

    def test_not_bare_color_with_nashi(self):
        mode, specs = _detect("ブラックチタニウムなし")
        assert mode is False
        assert specs == []

    def test_unknown_label_not_bare_color(self):
        mode, specs = _detect("パープル")
        assert mode is False
        assert specs == []

    def test_empty_text(self):
        mode, specs = _detect("")
        assert mode is False
        assert specs == []

    def test_none_text(self):
        mode, specs = _detect(None)
        assert mode is False
        assert specs == []


# ── 2. のみ suffix ───────────────────────────────────────────────────

class TestNomiSuffix:
    def test_single_color_nomi(self):
        mode, specs = _detect("コズミックオレンジのみ")
        assert mode is True
        assert len(specs) == 1
        assert specs[0] == ("コズミックオレンジ", 0, False)

    def test_multi_color_nomi(self):
        mode, specs = _detect("シルバー/ディープブルーのみ")
        assert mode is True
        assert len(specs) == 2
        labels = {s[0] for s in specs}
        assert "シルバー" in labels
        assert "ディープブルー" in labels

    def test_nomi_with_zencolor_prefix(self):
        mode, specs = _detect("全色-2000 / コズミックオレンジのみ")
        assert mode is True
        assert len(specs) == 1
        assert specs[0][0] == "コズミックオレンジ"
        assert specs[0][2] is False


# ── 3. Parenthetical pattern ─────────────────────────────────────────

class TestParenthetical:
    def test_single_inner_color(self):
        mode, specs = _detect("シルバー(コズミックオレンジ-2,500円)")
        assert mode is True
        assert len(specs) == 2

        outer = [s for s in specs if s[0] == "シルバー"]
        assert outer[0] == ("シルバー", 0, False)

        inner = [s for s in specs if s[0] == "コズミックオレンジ"]
        assert inner[0] == ("コズミックオレンジ", -2500, True)

    def test_multiple_inner_colors(self):
        mode, specs = _detect(
            "シルバー(コズミックオレンジ-2,500円、ブラックチタニウム-1,000円)"
        )
        assert mode is True
        assert len(specs) == 3
        d = {s[0]: (s[1], s[2]) for s in specs}
        assert d["シルバー"] == (0, False)
        assert d["コズミックオレンジ"] == (-2500, True)
        assert d["ブラックチタニウム"] == (-1000, True)

    def test_fullwidth_parens(self):
        mode, specs = _detect("シルバー（コズミックオレンジ-2,500円）")
        assert mode is True
        assert len(specs) == 2

    def test_positive_inner_delta(self):
        mode, specs = _detect("シルバー(コズミックオレンジ+1,000円)")
        assert mode is True
        inner = [s for s in specs if s[0] == "コズミックオレンジ"]
        assert inner[0] == ("コズミックオレンジ", 1000, True)


# ── 4. apply_color_only_stacking ─────────────────────────────────────

class TestApplyColorOnlyStacking:
    def test_no_all_delta(self):
        specs = [("シルバー", 0, False), ("コズミックオレンジ", -2500, True)]
        result = apply_color_only_stacking(specs, agg_all_delta=None)
        assert result == [("シルバー", 0), ("コズミックオレンジ", -2500)]

    def test_with_all_delta(self):
        specs = [("シルバー", 0, False), ("コズミックオレンジ", -2500, True)]
        result = apply_color_only_stacking(specs, agg_all_delta=-2000)
        assert result == [("シルバー", -2000), ("コズミックオレンジ", -2500)]

    def test_all_bare_with_delta(self):
        specs = [("シルバー", 0, False), ("ディープブルー", 0, False)]
        result = apply_color_only_stacking(specs, agg_all_delta=-1500)
        assert result == [("シルバー", -1500), ("ディープブルー", -1500)]

    def test_stacking_end_to_end(self):
        """End-to-end: detect → stack → verify final delta_specs"""
        _, specs = _detect("シルバー(コズミックオレンジ-2,500円)")
        result = dict(apply_color_only_stacking(specs, agg_all_delta=-2000))
        assert result["シルバー"] == -2000
        assert result["コズミックオレンジ"] == -2500


# ── 5. Edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_regular_delta_not_color_only(self):
        mode, _ = _detect("ブラックチタニウム-1000 / シルバーなし")
        assert mode is False

    def test_all_color_only_not_color_only(self):
        mode, specs = _detect("全色-2000")
        assert mode is False

    def test_mixed_separator_comma(self):
        mode, specs = _detect("シルバー、ディープブルーのみ")
        assert mode is True
        assert len(specs) == 2
