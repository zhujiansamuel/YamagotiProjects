"""
Tests for shop4 detect_color_only_filter().

Covers three detection modes:
  1. Bare color names (裸色名)
  2. のみ suffix
  3. Parenthetical delta pattern (括号)
  + 全色 delta stacking logic in clean_shop4 integration
"""
from __future__ import annotations

import os
import sys

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

from AppleStockChecker.utils.external_ingest.shop_cleaners_split.shop4_cleaner import (
    detect_color_only_filter,
)
from AppleStockChecker.utils.external_ingest.cleaner_tools import (
    _label_matches_color_unified,
)

# ── Fixtures ──────────────────────────────────────────────────────────

# Simulated color_to_pn for iPhone 17 Pro Max 256GB
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


# ── 1. Bare color names ──────────────────────────────────────────────

class TestBareColorNames:
    def test_single_bare_color(self):
        """単一の裸色名 → color_only_mode=True, delta=0"""
        mode, specs = detect_color_only_filter(
            "コズミックオレンジ", COLOR_MAP, MATCHER,
        )
        assert mode is True
        assert len(specs) == 1
        assert specs[0][0] == "コズミックオレンジ"
        assert specs[0][1] == 0
        assert specs[0][2] is False  # has_explicit_delta

    def test_multiple_bare_colors_slash(self):
        """スラッシュ区切りの複数裸色名"""
        mode, specs = detect_color_only_filter(
            "シルバー/ディープブルー", COLOR_MAP, MATCHER,
        )
        assert mode is True
        assert len(specs) == 2
        labels = {s[0] for s in specs}
        assert "シルバー" in labels
        assert "ディープブルー" in labels

    def test_not_bare_color_with_delta(self):
        """価格情報がある → color_only ではない"""
        mode, specs = detect_color_only_filter(
            "ブラック-1,000円", COLOR_MAP, MATCHER,
        )
        assert mode is False
        assert specs == []

    def test_not_bare_color_with_nashi(self):
        """なし表記 → color_only ではない（通常フロー）"""
        mode, specs = detect_color_only_filter(
            "ブラックチタニウムなし", COLOR_MAP, MATCHER,
        )
        assert mode is False
        assert specs == []

    def test_unknown_label_not_bare_color(self):
        """color_map に存在しないラベル → False"""
        mode, specs = detect_color_only_filter(
            "パープル", COLOR_MAP, MATCHER,
        )
        assert mode is False
        assert specs == []

    def test_empty_text(self):
        mode, specs = detect_color_only_filter("", COLOR_MAP, MATCHER)
        assert mode is False
        assert specs == []

    def test_none_text(self):
        mode, specs = detect_color_only_filter(None, COLOR_MAP, MATCHER)
        assert mode is False
        assert specs == []


# ── 2. のみ suffix ───────────────────────────────────────────────────

class TestNomiSuffix:
    def test_single_color_nomi(self):
        """「コズミックオレンジのみ」"""
        mode, specs = detect_color_only_filter(
            "コズミックオレンジのみ", COLOR_MAP, MATCHER,
        )
        assert mode is True
        assert len(specs) == 1
        assert specs[0][0] == "コズミックオレンジ"
        assert specs[0][1] == 0
        assert specs[0][2] is False

    def test_multi_color_nomi(self):
        """「シルバー/ディープブルーのみ」"""
        mode, specs = detect_color_only_filter(
            "シルバー/ディープブルーのみ", COLOR_MAP, MATCHER,
        )
        assert mode is True
        assert len(specs) == 2
        labels = {s[0] for s in specs}
        assert "シルバー" in labels
        assert "ディープブルー" in labels

    def test_nomi_with_zencolor_prefix(self):
        """「全色-2000 / コズミックオレンジのみ」→ 全色除去後にのみ検出"""
        mode, specs = detect_color_only_filter(
            "全色-2000 / コズミックオレンジのみ", COLOR_MAP, MATCHER,
        )
        assert mode is True
        assert len(specs) == 1
        assert specs[0][0] == "コズミックオレンジ"
        assert specs[0][2] is False  # will stack with 全色


# ── 3. Parenthetical pattern ─────────────────────────────────────────

class TestParenthetical:
    def test_single_inner_color(self):
        """「シルバー（コズミックオレンジ-2,500円）」"""
        mode, specs = detect_color_only_filter(
            "シルバー(コズミックオレンジ-2,500円)", COLOR_MAP, MATCHER,
        )
        assert mode is True
        assert len(specs) == 2

        outer = [s for s in specs if s[0] == "シルバー"]
        assert len(outer) == 1
        assert outer[0][1] == 0           # base price
        assert outer[0][2] is False       # not explicit delta

        inner = [s for s in specs if s[0] == "コズミックオレンジ"]
        assert len(inner) == 1
        assert inner[0][1] == -2500       # explicit delta
        assert inner[0][2] is True        # has_explicit_delta

    def test_multiple_inner_colors(self):
        """「シルバー(コズミックオレンジ-2,500円、ブラックチタニウム-1,000円)」"""
        mode, specs = detect_color_only_filter(
            "シルバー(コズミックオレンジ-2,500円、ブラックチタニウム-1,000円)",
            COLOR_MAP,
            MATCHER,
        )
        assert mode is True
        assert len(specs) == 3

        labels_deltas = {s[0]: (s[1], s[2]) for s in specs}
        assert labels_deltas["シルバー"] == (0, False)
        assert labels_deltas["コズミックオレンジ"] == (-2500, True)
        assert labels_deltas["ブラックチタニウム"] == (-1000, True)

    def test_fullwidth_parens(self):
        """全角括号もNFKCで半角に変換される前提でテスト"""
        # normalize_text_basic converts （）to ()
        mode, specs = detect_color_only_filter(
            "シルバー（コズミックオレンジ-2,500円）", COLOR_MAP, MATCHER,
        )
        assert mode is True
        assert len(specs) == 2

    def test_positive_inner_delta(self):
        """括号内プラス方向: 「シルバー(コズミックオレンジ+1,000円)」"""
        mode, specs = detect_color_only_filter(
            "シルバー(コズミックオレンジ+1,000円)", COLOR_MAP, MATCHER,
        )
        assert mode is True
        inner = [s for s in specs if s[0] == "コズミックオレンジ"]
        assert len(inner) == 1
        assert inner[0][1] == 1000
        assert inner[0][2] is True


# ── 4. 全色 delta stacking logic ─────────────────────────────────────

class TestZenshokuStacking:
    """
    全色 delta stacking is applied in clean_shop4, not in detect_color_only_filter.
    These tests verify the specs are correctly structured for downstream stacking.
    """

    def test_bare_color_specs_for_stacking(self):
        """裸色名 specs have has_explicit_delta=False → will stack with 全色"""
        _, specs = detect_color_only_filter(
            "コズミックオレンジ", COLOR_MAP, MATCHER,
        )
        assert all(not s[2] for s in specs)

    def test_paren_outer_stackable_inner_overrides(self):
        """括号: outer=stackable, inner=override"""
        _, specs = detect_color_only_filter(
            "シルバー(コズミックオレンジ-2,500円)", COLOR_MAP, MATCHER,
        )
        outer = [s for s in specs if s[0] == "シルバー"][0]
        inner = [s for s in specs if s[0] == "コズミックオレンジ"][0]
        assert outer[2] is False   # stackable
        assert inner[2] is True    # overrides 全色

    def test_stacking_simulation(self):
        """Simulate the stacking logic from clean_shop4"""
        _, specs = detect_color_only_filter(
            "シルバー(コズミックオレンジ-2,500円)", COLOR_MAP, MATCHER,
        )
        agg_all_delta = -2000  # simulated 全色-2000

        co_delta_specs = []
        for lbl, delta, has_explicit in specs:
            if has_explicit:
                co_delta_specs.append((lbl, delta))
            else:
                effective = agg_all_delta if agg_all_delta is not None else delta
                co_delta_specs.append((lbl, effective))

        result = dict(co_delta_specs)
        assert result["シルバー"] == -2000       # stacked with 全色
        assert result["コズミックオレンジ"] == -2500  # explicit, overrides 全色


# ── 5. Edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_regular_delta_not_color_only(self):
        """通常の delta パターン → color_only ではない"""
        mode, _ = detect_color_only_filter(
            "ブラックチタニウム-1000 / シルバーなし",
            COLOR_MAP,
            MATCHER,
        )
        assert mode is False

    def test_all_color_only_not_color_only(self):
        """全色のみ（具体色なし）→ False"""
        mode, specs = detect_color_only_filter(
            "全色-2000", COLOR_MAP, MATCHER,
        )
        assert mode is False

    def test_mixed_separator_comma(self):
        """読点区切り「シルバー、ディープブルーのみ」"""
        mode, specs = detect_color_only_filter(
            "シルバー、ディープブルーのみ", COLOR_MAP, MATCHER,
        )
        assert mode is True
        assert len(specs) == 2
