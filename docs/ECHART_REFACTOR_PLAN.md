# EChart.html リファクタリング計画

## 概要

`templates/apple_stock/EChart.html` (2529行) の大規模リファクタリング計画書

**目標**:
1. 全JavaScript関数に詳細なJSDocコメントを追加
2. 関連する関数をグループ化して配置
3. 設定パラメータをJavaScriptセクションの冒頭に抽出
4. コードの可読性と保守性を向上

---

## ファイル構造分析

### 現在の構造

```
行1-246:    HTML head (スタイル定義)
行247-452:  HTML body (UI要素、フォーム、チャート領域)
行453-2526: JavaScript (メインロジック)
行2527-2529: 閉じタグ
```

### JavaScript セクション構成 (現状)

| 行範囲 | 内容 |
|--------|------|
| 454-512 | グローバル変数・定数宣言 |
| 516-522 | タイムゾーンユーティリティ (utcMsToJstDate) |
| 524-547 | Y軸自動調整関数 (makeAutoY) |
| 549-628 | Legend自動調整関数群 |
| 631-643 | 日付ユーティリティ (floorDay, ceilDay) |
| 645-663 | 価格フィルタ定数・フォーマッタ |
| 665-705 | 統計ユーティリティ関数群 |
| 707-725 | 時刻変換関数 |
| 727-815 | 統計チャート初期化・関数群 |
| 830-870 | チャート拡張機能 (日付目盛、営業時間オーバーレイ、色管理) |
| 882-1017 | チャートパネル作成・初期化関数群 |
| 1018-1070 | 営業時間マークエリア作成関数群 |
| 1072-1122 | データ集約・レンダリング関数 |
| 1125-1201 | Part別チャート管理関数群 |
| 1203-1298 | メインチャート操作関数 |
| 1301-1365 | データローディング関数 (shops, iphones) |
| 1367-1427 | データフェッチ関数群 |
| 1429-1505 | Scopes抽出関数群 |
| 1507-1720 | UI マウント関数群 (datalist) |
| 1722-1812 | Feature関連ユーティリティ |
| 1815-2043 | Feature描画・更新関数群 |
| 2045-2076 | 並行実行制御・Scopes準備関数 |
| 2079-2430 | 5種類のロードハンドラ (handleLoadS1~S5) |
| 2433-2525 | イベントバインディング・初期化 |

---

## リファクタリング後の理想構造

### セクション1: 設定パラメータ (Configuration)

```javascript
/* ========================================
 * 📋 設定パラメータ (Configuration Parameters)
 * ======================================== */

// === API エンドポイント ===
const SERIES_API_BASE = "/AppleStockChecker/export/purchasing-shop-time-analysis-per-iphone/";
const ENDPOINT_SHOPS = "/AppleStockChecker/shops/";
const ENDPOINT_IPHONES = "/AppleStockChecker/iphones/";
const ENDPOINT_SCOPES = "/AppleStockChecker/export/feature-scopes/";
const FEATURE_API_BASE = "/AppleStockChecker/export/purchasing-shop-time-analysis-feature-points/";

// === 時間・タイムゾーン ===
const JST_OFFSET_MS = 9 * 60 * 60 * 1000;  // +09:00
const ONE_DAY = 24 * 3600 * 1000;           // 1日のミリ秒

// === 価格フィルタ ===
const PRICE_VISIBLE_MIN = 100000;  // 10万円
const PRICE_VISIBLE_MAX = 400000;  // 40万円

// === チャート表示設定 ===
const YPAD_RATIO = 0.1;            // Y軸の上下留白 (10%)

// === データフェッチ制御 ===
const RAW_MAX_SERIES = 60;         // 最大系列数
const RAW_CONC_LIMIT = 8;          // 並行フェッチ制限

// === カラーパレット ===
const SHOP_COLOR_RULES = [
    {match: /買取商店/, color: "#DD1133", shop_id: 14, order: 1},
    // ... (既存のルール)
];
const LINE_PALETTE = ["#2A9D8F", "#E76F51", "#F4A261", "#E9C46A", "#264653", ...];

// === Feature名デフォルト ===
const FEATURE_NAMES_DEFAULT = ['mean', 'std'];
```

### セクション2: グローバル変数 (Global State)

```javascript
/* ========================================
 * 🌍 グローバル状態 (Global State)
 * ======================================== */

// === データインデックス ===
let SHOPS_INDEX = new Map();        // shop_id -> shop_name
let IPHONE_ID_BY_PART = new Map();  // part_number -> iphone_id
let IPHONE_INFO_BY_PART = new Map(); // part_number -> {id, model_name, capacity_gb, color, ...}

// === Scopes データ ===
let SHOP_PROFILES = [];             // [{id, slug, title, label, items:[{shop_id, shop_name, ...}], ...}]
let COHORTS = [];                   // [{id, slug, title, label, members:[{iphone_id, part_number, ...}]}]
let PROFILE_BY_ID = new Map();
let COHORT_BY_ID = new Map();
let _SCOPES_CACHE = null;

// === 現在の選択状態 ===
let ACTIVE_PART_NUMBER = '';
let ACTIVE_IPHONE_ID = null;
let SELECTED_SHOP_IDS = new Set();
let SELECTED_START_MS = null;
let SELECTED_END_MS = null;

// === タイマー ===
let END_AUTO_TIMER = null;
```

### セクション3: ユーティリティ関数 (Utilities)

**3.1 DOM操作**
- `$(id)` - getElementById shorthand

**3.2 日付・時刻**
- `utcMsToJstDate(ms)` - UTC ms を JST の Date に変換
- `floorDay(ms)` - 日付を0時に切り下げ
- `ceilDay(ms)` - 次の日の0時に切り上げ
- `toMillis(s)` - ISO文字列をミリ秒に変換
- `toIsoLocal(dtStr)` - ローカルJST文字列をISO UTC文字列に変換
- `toInputValueJST(dateUtc)` - UTC DateをJST入力値形式に変換
- `floorToMinute(date)` - 分単位に切り下げ
- `nearestPastMinuteMs(d)` - 最も近い過去の分のミリ秒
- `nearestPastMinuteInputValueJST()` - 最も近い過去の分のJST入力値

**3.3 フォーマット**
- `fmtJPY(n)` - 円通貨フォーマット
- `verticalTickFormatter(ms)` - 縦書き日付フォーマット (月\n日\n(曜))
- `iphoneHumanLabelByPart(pn)` - Part NumberからiPhoneの人間可読ラベルを生成
- `humanNameOfPart(partNumber, {withPn})` - Part Numberの人間可読名

**3.4 配列・統計**
- `_yOf(item)` - データポイントからY値を抽出
- `_latestY(data)` - 系列から最後の有効なY値を取得
- `_median(sortedNums)` - ソート済み配列の中央値
- `arrFirst(obj, keys)` - オブジェクトから最初に見つかった配列値を返す

### セクション4: チャート設定関数 (Chart Configuration)

**4.1 基本設定**
- `makeAutoY({name, position, clamp})` - Y軸自動スケール設定
- `baseOptionForTimeseries(yName)` - 時系列チャートの基本オプション
- `baseOptionForPrice()` - 価格チャートの基本オプション
- `baseOptionForFeature()` - Featureチャートの基本オプション

**4.2 Legend調整**
- `_px(v, fallback)` - px文字列を数値に変換
- `_legendOpt(inst)` - EChartsインスタンスからlegendオプションを取得
- `_gridOpt(inst)` - EChartsインスタンスからgridオプションを取得
- `estimateLegendRows(inst)` - Legend行数を推定
- `adjustGridTopForLegend(inst)` - Legendに合わせてgrid.topを調整

**4.3 視覚的強化**
- `ensureDailyTicks()` - メインチャートに日付目盛を追加
- `ensureDailyTicksOn(inst, {minMs, maxMs})` - 指定チャートに日付目盛
- `ensureDailyTicksAll()` - 全チャートに日付目盛
- `buildBusinessHoursMarkArea10to19(startMs, endMs)` - 営業時間マークエリア作成
- `applyWorkHoursOverlay(startMs, endMs)` - メインチャートに営業時間オーバーレイ
- `applyWorkHoursOverlayOn(inst, startMs, endMs)` - 指定チャートに営業時間オーバーレイ
- `applyWorkHoursOverlayAll(startMs, endMs)` - 全チャートに営業時間オーバーレイ

### セクション5: 店舗・カラー管理 (Shop & Color Management)

- `getRuleForShop({name, id})` - 店舗のカラールールを取得
- `getOrderForShop({name, id})` - 店舗の表示順序を取得
- `getColorForShop({name, id}, idx)` - 店舗のカラーを取得

### セクション6: チャート管理 (Chart Management)

**6.1 パネル作成**
- `createPanel(host, titleText, isFeature)` - チャートパネルDOM作成
- `initChartHostsOnce()` - チャートホストを初期化 (一度のみ)
- `createPanelAndInitChart(host, title, isFeature)` - パネル作成+ECharts初期化

**6.2 Part別チャート**
- `titleForPrice(partNumber)` - 価格チャートのタイトル
- `titleForFeature(partNumber)` - Featureチャートのタイトル
- `ensurePriceChartForPart(partNumber)` - Part別価格チャートを取得/作成
- `ensureFeatureChartForPart(partNumber)` - Part別Featureチャートを取得/作成
- `disposeAllPartCharts()` - 全Part別チャートを破棄

**6.3 メインチャート操作**
- `getMainChart()` - メインチャートインスタンスを取得/作成
- `addOrUpdateSeries(id, name, color, data)` - メインチャートに系列追加/更新
- `addOrUpdateSeriesForPart(part, id, name, color, data)` - Part別チャートに系列追加

### セクション7: 統計チャート (Stats Chart)

- `computeStatsFromMainChart()` - メインチャートから統計を計算
- `renderStatsChart(stats)` - 統計チャートをレンダリング
- `updateStatsChart()` - 統計チャートを更新

### セクション8: データローディング (Data Loading)

**8.1 基本データ**
- `loadShops()` - 全店舗を読み込み、UIにマウント
- `loadIphones()` - 全iPhoneを読み込み、datalistにマウント

**8.2 Scopes**
- `loadScopesOnce()` - Scopesデータを読み込み (キャッシュ付き)
- `extractProfilesFromScopes(scopes)` - Scopesから店舗プロファイルを抽出
- `extractCohortsFromScopes(scopes)` - ScopesからiPhoneコホートを抽出
- `extractUniqueIphonesFromScopes(scopes)` - ScopesからユニークなiPhone一覧を抽出
- `ensureScopesReady()` - Scopesデータが準備されていることを保証

**8.3 時系列データ**
- `buildURL(base, startIsoJST, endIsoJST, shopId, partNumber)` - API URLを構築
- `fetchOneShopSeries(base, startIsoJST, endIsoJST, shopId, shopName, partNumber)` - 1店舗の時系列を取得

**8.4 Feature データ**
- `buildFeatureScope({shopId, profileSlug, iphoneId})` - Feature scope オブジェクトを構築
- `fetchFeaturePointsSimple(scope, name, startUtc, endUtc)` - Feature ポイントをフェッチ (シンプル版)
- `fetchFeatureSeries({scope, name, startIso, endIso})` - Feature 系列をフェッチ

### セクション9: データ処理 (Data Processing)

- `appendShadowToNowIfNeeded(seriesData)` - 現在時刻まで影のポイントを追加
- `mergeSeriesAverage(seriesList)` - 複数系列の平均を計算
- `renderAggregateOnMain(aggByShopMap, startMs, endMs)` - 店舗別集約をメインチャートに描画
- `applyFeatureMetricParam(u, metricKey)` - Feature メトリックをURLパラメータに適用

### セクション10: Feature関連 (Feature Functions)

- `buildMeanStdBand(meanData, stdData)` - Mean±Stdのバンドデータを構築
- `bandSeriesForMeanStd(legendBase, color, meanData, stdData)` - Mean±Std のECharts系列配列を生成
- `drawFeatureLinesForPart(partNumber, scope, names, startUtc, endUtc)` - Part別Featureラインを描画
- `updateFeatureForScopePart({partNumber, scope, startIso, endIso, names})` - Scope+Part のFeatureを更新
- `updateFeatureSnapshot({seriesInputs, title, startIso, endIso})` - FeatureスナップショットをメインFeatureチャートに描画

### セクション11: UI マウント (UI Mounting)

- `mountDatalistSingle({inputId, listId, hintId, hiddenId, swatchId, items, valueBuilder, colorResolver, nameResolver, parseByPattern, labelBuilder})` - 単一選択datalistをマウント
- `mountAllScopeSelectors()` - 全Scopeセレクタ (A/B/C/D/E) をマウント

### セクション12: 並行実行制御 (Concurrency Control)

- `runWithLimit(tasks, limit)` - 並行数を制限してタスクを実行

### セクション13: メインロードハンドラ (Main Load Handlers)

- `handleLoadS1()` - パターンA: 単一店舗 × 単一iPhone
- `handleLoadS2()` - パターンB: 全店舗 × iPhoneコホート
- `handleLoadS3()` - パターンC: 店舗プロファイル × 単一iPhone
- `handleLoadS4()` - パターンD: 単一店舗 × iPhoneコホート
- `handleLoadS5()` - パターンE: 店舗プロファイル × iPhoneコホート

### セクション14: 時刻同期・初期化 (Time Sync & Initialization)

- `syncAllEndToNow()` - 全終了時刻入力を現在時刻に同期
- `startAutoEndSync()` - 終了時刻の自動同期を開始
- `initDefaults()` - ページデフォルト値を初期化 (IIFE)

### セクション15: イベントバインディング (Event Bindings)

- Reset ボタン
- A/B/C/D/E ロードボタン
- Select All / Clear All ボタン
- Window resize イベント

---

## JSDoc コメントテンプレート

各関数に以下の形式でJSDocを追加:

```javascript
/**
 * 関数の簡潔な説明
 *
 * より詳細な説明（必要に応じて）
 *
 * @param {型} パラメータ名 - パラメータの説明
 * @returns {型} 戻り値の説明
 * @throws {エラー型} エラー発生条件
 * @example
 * // 使用例
 * const result = functionName(arg1, arg2);
 */
function functionName(arg1, arg2) {
    // ...
}
```

### 具体例

```javascript
/**
 * UTC ミリ秒を JST の Date オブジェクトに変換
 *
 * ブラウザのローカルタイムゾーンに影響されずに、
 * 常に JST (+09:00) として日時を扱うための変換関数。
 * 変換後の Date オブジェクトは getUTC* メソッドで
 * JST の年月日時分秒を取得可能。
 *
 * @param {number} ms - UTC タイムスタンプ (ミリ秒)
 * @returns {Date} JST オフセット適用後の Date オブジェクト
 * @example
 * const jstDate = utcMsToJstDate(1640995200000);
 * console.log(jstDate.getUTCHours()); // JST の時刻
 */
function utcMsToJstDate(ms) {
    return new Date(ms + JST_OFFSET_MS);
}
```

---

## 実装手順

### フェーズ1: パラメータ抽出 (行454-520)

1. 全定数を「設定パラメータ」セクションに移動
2. グローバル変数を「グローバル状態」セクションに整理
3. セクション区切りコメントを追加

### フェーズ2: ユーティリティ関数整理 (行520-725)

1. 日付・時刻関数をグループ化
2. フォーマット関数をグループ化
3. 統計関数をグループ化
4. 各関数にJSDocコメント追加

### フェーズ3: チャート関連関数整理 (行727-1298)

1. チャート設定関数をグループ化
2. Legend/Grid調整関数をグループ化
3. パネル作成関数をグループ化
4. Part別チャート管理をグループ化
5. 各関数にJSDocコメント追加

### フェーズ4: データローディング整理 (行1301-1812)

1. 基本データローディングをグループ化
2. Scopes関連をグループ化
3. Feature関連をグループ化
4. 各関数にJSDocコメント追加

### フェーズ5: メインロジック整理 (行1815-2525)

1. ロードハンドラをグループ化
2. 並行実行制御を整理
3. UI マウント関数を整理
4. イベントバインディングを整理
5. 各関数にJSDocコメント追加

---

## 期待される効果

### Before (現状の課題)

- ❌ 関数の役割が不明瞭 (コメントなし)
- ❌ 関連する関数が離れた場所に配置
- ❌ 定数が複数箇所に散在
- ❌ 新規参画者が理解困難

### After (改善後)

- ✅ 全関数にJSDoc完備 → IDE補完・ホバー表示が有効
- ✅ 論理的なセクション分割 → 目的の関数を即座に発見
- ✅ 設定パラメータが一元管理 → 調整が容易
- ✅ 保守性・可読性が大幅向上

---

## リスク管理

### リスク

1. **動作への影響**: 関数の並び替えや整理中の誤り
2. **大規模変更**: 2529行の大規模修正によるバグ混入

### 対策

1. ✅ バックアップファイル作成済み (`EChart.html.backup`)
2. ⚠️ セクションごとに段階的にリファクタリング
3. ⚠️ 各フェーズ後に機能テスト実施
4. ⚠️ Git でコミットを細かく分ける

---

## 次のステップ

1. ✅ 本計画書のレビュー
2. ⏳ フェーズ1から順次実装
3. ⏳ 各フェーズ後の動作確認
4. ⏳ 完了後の総合テスト
5. ⏳ ドキュメント更新 (CLAUDE.md への記載)

---

生成日時: 2025-11-26
対象ファイル: `templates/apple_stock/EChart.html`
バックアップ: `templates/apple_stock/EChart.html.backup`
関数総数: 約80個
