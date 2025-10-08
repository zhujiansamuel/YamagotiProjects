import os, json, re, hashlib, sys
from io import BytesIO
from urllib.parse import urljoin
import requests
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image, ImageOps, ImageDraw
import imagehash
import csv, re
import pytz
import datetime
# ========= 配置 =========
BASE = "https://www.kaitorishouten-co.jp"
CID  = "709"  # ← 改成你的分类ID（例如 iPhone 17 Pro Max = 711）
CAT_URL = f"{BASE}/category/1/{CID}"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "ja,en;q=0.9,zh-CN;q=0.8"}
CACHE_SPRITE2IDXCHAR = "sprite_idxchar_by_md5.json"   # 雪碧图MD5 → {idx:char}
TEMPLATE_DB = "digit_templates.json"                  # 全局模板库：phash(hex) → {'ch': '0', 'w':10,'h':16}
TEMPLATE_IMG_DIR = "digit_templates"                  # 可视化保存每个模板的小图
os.makedirs(TEMPLATE_IMG_DIR, exist_ok=True)

LIST_URL = f"{BASE}/category/1/{CID}"
# =======================

def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def extract_csrf_token(html: str) -> str | None:
    soup = BeautifulSoup(html, 'html.parser')
    # 1) 精确：<meta name="eccube-csrf-token" content="...">
    m = soup.find('meta', attrs={'name': lambda v: v and v.lower()=='eccube-csrf-token'})
    if m and m.get('content'):
        return m['content']
    # 2) 宽松：任何 name 含 csrf 的 meta
    m = soup.find('meta', attrs={'name': lambda v: v and 'csrf' in v.lower()})
    if m and m.get('content'):
        return m['content']
    # 3) 正则兜底（属性顺序/单双引号）
    r = re.search(r'name=["\']eccube-csrf-token["\'][^>]*content=["\']([^"\']+)["\']', html, re.I)
    if r: return r.group(1)
    r = re.search(r'content=["\']([^"\']+)["\'][^>]*name=["\']eccube-csrf-token["\']', html, re.I)
    if r: return r.group(1)
    return None

def extract_sprite_info(soup):
    # 从 <style> 找到 .encrypt-num 的 background-image 与宽高
    css = "\n".join((s.string or s.text or "") for s in soup.find_all("style"))
    m_url = re.search(r'\.encrypt-num[^}]*background-image\s*:\s*url\((["\']?)([^)\'"]+)\1\)', css)
    if not m_url:
        return None, 10, 16
    sprite_url = urljoin(BASE, m_url.group(2).strip())
    m_w = re.search(r'\.encrypt-num[^}]*width\s*:\s*(\d+)px', css)
    m_h = re.search(r'\.encrypt-num[^}]*height\s*:\s*(\d+)px', css)
    cell_w = int(m_w.group(1)) if m_w else 10
    cell_h = int(m_h.group(1)) if m_h else 16
    return sprite_url, cell_w, cell_h

def open_image_bytes(raw, content_type_hint=""):
    # 打开 PNG/WEBP/AVIF（若为 AVIF 需 pillow-heif 或 pillow-avif-plugin）
    try:
        img = Image.open(BytesIO(raw)); img.load()
        return img.convert("RGBA")
    except Exception:
        # AVIF 兜底
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
            img = Image.open(BytesIO(raw)); img.load()
            return img.convert("RGBA")
        except Exception:
            try:
                import pillow_avif  # noqa
                img = Image.open(BytesIO(raw)); img.load()
                return img.convert("RGBA")
            except Exception as e2:
                raise RuntimeError("无法打开图片，若为 AVIF 请安装 pillow-heif 或 pillow-avif-plugin") from e2

def download_sprite(sess, list_url, sprite_url):
    r = sess.get(sprite_url, headers={
        **HEADERS,
        "Referer": list_url,
        "Accept": "image/webp,image/png,*/*;q=0.8",  # 尽量避免 AVIF
    }, timeout=30)
    r.raise_for_status()
    raw = r.content
    ct = r.headers.get("Content-Type","").lower()
    if (ct.startswith("text/html") or not ct) and (raw[:64].lstrip().startswith(b"<!") or b"<html" in raw[:256].lower()):
        open("sprite_debug.html","wb").write(raw)
        raise RuntimeError("拿到 HTML（反盗链/旧 token）。已保存 sprite_debug.html")
    return open_image_bytes(raw, ct), raw, ct

def cut_row_tiles(sprite_img: Image.Image, cell_w: int, cell_h: int) -> list[Image.Image]:
    """假设单行雪碧图；如果是多行，可自行扩展循环 y 行"""
    cols = sprite_img.width // cell_w
    tiles = []
    for c in range(cols):
        x0 = c*cell_w
        tiles.append(sprite_img.crop((x0, 0, x0+cell_w, cell_h)))
    return tiles

def tiles_phash(tiles):
    """对每个小格计算 pHash（16x16），返回 list[str(hex)]"""
    hashes = []
    for t in tiles:
        # 先转灰度；pHash 对轻微压缩/噪声鲁棒
        h = imagehash.phash(ImageOps.grayscale(t), hash_size=8)  # 64-bit
        hashes.append(h.__str__())
    return hashes

def load_json(path, default):
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def show_and_label_once(tiles, cell_w, cell_h):
    """首次遇到某张雪碧图时，输出预览让你一次性标注 11 个字符"""
    grid = Image.new("RGB", (len(tiles)*cell_w, cell_h), "white")
    draw = ImageDraw.Draw(grid)
    x = 0
    for i, t in enumerate(tiles):
        grid.paste(t.convert("RGB"), (x,0))
        draw.text((x+1,1), str(i), fill=(255,0,0))
        x += cell_w
    big = grid.resize((grid.width*8, grid.height*8), Image.NEAREST)
    big.save("sprite_tiles_preview.png")
    print("请打开 sprite_tiles_preview.png，对照从左到右输入每格字符（仅 0-9 与 ,）")
    idx2char = {}
    for i in range(len(tiles)):
        while True:
            ch = input(f"第 {i} 格 = ").strip()
            if ch in "0123456789,":
                idx2char[str(i)] = ch
                break
            print("只能输入 0-9 或 ,")
    return idx2char

def build_or_update_template_db(tiles, labels, db_path=TEMPLATE_DB):
    """把本次标注的每个 tile 作为“字符模板”存进全局库（pHash → char）"""
    db = load_json(db_path, {})
    for i, t in enumerate(tiles):
        ch = labels[str(i)]
        h = imagehash.phash(ImageOps.grayscale(t), hash_size=8).__str__()
        db[h] = {"ch": ch, "w": t.width, "h": t.height}
        # 也存一份方便可视化
        t.save(os.path.join(TEMPLATE_IMG_DIR, f"{ch}_{h}.png"))
    save_json(db_path, db)
    return db

def hamming(a_hex: str, b_hex: str) -> int:
    return bin(int(a_hex,16) ^ int(b_hex,16)).count("1")

def map_idx_to_char_by_templates(tiles, db, ham_thr=6):
    """
    对当前雪碧图的小格，用模板库做匹配，得到 idx→char。
    先用 pHash 最近邻（汉明距离<=阈值），否则再用相关系数打分。
    """
    idx2char = {}
    # 预备模板数组
    tpl_imgs = []
    tpl_chars = []
    for h, meta in db.items():
        path = None
        # 尝试从磁盘读模板小图；若不存在，仅凭 phash 匹配
        for f in os.listdir(TEMPLATE_IMG_DIR):
            if f.endswith(f"{h}.png"):
                path = os.path.join(TEMPLATE_IMG_DIR, f)
                break
        if path and os.path.exists(path):
            tpl_imgs.append(Image.open(path).convert("L"))
            tpl_chars.append(meta["ch"])
        else:
            tpl_imgs.append(None)
            tpl_chars.append(meta["ch"])
    tpl_hashes = list(db.keys())

    def ncc(a: Image.Image, b: Image.Image) -> float:
        # 归一化互相关（尺寸必须一致）
        A = np.asarray(ImageOps.grayscale(a), dtype=np.float32)
        B = np.asarray(ImageOps.grayscale(b.resize(a.size, Image.NEAREST)), dtype=np.float32)
        A = (A - A.mean()) / (A.std() + 1e-6)
        B = (B - B.mean()) / (B.std() + 1e-6)
        return float((A*B).mean())

    for i, t in enumerate(tiles):
        h = imagehash.phash(ImageOps.grayscale(t), hash_size=8).__str__()
        # 第一轮：pHash 最近邻
        best_j, best_d = None, 999
        for j, th in enumerate(tpl_hashes):
            d = hamming(h, th)
            if d < best_d:
                best_d, best_j = d, j
        if best_d <= ham_thr:
            idx2char[i] = tpl_chars[best_j]
            continue
        # 第二轮：NCC 与所有有图模板比一遍
        best_score, best_char = -2.0, None
        for j, tpl in enumerate(tpl_imgs):
            if tpl is None:  # 没有模板图，跳过
                continue
            score = ncc(t, tpl)
            if score > best_score:
                best_score, best_char = score, tpl_chars[j]
        if best_char is not None:
            idx2char[i] = best_char
        else:
            idx2char[i] = "?"  # 仍然未知，稍后可手动补

    return idx2char

def decode_row_price(tr, cell_w, idx2char):
    out = []
    for sp in tr.select("span.encrypt-num"):
        st = sp.get("style","")
        m = re.search(r'(-?\d+)px', st)
        if not m: out.append("?"); continue
        x = int(m.group(1))
        idx = (-x) // cell_w
        out.append(idx2char.get(idx, "?"))
    return "".join(out)

def decode_price_in(box, cell_w, idx2char):
    out = []
    for sp in box.select('span.encrypt-num'):
        m = re.search(r'(-?\d+)px', sp.get('style',''))
        if not m:
            out.append('?'); continue
        x = int(m.group(1))
        idx = (-x) // cell_w
        out.append(idx2char.get(idx, '?'))
    return ''.join(out)

if __name__ == "__main__":
    CID_LIST = ["711", "710", "709", "708"]
    try:
        for index,cid in enumerate(CID_LIST):

            LIST_URL = f"{BASE}/category/1/{cid}"
            CAT_URL = f"{BASE}/category/1/{cid}"
            with requests.Session() as s:
                s.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
                    'Accept-Language': 'ja,en;q=0.9,zh-CN;q=0.8',
                })
                # 1) 先进首页拿会话 + CSRF
                r0 = s.get('https://www.kaitorishouten-co.jp/')
                token = extract_csrf_token(r0.text)
                # （可选）再进分类页，部分站点 token 会在分类页覆盖/刷新
                r1 = s.get(LIST_URL)
                token2 = extract_csrf_token(r1.text) or token
                # print('final token =', token2)
                # 2) 带 XHR + Referer + CSRF 访问接口（GET，返回 HTML 片段）
                headers = {
                    'X-Requested-With': 'XMLHttpRequest',
                    'Referer': LIST_URL,
                    "Accept": "image/webp,image/png,*/*;q=0.8",
                }
                if token2:
                    headers['eccube-csrf-token'] = token2
                try:
                    r2 = s.get(LIST_URL, headers=headers)
                    # print('status:', r2.status_code, r2.headers.get('Content-Type'))
                except requests.exceptions.RequestException as e:
                    print(e)

                r2.raise_for_status()
                r2.headers.get("Content-Type", "").startswith("image/")
                soup = BeautifulSoup(r2.content, "html.parser")

                # 2) 找雪碧图 + 单格尺寸
                sprite_url, cell_w, cell_h = extract_sprite_info(soup)
                if not sprite_url:
                    print("未找到 .encrypt-num 的 background-image；检查页面。")
                    # return
                # print("Sprite URL:", sprite_url, "| cell:", cell_w, "x", cell_h)

                # 3) 下载雪碧图
                sprite_img, sprite_bytes, ct = download_sprite(s, CAT_URL, sprite_url)
                sprite_md5 = md5(sprite_bytes)
                # print("Sprite MD5:", sprite_md5, "|", ct)

                # 4) 切出模板格（默认单行 11~12 格）
                tiles = cut_row_tiles(sprite_img, cell_w, cell_h)
                # 只取前 16 格（通常 11 格就够）
                tiles = tiles[:16]

                # 5) 如果这张雪碧图之前标注过，直接用；否则：用模板库自动匹配 → 仍缺就手动标注
                all_maps = load_json(CACHE_SPRITE2IDXCHAR, {})
                idx2char = all_maps.get(sprite_md5)

                if not idx2char:
                    # 尝试用历史模板自动匹配
                    tpl_db = load_json(TEMPLATE_DB, {})
                    if tpl_db:
                        auto_map = map_idx_to_char_by_templates(tiles, tpl_db, ham_thr=6)
                        # print("自动匹配得到（可能不全）：", auto_map)
                        if len([v for v in auto_map.values() if v != "?"]) >= 10:
                            idx2char = {str(i): ch for i, ch in auto_map.items()}
                    # 若仍然不全，走一次人工标注（只此一次）
                    if not idx2char or "?" in idx2char.values():
                        labels = show_and_label_once(tiles, cell_w, cell_h)  # {'0':'2','1':'4',...}
                        # 把标注写入全局模板库
                        build_or_update_template_db(tiles, labels, db_path=TEMPLATE_DB)
                        idx2char = labels

                    # 记到“按雪碧图MD5”的缓存
                    all_maps[sprite_md5] = idx2char
                    save_json(CACHE_SPRITE2IDXCHAR, all_maps)
                # print("本次 idx→char 映射：", idx2char)

                # 6) 解码整页价格并打印
                # print("\n=== 解码结果 ===")

                rows = []
                for tr in soup.select("tr.price_list_item"):
                    first_box = tr.select_one('div.item-price.encrypt-price')
                    if not first_box:
                        continue
                    price = decode_price_in(first_box, cell_w, {int(k): v for k, v in idx2char.items()})
                    price = price.replace(',', '')

                    jan = None
                    spans = tr.select('span.product-code-default')
                    for i, sp in enumerate(spans):
                        if sp.get_text(strip=True).upper().startswith('JAN'):
                            if i + 1 < len(spans):
                                jan = spans[i + 1].get_text(strip=True)
                            break
                    if not jan:
                        m = re.search(r'JAN:\s*([0-9]{8,14})', tr.get_text(' ', strip=True))
                        if m:
                            jan = m.group(1)

                    if jan and price:
                        tokyo = pytz.timezone('Asia/Tokyo')
                        ts = datetime.datetime.now(tokyo).isoformat(
                            timespec='seconds')  # e.g. 2025-10-08T18:22:31+09:00
                        rows.append({'JAN': jan, 'price': int(price), 'time-scraped': ts})
                if index==0:
                    with open('shop1_1.json', 'w', encoding='utf-8') as f:
                        json.dump(rows, f, ensure_ascii=False, indent=2)
                else:
                    with open('shop1_1.json', 'a', encoding='utf-8') as f:
                        json.dump(rows, f, ensure_ascii=False, indent=2)
                print(f"Wrote {len(rows)} rows to shop1.json")

        with open('shop1_1.json', 'r', encoding='utf-8') as f:
            text = f.read().strip()
        arrays = re.findall(r'\[[^\[\]]*\]', text)
        merged = []
        for arr in arrays:
            merged.extend(json.loads(arr))
        with open('shop1.json', 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)


    except Exception as e:
        print("ERROR:", e)
        raise
