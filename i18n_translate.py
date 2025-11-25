#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import pathlib
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

try:
    # 官方 Python SDK：from openai import OpenAI，默认使用环境变量 OPENAI_API_KEY
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 匹配中日韩字符：中日韩统一表意文字 + 平假名 + 片假名
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]+")

# 要扫描的文件后缀（按需增删）
INCLUDE_EXT = {".php", ".html", ".htm", ".js", ".ts", ".vue"}

# 不进入的目录（按你项目情况可以调整）
EXCLUDE_DIRS = {
    ".git",
    "vendor",
    "runtime",
    "storage",
    "node_modules",
    "dist",
    "build",
    "public/dist",
}


@dataclass
class Occurrence:
    file: str   # 相对项目根目录的路径
    line: int   # 行号（1 开始）
    context: str  # 当前整行文本，用于翻译参考


@dataclass
class Phrase:
    id: int
    text: str
    occurrences: List[Occurrence]


# ============== scan：扫描项目，生成 JSON（你已经跑过，可以跳过） ==============

def scan_project(root: pathlib.Path) -> List[Phrase]:
    """
    扫描项目，按“短语去重 + 记录所有出现位置”。
    """
    phrases: Dict[str, Phrase] = {}
    seen_occ = set()
    next_id = 1
    root = root.resolve()

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            ext = pathlib.Path(filename).suffix.lower()
            if ext not in INCLUDE_EXT:
                continue

            path = pathlib.Path(dirpath) / filename
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # 非 UTF-8 文件，直接跳过
                continue

            rel_path = path.relative_to(root)

            for lineno, line in enumerate(text.splitlines(), start=1):
                if not CJK_RE.search(line):
                    continue

                for m in CJK_RE.finditer(line):
                    phrase_str = m.group(0).strip()
                    if not phrase_str:
                        continue

                    key = (phrase_str, str(rel_path), lineno)
                    if key in seen_occ:
                        continue
                    seen_occ.add(key)

                    if phrase_str not in phrases:
                        phrases[phrase_str] = Phrase(
                            id=next_id,
                            text=phrase_str,
                            occurrences=[],
                        )
                        next_id += 1

                    phrases[phrase_str].occurrences.append(
                        Occurrence(
                            file=str(rel_path).replace("\\", "/"),
                            line=lineno,
                            context=line.rstrip("\n"),
                        )
                    )

    return list(phrases.values())


def cmd_scan(args):
    root = pathlib.Path(args.root)
    phrases = scan_project(root)

    items = []
    for p in sorted(phrases, key=lambda x: x.id):
        d = asdict(p)
        d["translated"] = None  # 预留翻译字段
        items.append(d)

    out_obj = {"items": items}
    out_path = pathlib.Path(args.out)
    out_path.write_text(
        json.dumps(out_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"扫描完成，发现 {len(items)} 个唯一短语，已写入 {out_path}")


# ============== translate：用 OpenAI 批量翻译 JSON 中的短语 ==============

def _load_items(path: pathlib.Path):
    """
    兼容两种结构：
    1) {"items": [...]}
    2) [ ... ]
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        raise SystemExit("JSON 格式不正确，必须是 {\"items\": [...]} 或纯数组")
    return items, data


def call_openai_batch(client, batch, src_lang, tgt_lang, model):
    """
    批量翻译一组短语。
    batch: list[dict]，来自 JSON 里的 items，每个至少包含 id, text, occurrences。
    返回: {id: translated_text}
    """
    import time
    import json

    payload = []
    for item in batch:
        occs = item.get("occurrences") or []
        context = ""
        file = ""

        if occs:
            context = occs[0].get("context", "") or ""
            file = occs[0].get("file", "") or ""

            # 🔥 关键：截断过长 context，避免某些一行几万字符把上下文撑爆
            max_ctx_len = 200  # 你可以按需调，比如 200/300
            if len(context) > max_ctx_len:
                phrase = item.get("text", "") or ""
                pos = context.find(phrase) if phrase else -1
                if pos != -1:
                    # 尝试保留短语附近的一小段上下文
                    start = max(0, pos - 60)
                    end = min(len(context), pos + len(phrase) + 60)
                    context = context[start:end]
                else:
                    # 找不到就简单截前 200 字符
                    context = context[:max_ctx_len]

        payload.append(
            {
                "id": item["id"],
                "text": item["text"],
                "context": context,
                "file": file,
            }
        )

    system_prompt = (
        "You are a professional software localization translator. "
        "Translate UI strings for a web application."
    )

    user_prompt = (
        f"请把下面 JSON 数组里的 text 字段从{src_lang}翻译成{tgt_lang}。\n"
        "要求：\n"
        "1. 严格保留变量名、占位符和格式，例如 {name}、{0}、%s、%d、%1$s 等，不能改动。\n"
        "2. 严格保留 HTML 标签和属性，只翻译标签内用户可见的文字。\n"
        "3. 参考 context 和 file 判断语境，使翻译适合网站/后台管理的 UI 文案。\n"
        "4. 不要添加任何解释或注释。\n"
        "5. 只返回一个 JSON 对象，结构严格为：\n"
        '   { \"items\": [ {\"id\": 1, \"translated\": \"...\"}, ... ] }\n\n'
        f"下面是待翻译数组：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    # 用 Responses API，并用 text.format 要求返回 JSON 对象
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_object"
            }
        },
    )

    raw = resp.output_text  # SDK 会把所有文本输出拼到这里

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"模型返回的不是合法 JSON：\n{raw}")

    items = obj.get("items", [])
    mapping: Dict[int, str] = {}
    for it in items:
        mapping[int(it["id"])] = it["translated"]

    # 简单限速，防一手 QPS 过高（你也可以关掉）
    time.sleep(0.3)
    return mapping



def cmd_translate(args):
    if OpenAI is None:
        raise SystemExit("请先 `pip install openai` 再运行 translate 命令。")

    import json
    import time
    from pathlib import Path

    client = OpenAI()  # 使用环境变量 OPENAI_API_KEY
    src_lang = args.src
    tgt_lang = args.tgt
    model = args.model
    batch_size = args.batch
    sleep_sec = getattr(args, "sleep", 0.2)

    input_path = Path(args.input)
    out_path = Path(args.out)

    # 兼容两种结构：{"items": [...]} 或纯数组 [...]
    raw = input_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict) and "items" in data:
        items = data["items"]
        container = data  # 用于后面写 meta
    elif isinstance(data, list):
        items = data
        container = None
    else:
        raise SystemExit("JSON 格式不正确，必须是 {\"items\": [...]} 或纯数组")

    # 只翻译还没有 translated 的条目（支持断点续跑）
    to_translate = [it for it in items if not it.get("translated")]
    total = len(to_translate)
    if total == 0:
        print("所有条目的 translated 字段都已存在，不需要翻译。")
        return

    print(f"共有 {total} 条需要翻译，将分批调用 OpenAI 模型 {model} ...")

    def save_progress():
        """每一批翻完写一次 JSON，防止中途挂掉丢进度。"""
        if container is not None:
            container["items"] = items
            meta = container.setdefault("meta", {})
            meta["src_lang"] = src_lang
            meta["tgt_lang"] = tgt_lang
            meta["model"] = model
            meta["translated_count"] = sum(1 for it in items if it.get("translated"))
            out_obj = container
        else:
            out_obj = items

        out_path.write_text(
            json.dumps(out_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [已保存进度到 {out_path}]")

    # 按 batch 分批调用
    for i in range(0, total, batch_size):
        batch = to_translate[i: i + batch_size]
        print(f"  -> 翻译第 {i + 1} ~ {i + len(batch)} 条...")

        mapping = call_openai_batch(client, batch, src_lang, tgt_lang, model)

        # 写回到 items（batch 里的 dict 本身就是 items 的引用）
        for item in batch:
            _id = int(item["id"])
            if _id in mapping:
                item["translated"] = mapping[_id]

        save_progress()
        time.sleep(sleep_sec)

    print("全部翻译完成。")


# ============== apply：按 file+line 精确回写到源码 ==============

def cmd_apply(args):
    root = pathlib.Path(args.root).resolve()
    mapping_path = pathlib.Path(args.mapping)

    items, _ = _load_items(mapping_path)

    # 构建：file -> line -> [(src, tgt), ...]
    file_line_map: Dict[str, Dict[int, List[Tuple[str, str]]]] = {}

    for row in items:
        text = row["text"]
        translated = row.get("translated")
        if not translated:
            continue

        occs = row.get("occurrences") or []
        for occ in occs:
            file = occ["file"]
            line = int(occ["line"])
            file_dict = file_line_map.setdefault(file, {})
            line_list = file_dict.setdefault(line, [])
            line_list.append((text, translated))

    if not file_line_map:
        print("没有任何 translated 字段，apply 不会做修改。")
        return

    for rel_file, line_map in file_line_map.items():
        path = (root / rel_file).resolve()
        if not path.exists():
            print(f"[警告] 文件不存在，跳过：{rel_file}")
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"[警告] 文件不是 UTF-8 编码，跳过：{rel_file}")
            continue

        lines = content.splitlines(keepends=True)
        changed = False

        for line_no, repl_list in line_map.items():
            idx = line_no - 1
            if idx < 0 or idx >= len(lines):
                print(f"[警告] 行号超出范围：{rel_file}:{line_no}")
                continue

            line_text = lines[idx]
            original_line = line_text

            # 同一行可能有多个短语，逐个替换
            for src, tgt in repl_list:
                line_text = line_text.replace(src, tgt)

            if line_text != original_line:
                lines[idx] = line_text
                changed = True

        if changed:
            new_content = "".join(lines)
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                backup.write_text(content, encoding="utf-8")
                print(f"已生成备份：{backup}")
            path.write_text(new_content, encoding="utf-8")
            print(f"已应用翻译：{rel_file}")
        else:
            print(f"未修改：{rel_file}")


# ============== CLI 入口 ==============

def main():
    parser = argparse.ArgumentParser(description="扫描/翻译/回写项目中的中日韩文本")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # scan（你已经跑过了，可以忽略）
    p_scan = sub.add_parser("scan", help="扫描项目，生成待翻译 JSON")
    p_scan.add_argument("--root", default=".", help="项目根目录")
    p_scan.add_argument("--out", default="translations.todo.json", help="输出 JSON 文件")
    p_scan.set_defaults(func=cmd_scan)

    # translate：用 OpenAI 批量翻译
    p_trans = sub.add_parser("translate", help="调用 OpenAI API 批量翻译 JSON 中的短语")
    p_trans.add_argument("--input", required=True, help="scan 生成的 JSON 文件路径")
    p_trans.add_argument(
        "--out",
        default="translations.done.json",
        help="翻译后输出 JSON 文件路径",
    )
    p_trans.add_argument(
        "--src",
        default="日文",
        help="源语言描述（用于 prompt，例如：日文 / 简体中文 / 繁体中文 / 英文）",
    )
    p_trans.add_argument(
        "--tgt",
        default="简体中文",
        help="目标语言描述（用于 prompt）",
    )
    p_trans.add_argument(
        "--model",
        default="gpt-5.1",
        help="模型名称（如 gpt-4.1-mini, gpt-4.1 等）",
    )
    p_trans.add_argument(
        "--batch",
        type=int,
        default=50,
        help="每次请求翻译多少条（可以根据 token 情况调大/调小）",
    )
    p_trans.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="两次请求之间 sleep 秒数，防止过快触发限速",
    )
    p_trans.set_defaults(func=cmd_translate)

    # apply：回写到源码
    p_apply = sub.add_parser("apply", help="根据 JSON mapping 回写翻译结果到源码文件")
    p_apply.add_argument(
        "--root",
        default=".",
        help="项目根目录（与 scan 时保持一致）",
    )
    p_apply.add_argument(
        "--mapping",
        required=True,
        help="translate 生成的 JSON 文件路径",
    )
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
