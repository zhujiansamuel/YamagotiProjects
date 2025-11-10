#!/usr/bin/env python3
import ast
import os
import sys
from pathlib import Path

try:
    # Python 3.8+
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore

def get_stdlib_names():
    # 优先使用 3.10+ 的 sys.stdlib_module_names
    names = set()
    if hasattr(sys, "stdlib_module_names") and sys.stdlib_module_names:
        names.update(sys.stdlib_module_names)
    # 常见内置别名补齐
    names.update({"__future__", "typing", "dataclasses"})
    return names

def parse_top_level_imports(py_path: Path):
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_path))
    imports = set()

    for node in ast.walk(tree):
        # 跳过相对导入: from . import x / from ..pkg import y
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                top = node.module.split(".")[0]
                imports.add(top)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                imports.add(top)
    return imports

def map_import_to_distribution(import_names):
    # 建立 import 名 -> 发行包名 的映射
    pkg_dist = importlib_metadata.packages_distributions()  # dict: import_name -> [dist, ...]
    mapping = {}
    unknown = set()

    # 一些常见别名的手动修正
    manual_map = {
        "bs4": "beautifulsoup4",
        "PIL": "pillow",
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "cv2": "opencv-python",
        "Crypto": "pycryptodome",
        "dateutil": "python-dateutil",
        "lxml": "lxml",
        "regex": "regex",
        "ujson": "ujson",
        "orjson": "orjson",
        "win32com": "pypiwin32",
    }

    for name in sorted(import_names):
        if name in manual_map:
            mapping[name] = manual_map[name]
            continue
        dists = pkg_dist.get(name)
        if dists:
            # 取第一个最可能的发行包名
            mapping[name] = dists[0]
        else:
            unknown.add(name)
    return mapping, unknown

def main():
    if len(sys.argv) < 2:
        print("用法: python gen_requirements_from_py.py your_script.py [output_requirements.txt]")
        sys.exit(1)

    py_file = Path(sys.argv[1]).resolve()
    if not py_file.exists():
        print(f"找不到文件: {py_file}")
        sys.exit(1)

    out_file = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else Path("requirements.txt").resolve()

    stdlib = get_stdlib_names()
    imports = parse_top_level_imports(py_file)

    # 过滤内置/标准库名与特殊前缀
    filtered = {
        n for n in imports
        if n and n not in stdlib and n not in {"__future__", "typing"} and not n.startswith("_")
    }

    mapping, unknown = map_import_to_distribution(filtered)

    lines = []
    # 固定版本号（基于当前环境已安装）
    for import_name in sorted(mapping):
        dist = mapping[import_name]
        try:
            ver = importlib_metadata.version(dist)
            lines.append(f"{dist}=={ver}")
        except importlib_metadata.PackageNotFoundError:
            # 当前环境没安装就不锁版本，只给包名
            lines.append(f"{dist}")

    # 把未知的包名放在文件末尾，供你手动确认
    if unknown:
        lines.append("")
        lines.append("# --- 未能自动映射到 PyPI 发行包，请确认后手动添加 ---")
        for n in sorted(unknown):
            lines.append(f"# unknown: {n}   （例如 bs4->beautifulsoup4, PIL->pillow）")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成: {out_file}")

if __name__ == "__main__":
    main()