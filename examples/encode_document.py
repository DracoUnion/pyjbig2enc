#!/usr/bin/env python3
"""
JBIG2编码示例 - 文档压缩

展示如何使用Python API压缩多页文档。

用法:
    python encode_document.py page1.png page2.png ... -o output_prefix

要求:
    pip install Pillow numpy numba
"""

import sys
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from pyjbig2enc import Jbig2Context, auto_threshold_using_hash


def main():
    parser = argparse.ArgumentParser(description='Encode documents with JBIG2')
    parser.add_argument('files', nargs='+', help='Input image files')
    parser.add_argument('-o', '--output', default='output',
                        help='Output file prefix (default: output)')
    parser.add_argument('-a', '--auto-thresh', action='store_true',
                        help='Enable automatic thresholding')
    parser.add_argument('-t', '--threshold', type=float, default=0.92,
                        help='Classification threshold (default: 0.92)')
    parser.add_argument('-p', '--pdf-mode', action='store_true',
                        help='PDF mode (no file headers)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # 显示加速状态
    print(f"JBIG2 Encoder - Python版本 (Numba加速已启用)")
    print(f"输入文件: {len(args.files)} 页")
    print()

    # 创建编码上下文
    ctx = Jbig2Context(
        threshold=args.threshold,
        full_headers=not args.pdf_mode
    )

    # 加载并添加所有页面
    for i, filepath in enumerate(args.files):
        if args.verbose:
            print(f"处理页面 {i+1}/{len(args.files)}: {filepath}")

        try:
            img = Image.open(filepath)
            if img.mode != '1':
                img = img.convert('1')
            ctx.add_page(img)
        except Exception as e:
            print(f"错误: 无法加载 {filepath}: {e}")
            return 1

    print(f"提取到 {len(ctx.symbols)} 个唯一符号")

    # 自动阈值处理
    if args.auto_thresh:
        print("执行自动阈值处理...")
        auto_threshold_using_hash(ctx)
        print(f"合并后: {len([s for s in ctx.symbols if ctx.symbol_use_count.get(id(s), 0) > 0])} 个符号")

    # 编码符号表
    print("编码符号表...")
    sym_data = ctx.pages_complete(verbose=args.verbose)

    if args.pdf_mode:
        sym_file = f"{args.output}.sym"
        with open(sym_file, 'wb') as f:
            f.write(sym_data)
        print(f"符号表已保存: {sym_file} ({len(sym_data)} 字节)")
    else:
        with open(f"{args.output}.jb2", 'wb') as f:
            f.write(sym_data)

    # 编码每一页
    for i in range(len(ctx.pages)):
        if args.verbose:
            print(f"编码页面 {i+1}...")

        page_data = ctx.produce_page(i)

        if args.pdf_mode:
            page_file = f"{args.output}.{i:04d}"
        else:
            page_file = f"{args.output}_{i:04d}.jb2"

        with open(page_file, 'wb') as f:
            f.write(page_data)

        print(f"页面 {i+1} 已保存: {page_file} ({len(page_data)} 字节)")

    # 清理
    ctx.destroy()
    print("\n完成!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
