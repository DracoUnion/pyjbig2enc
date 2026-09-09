#!/usr/bin/env python3
"""
JBIG2单页编码示例

展示如何使用Python API压缩单张图像。

用法:
    python encode_single.py input.png -o output.jb2

要求:
    pip install Pillow numpy numba
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from pyjbig2enc import encode_generic_single


def main():
    parser = argparse.ArgumentParser(description='Encode single image with JBIG2')
    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', default='output.jb2',
                        help='Output file (default: output.jb2)')
    parser.add_argument('-d', '--duplicate-line-removal', action='store_true',
                        help='Enable TPGD (faster encoding)')
    parser.add_argument('--no-header', action='store_true',
                        help='Output raw data without JBIG2 file header')

    args = parser.parse_args()

    print(f"JBIG2单页编码 (Numba加速已启用)")
    print()

    # 加载图像
    print(f"加载: {args.input}")
    try:
        img = Image.open(args.input)
    except Exception as e:
        print(f"错误: 无法加载图像: {e}")
        return 1

    # 显示原始信息
    original_size = img.size
    print(f"原始尺寸: {original_size[0]}x{original_size[1]} ({img.mode})")

    # 转换为二值
    if img.mode != '1':
        img = img.convert('L').point(lambda x: 0 if x < 128 else 255, '1')

    # 编码
    print("编码中...")
    try:
        data = encode_generic_single(
            img,
            full_headers=not args.no_header,
            duplicate_line_removal=args.duplicate_line_removal
        )
    except Exception as e:
        print(f"错误: 编码失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 保存
    with open(args.output, 'wb') as f:
        f.write(data)

    # 显示结果
    input_bytes = Path(args.input).stat().st_size
    output_bytes = len(data)
    ratio = input_bytes / output_bytes if output_bytes > 0 else 0

    print(f"\n结果:")
    print(f"  输入大小: {input_bytes:,} 字节")
    print(f"  输出大小: {output_bytes:,} 字节")
    print(f"  压缩比: {ratio:.2f}:1")
    print(f"  节省空间: {(1 - 1/ratio)*100:.1f}%")
    print(f"  输出文件: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
