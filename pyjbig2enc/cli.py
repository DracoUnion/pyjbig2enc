# Copyright (c) 2026 wizardforcel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
JBIG2命令行工具

这是jbig2enc的Python重写版本的命令行界面。
提供与原始C++版本相似的命令行接口。

用法：
    python -m jbig2enc [options] <input files...>

主要功能：
1. 单页通用编码 - 直接压缩图像
2. 多页符号编码 - 提取符号并压缩多页文档
3. PDF输出模式 - 生成适合PDF嵌入的数据

示例：
    # 单页压缩
    python -m jbig2enc image.png > output.jb2

    # 多页符号编码
    python -m jbig2enc -s page1.png page2.png -b output

    # PDF模式
    python -m jbig2enc -s -p -a *.png -b output
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional, Callable
from io import BytesIO
from PIL import Image
from argparse import Namespace

from . import __version__
from .encoder import Jbig2Context, encode_generic_single, auto_threshold, auto_threshold_using_hash
from .structs import JBIG2_FILE_MAGIC
from .pdf import make_jb2_pdf
from .adathres import adathres


# 默认参数值
JBIG2_THRESHOLD_MIN = 0.4
JBIG2_THRESHOLD_MAX = 0.97
JBIG2_THRESHOLD_DEF = 0.92
JBIG2_WEIGHT_MIN = 0.1
JBIG2_WEIGHT_MAX = 0.9
JBIG2_WEIGHT_DEF = 0.5
BW_THRESHOLD_MIN = 0
BW_THRESHOLD_MAX = 255
BW_LOCAL_THRESHOLD_DEF = 200
BW_GLOBAL_THRESHOLD_DEF = 128


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器

    定义所有支持的命令行选项和参数。
    """
    parser = argparse.ArgumentParser(
        prog='jbig2enc',
        description='JBIG2 Encoder - Python rewrite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Encode single image
  %(prog)s image.png > output.jb2

  # Multi-page symbol mode
  %(prog)s -s page1.png page2.png -b output

  # PDF mode with auto-threshold
  %(prog)s -s -p -a *.png -b output

  # Generic encoding with duplicate line removal
  %(prog)s -d image.png > output.jb2
''')

    # 版本信息
    parser.add_argument('-V', '--version', action='version',
                        version=f'%(prog)s {__version__}')

    # 输出选项
    parser.add_argument('-b', '--basename', default='output',
                        help='Output file root name for symbol mode (default: output)')
    parser.add_argument('-p', '--pdf', action='store_true',
                        help='Produce PDF-ready data (no file headers)')
    parser.add_argument('-O', '--output-image', metavar='FILE',
                        help='Dump thresholded image as PNG')

    # 编码模式选项
    parser.add_argument('-s', '--symbol-mode', action='store_true',
                        help='Use text region coding (symbol mode) instead of generic')
    parser.add_argument('-d', '--duplicate-line-removal', action='store_true',
                        help='Use TPGD in generic region coder')
    parser.add_argument('-r', '--refine', action='store_true',
                        help='Use refinement (requires -s, lossless)')

    # 符号分类参数
    parser.add_argument('-t', '--threshold', type=float, default=JBIG2_THRESHOLD_DEF,
                        help=f'Symbol classification threshold (default: {JBIG2_THRESHOLD_DEF})')
    parser.add_argument('-w', '--weight', type=float, default=JBIG2_WEIGHT_DEF,
                        help=f'Symbol classification weight (default: {JBIG2_WEIGHT_DEF})')

    # 图像处理选项
    parser.add_argument('-T', '--bw-threshold', type=int, default=BW_LOCAL_THRESHOLD_DEF,
                        help=f'B&W threshold for 8bpp images (default: {BW_LOCAL_THRESHOLD_DEF})')
    parser.add_argument('-G', '--global', dest='globalmode', action='store_true',
                        help='Use global B&W threshold instead of adaptive')
    parser.add_argument('-2', '--upscale-2x', action='store_true',
                        help='Upsample 2x before thresholding')
    parser.add_argument('-4', '--upscale-4x', action='store_true',
                        help='Upsample 4x before thresholding')

    # 自动阈值选项
    parser.add_argument('-a', '--auto-thresh', action='store_true',
                        help='Use automatic thresholding in symbol encoder')
    parser.add_argument('--no-hash', action='store_true',
                        help='Disable hash function for auto-thresholding')

    # 分割选项
    parser.add_argument('-S', '--segment', action='store_true',
                        help='Segment image into text and graphics regions')
    parser.add_argument('-j', '--jpeg-output', action='store_true',
                        help='Write graphics as JPEG (default: PNG)')

    # 其他选项
    parser.add_argument('-D', '--dpi', type=int, default=0,
                        help='Force DPI resolution')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Be verbose')

    # 输入文件
    parser.add_argument('files', nargs='+', help='Input image files')

    return parser


def load_image(filepath: str | bytes, args) -> Optional[Image.Image]:
    """
    加载并预处理图像

    根据命令行参数处理输入图像：
    1. 加载图像
    2. 转换颜色空间
    3. 上采样（如果请求）
    4. 二值化

    参数:
        filepath: 图像文件路径
        args: 命令行参数

    返回:
        Image.Image: 处理后的二值图像，失败返回None
    """
    try:
        if isinstance(filepath, bytes):
            filepath = BytesIO(filepath)
        img = Image.open(filepath)
    except Exception as e:
        print(f"Error: Cannot open '{filepath}': {e}", file=sys.stderr)
        return None

    if args.verbose:
        print(f"Processing '{filepath}': {img.size} ({img.mode})", file=sys.stderr)

    # 设置DPI（如果指定）
    if args.dpi > 0:
        img.info['dpi'] = (args.dpi, args.dpi)

    # 转换颜色空间
    if img.mode == 'P':
        # 调色板图像，转换为RGB
        img = img.convert('RGB')

    if img.mode in ('RGB', 'RGBA', 'L'):
        # 灰度或彩色图像，需要二值化
        if img.mode == 'RGBA':
            # 处理透明通道
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        # 转换为灰度
        if img.mode != 'L':
            img = img.convert('L')

        # 上采样（如果请求）
        if args.upscale_2x:
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        elif args.upscale_4x:
            img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)

        # 二值化
        if args.globalmode:
            # 全局阈值
            img = img.point(lambda x: 0 if x < args.bw_threshold else 255, '1')
        else:
            bio = BytesIO()
            img.save(bio, 'PNG')
            img_data = adathres(bio.getvalue())
            img = Image.open(BytesIO(img_data))
            img = img.convert('1')

    elif img.mode != '1':
        # 其他模式，直接转换为二值
        img = img.convert('1')

    return img


def encode_generic_mode(files: List[str], args) -> bytes:
    """
    编码单页图像

    使用通用区域编码压缩单张图像。

    参数:
        img: 输入图像
        args: 命令行参数

    返回:
        bytes: 编码后的JBIG2数据
    """
    l = len(str(len(files)))
    for i, f in enumerate(files):
        img = load_image(f, args)
        if img is None: continue
        if args.output_image:
            img.save(args.output_image)
        data = encode_generic_single(
            f,
            full_headers=not args.pdf,
            duplicate_line_removal=args.duplicate_line_removal
        )
        # 输出到stdout
        output_fname = str(i).zfill(l)
        output_fname = f"{args.basename}_{output_fname}.jb2"
        open(output_fname, 'wb').write(data)
        if args.verbose:
            print(f"JB2 written to {output_fname}", file=sys.stderr)



def encode_symbol_mode(files: List[str], args) -> None:
    """
    编码多页文档

    使用符号模式压缩多个页面，提取并共享符号。

    参数:
        files: 输入文件列表
        args: 命令行参数
    """
    # 验证阈值参数
    if not (JBIG2_THRESHOLD_MIN <= args.threshold <= JBIG2_THRESHOLD_MAX):
        print(f"Error: Threshold must be between {JBIG2_THRESHOLD_MIN} and {JBIG2_THRESHOLD_MAX}",
              file=sys.stderr)
        sys.exit(1)

    if not (JBIG2_WEIGHT_MIN <= args.weight <= JBIG2_WEIGHT_MAX):
        print(f"Error: Weight must be between {JBIG2_WEIGHT_MIN} and {JBIG2_WEIGHT_MAX}",
              file=sys.stderr)
        sys.exit(1)

    # 细化级别
    refine_level = 10 if args.refine else -1

    # 创建编码上下文
    ctx = Jbig2Context(
        threshold=args.threshold,
        weight=args.weight,
        full_headers=not args.pdf,
        refine_level=refine_level
    )

    # 加载并添加所有页面
    for filepath in files:
        img = load_image(filepath, args)
        if img is None:
            continue

        # 输出阈值图像（如果请求）
        if args.output_image:
            img.save(args.output_image)
            args.output_image = None  # 只输出一次

        ctx.add_page(img)

    if len(ctx.pages) == 0:
        print("Error: No valid pages to encode", file=sys.stderr)
        sys.exit(1)

    # 自动阈值处理
    if args.auto_thresh:
        if args.verbose:
            print("Performing auto-thresholding...", file=sys.stderr)
        if args.no_hash:
            auto_threshold(ctx)
        else:
            auto_threshold_using_hash(ctx)

    # 编码符号表
    if args.verbose:
        print("Encoding symbol table...", file=sys.stderr)

    sym_data = ctx.pages_complete(verbose=args.verbose)

    # 输出符号表
    sym_file = f"{args.basename}.sym"
    with open(sym_file, 'wb') as f:
        f.write(sym_data)
    if args.verbose:
        print(f"Symbol table written to {sym_file}", file=sys.stderr)

    # 编码并输出每一页
    page_data_list = []
    for i in range(len(ctx.pages)):
        if args.verbose:
            print(f"Encoding page {i+1}/{len(ctx.pages)}...", file=sys.stderr)

        page_data = ctx.produce_page(i)
        page_data_list.append(page_data)

        page_file = f"{args.basename}.{i:04d}"
        with open(page_file, 'wb') as f:
            f.write(page_data)
        if args.verbose:
            print(f"Page {i} written to {page_file}", file=sys.stderr)

    if args.pdf:
        pdf_fname = f"{args.basename}.pdf"
        pdf_data = make_jb2_pdf(sym_data, page_data_list)
        open(pdf_fname, 'wb').write(pdf_data)
        if args.verbose:
            print(f"pdf written to {pdf_fname}", file=sys.stderr)

    # 清理
    ctx.destroy()


def img2jb2pdf(imgs: str[bytes | str | Callable]) -> bytes:
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            imgs[i] = open(img, 'rb').read()
        elif isinstance(img, bytes):
            pass
        elif callable(getattr(img, 'read', None)):
            imgs[i] = img.read()
        else:
            raise ValueError('param must be str, bytes or file')
    # 创建编码上下文
    ctx = Jbig2Context(
        threshold=JBIG2_THRESHOLD_DEF,
        weight=JBIG2_WEIGHT_DEF,
        full_headers=False,
        refine_level=10,
    )
    # 加载并添加所有页面
    for img_data in imgs:
        img_data = adathres(img_data)
        img = Image.open(BytesIO(img_data))
        if img.mode != '1':
            img = img.convert('1')
        ctx.add_page(img)
    # 自动阈值处理
    auto_threshold_using_hash(ctx)
    # 符号表
    sym_data = ctx.pages_complete()
    # 编码并输出每一页
    page_data_list = [
        ctx.produce_page(i)
        for i in range(len(ctx.pages))
    ]
    # 合成 PDF
    pdf = make_jb2_pdf(sym_data, page_data_list)
    # 清理
    ctx.destroy()
    
    return pdf

def main() -> int:
    """
    主函数

    解析命令行参数并执行相应的编码操作。

    返回:
        int: 退出码（0表示成功）
    """
    parser = create_parser()
    args = parser.parse_args()

    # 验证输入文件存在
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            return 1

    # 验证互斥选项
    if args.upscale_2x and args.upscale_4x:
        print("Error: Cannot use both -2 and -4", file=sys.stderr)
        return 1

    if args.refine and not args.symbol_mode:
        print("Error: Refinement (-r) requires symbol mode (-s)", file=sys.stderr)
        return 1

    try:
        if args.symbol_mode:
            # 符号模式
            encode_symbol_mode(args.files, args)
        else:
            # 通用模式
            encode_generic_mode(args.files, args)

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
