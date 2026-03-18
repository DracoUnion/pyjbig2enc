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
符号编码和文本区域编码模块

本模块处理JBIG2的符号字典编码和文本区域编码。

符号字典编码：
- 将符号（字形）按高度分类
- 对每个高度类，按宽度排序
- 使用增量编码压缩高度和宽度
- 使用算术编码压缩位图

文本区域编码：
- 将符号实例按条带（strip）组织
- 条带内按X坐标排序
- 编码符号ID和位置增量
"""

from typing import List, Dict, Tuple, Optional
from PIL import Image
import math

from .arith import (
    ArithmeticEncoder,
    encode_bitimage,
    encode_int,
    encode_oob,
    encode_iaid,
    IntProc,
)


# 符号边界大小 - Leptonica存储的符号有6像素的边界
BORDER_SIZE = 6


def encode_symbol_table(encoder: ArithmeticEncoder,
                        symbols: List[Image.Image],
                        symbol_list: List[int],
                        symmap: Dict[int, int],
                        unborder_symbols: bool = True) -> None:
    """
    编码符号字典（符号表）

    符号字典是JBIG2文本压缩的核心。它将文档中出现的所有唯一符号
    （如字符字形）存储一次，后续文本区域通过引用这些符号来表示内容。

    编码策略：
    1. 按高度对符号分类
    2. 对每个高度类，按宽度排序
    3. 使用增量编码（IADH, IADW）压缩尺寸信息
    4. 使用算术编码压缩符号位图

    参数:
        encoder: 算术编码器
        symbols: 所有符号的图像列表
        symbol_list: 要编码的符号索引列表
        symmap: 输出参数，映射原始符号索引到编码后的符号编号
        unborder_symbols: 是否去除符号周围的边界
    """
    n = len(symbol_list)
    number = 0

    # 创建符号索引列表并按高度排序
    syms = list(symbol_list)
    syms.sort(key=lambda i: symbols[i].height)

    # 按高度分组编码
    hcheight = 0  # 当前高度类的高度
    i = 0

    while i < n:
        # 获取当前高度类的高度
        first_sym_idx = syms[i]
        height = symbols[first_sym_idx].height
        if unborder_symbols:
            height -= 2 * BORDER_SIZE

        # 收集同一高度的所有符号
        hc = [syms[i]]  # 高度类的符号列表
        j = i + 1
        while j < n:
            sym_idx = syms[j]
            sym_height = symbols[sym_idx].height
            if unborder_symbols:
                sym_height -= 2 * BORDER_SIZE
            if sym_height != height:
                break
            hc.append(sym_idx)
            j += 1

        # 按宽度排序
        hc.sort(key=lambda idx: symbols[idx].width)

        # 编码增量高度
        deltaheight = height - hcheight
        encode_int(encoder, IntProc.IADH, deltaheight)
        hcheight = height

        # 编码这个高度类的所有符号
        symwidth = 0
        for sym_idx in hc:
            this_width = symbols[sym_idx].width
            if unborder_symbols:
                this_width -= 2 * BORDER_SIZE

            # 编码增量宽度
            deltawidth = this_width - symwidth
            encode_int(encoder, IntProc.IADW, deltawidth)
            symwidth += deltawidth

            # 获取符号图像（可能需要去边界）
            sym_img = symbols[sym_idx]
            if unborder_symbols:
                # 去除边界
                w, h = sym_img.size
                sym_img = sym_img.crop((
                    BORDER_SIZE,
                    BORDER_SIZE,
                    w - BORDER_SIZE,
                    h - BORDER_SIZE
                ))

            # 编码符号位图
            _encode_symbol_bitmap(encoder, sym_img, this_width, height)

            # 记录符号映射
            symmap[sym_idx] = number
            number += 1

        # OOB标记高度类结束
        encode_oob(encoder, IntProc.IADW)
        i = j

    # 编码导出符号信息
    # 第一个数字：跳过的符号数（0表示从第一个开始）
    encode_int(encoder, IntProc.IAEX, 0)
    # 第二个数字：导出的符号数
    encode_int(encoder, IntProc.IAEX, n)

    # 完成编码
    encoder.finalize()


def _encode_symbol_bitmap(encoder: ArithmeticEncoder,
                          img: Image.Image,
                          width: int,
                          height: int) -> None:
    """
    编码单个符号的位图

    将PIL图像转换为适合编码的格式，并调用算术编码器。

    参数:
        encoder: 算术编码器
        img: 符号图像（1位）
        width: 宽度
        height: 高度
    """
    # 转换为打包格式
    import numpy as np

    arr = np.array(img.convert('1'))
    if len(arr.shape) > 2:
        arr = arr[:, :, 0]

    # 确保每行32位对齐（Leptonica格式）
    words_per_row = (width + 31) // 32
    packed = bytearray(words_per_row * 4 * height)

    for y in range(height):
        for x in range(width):
            word_idx = x // 32
            bit_idx = 31 - (x % 32)  # 大端位序
            if arr[y, x]:
                byte_idx = y * words_per_row * 4 + word_idx * 4 + (3 - bit_idx // 8)
                bit_pos = bit_idx % 8
                packed[byte_idx] |= (1 << bit_pos)

    # 使用打包格式编码
    encode_bitimage(encoder, bytes(packed), width, height, False)


def encode_text_region(encoder: ArithmeticEncoder,
                       symmap: Dict[int, int],
                       symmap2: Dict[int, int],
                       comps: List[int],
                       positions: List[Tuple[int, int]],
                       symbols: List[Image.Image],
                       assignments: List[int],
                       stripwidth: int,
                       symbits: int) -> None:
    """
    编码文本区域

    文本区域编码文档中的一段文本，通过引用符号字典中的符号
    来表示内容，而不是直接编码像素。

    编码策略：
    1. 将符号实例按条带（strip）分组（条带高度=stripwidth）
    2. 每个条带内按X坐标排序
    3. 编码条带位置增量（IADT）
    4. 编码符号位置（IAFS, IADS）
    5. 编码符号ID（IAID）

    参数:
        encoder: 算术编码器
        symmap: 全局符号映射（符号索引->符号编号）
        symmap2: 每页符号映射（可选）
        comps: 组件索引列表（要编码的符号实例）
        positions: 每个组件的位置（x, y）列表
        symbols: 符号图像列表
        assignments: 每个组件对应的符号索引
        stripwidth: 条带宽度（1, 2, 4, 或 8）
        symbits: 符号ID的位数
    """
    # 验证条带宽度
    if stripwidth not in (1, 2, 4, 8):
        raise ValueError(f"Invalid stripwidth: {stripwidth}")

    n = len(comps)
    if n == 0:
        encoder.finalize()
        return

    # 准备符号实例列表
    syms = list(comps)
    # 按Y坐标（底部）排序
    syms.sort(key=lambda i: positions[i][1])

    # 初始化编码状态
    stript = 0  # 当前条带位置
    firsts = 0  # 第一条带的首符号X位置

    # 编码初始条带位置
    encode_int(encoder, IntProc.IADT, 0)

    # 处理每个条带
    i = 0
    while i < n:
        # 确定当前条带的高度范围
        height = (positions[syms[i]][1] // stripwidth) * stripwidth

        # 收集条带内的所有符号
        strip = [syms[i]]
        j = i + 1
        while j < n:
            y = positions[syms[j]][1]
            if y >= height + stripwidth:
                break
            strip.append(syms[j])
            j += 1

        # 按X坐标排序
        strip.sort(key=lambda idx: positions[idx][0])

        # 编码条带位置增量
        deltat = height - stript
        encode_int(encoder, IntProc.IADT, deltat // stripwidth)
        stript = height

        # 编码条带内的符号
        firstsymbol = True
        curs = 0

        for sym in strip:
            x, y = positions[sym]

            if firstsymbol:
                firstsymbol = False
                # 编码首个符号的X位置
                deltafs = x - firsts
                encode_int(encoder, IntProc.IAFS, deltafs)
                firsts += deltafs
                curs = firsts
            else:
                # 编码后续符号的X增量
                deltas = x - curs
                encode_int(encoder, IntProc.IADS, deltas)
                curs += deltas

            # 编码Y偏移（仅在stripwidth > 1时需要）
            if stripwidth > 1:
                deltat = y - stript
                encode_int(encoder, IntProc.IAIT, deltat)

            # 获取符号ID
            assigned = assignments[sym]

            # 查找符号编号
            if assigned in symmap:
                symid = symmap[assigned]
            elif assigned in symmap2:
                symid = symmap2[assigned] + len(symmap)
            else:
                raise ValueError(f"Symbol {assigned} not found in symbol maps")

            # 编码符号ID
            encode_iaid(encoder, symbits, symid)

            # 更新当前X位置（加上符号宽度）
            sym_width = symbols[assigned].width if assigned < len(symbols) else 10
            if unborder_symbols:
                sym_width -= 2 * BORDER_SIZE
            curs += sym_width - 1

        # OOB标记条带结束
        encode_oob(encoder, IntProc.IADS)
        i = j

    # 完成编码
    encoder.finalize()


# 去边界标志（模块级别）
unborder_symbols = True


def set_unborder_symbols(value: bool) -> None:
    """设置是否去除符号边界"""
    global unborder_symbols
    unborder_symbols = value
