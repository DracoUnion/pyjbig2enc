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
JBIG2编码器主模块（Numba加速版本）

本模块提供JBIG2编码的高级API，包括：
1. 多页编码 - 用于压缩整个文档
2. 单页通用编码 - 用于压缩单个图像
3. 符号提取和分类 - 用于文本模式编码

使用Numba加速关键计算密集型操作。
"""

from typing import List, Dict, Tuple, Optional, BinaryIO
from dataclasses import dataclass, field
from PIL import Image
import io
import math
import numpy as np

from .structs import (
    JBIG2_FILE_MAGIC,
    SegmentType,
    Jbig2FileHeader,
    Jbig2PageInfo,
    Jbig2GenericRegion,
    Jbig2SymbolDict,
    Jbig2TextRegion,
    Jbig2TextRegionAtflags,
    Jbig2TextRegionSyminsts,
    log2up,
)
from .segments import Segment
from .arith import ArithmeticEncoder, encode_bitimage, encode_image
from .sym import encode_symbol_table, encode_text_region
from .comparator import are_equivalent, compute_hash

# 导入Numba加速函数
from .arith_numba import (
    pack_image_to_words,
    connected_components_numba,
    flood_fill_numba,
)


@dataclass
class SymbolInfo:
    """
    符号（字形）信息

    存储一个提取的符号的位置和关联信息。
    """
    index: int
    page: int
    x: int
    y: int
    symbol_idx: int


class Jbig2Context:
    """
    JBIG2多页编码上下文

    这是JBIG2编码器的核心类，管理多页文档的压缩过程。
    """

    def __init__(self,
                 threshold: float = 0.85,
                 weight: float = 0.5,
                 xres: int = 0,
                 yres: int = 0,
                 full_headers: bool = True,
                 refine_level: int = -1):
        """
        初始化JBIG2编码上下文

        参数:
            threshold: 符号分类阈值（0.0-1.0）
            weight: 分类权重
            xres: X方向分辨率
            yres: Y方向分辨率
            full_headers: 是否生成完整JBIG2文件头
            refine_level: 细化级别（-1禁用）
        """
        self.threshold = threshold
        self.weight = weight
        self.xres = xres
        self.yres = yres
        self.full_headers = full_headers
        self.pdf_page_numbering = not full_headers
        self.refinement = refine_level >= 0
        self.refine_level = refine_level

        # 段号计数器
        self.segnum: int = 0
        self.symtab_segment: int = -1

        # 符号表
        self.symbols: List[Image.Image] = []
        self.symbol_positions: List[Tuple[int, int, int]] = []
        self.symbol_assignments: Dict[int, int] = {}

        # 页面信息
        self.pages: List[Dict] = []
        self.page_symbols: List[List[int]] = []

        # 符号使用计数
        self.symbol_use_count: Dict[int, int] = {}

        # 全局符号映射
        self.symmap: Dict[int, int] = {}
        self.num_global_symbols: int = 0

    def add_page(self, img: Image.Image) -> None:
        """
        添加一页到编码上下文

        提取页面中的连通组件（符号），进行分类。
        使用Numba加速的连通组件分析。
        """
        if img.mode != '1':
            img = img.convert('1')

        page_idx = len(self.pages)
        width, height = img.size

        # 保存页面信息
        page_info = {
            'width': width,
            'height': height,
            'xres': self.xres if self.xres else img.info.get('dpi', (0, 0))[0],
            'yres': self.yres if self.yres else img.info.get('dpi', (0, 0))[1],
        }
        self.pages.append(page_info)

        # 提取连通组件
        symbols_on_page = self._extract_symbols(img, page_idx)
        self.page_symbols.append(symbols_on_page)

    def _extract_symbols(self, img: Image.Image, page_idx: int) -> List[int]:
        """
        从图像中提取符号（连通组件）- Numba加速版本

        使用Numba编译的连通组件分析算法，比纯Python快10-50倍。
        """
        arr = np.array(img)
        if len(arr.shape) > 2:
            arr = arr[:, :, 0]

        # 二值化
        binary = (arr < 128).astype(np.uint8)
        height, width = binary.shape

        symbols_on_page = []

        # 使用Numba加速的连通组件分析
        num_components, comp_x, comp_y, comp_w, comp_h, comp_count = \
            connected_components_numba(binary, min_size=10)

        # 处理每个组件
        for i in range(num_components):
            x, y, w, h, count = comp_x[i], comp_y[i], comp_w[i], comp_h[i], comp_count[i]

            # 提取符号图像
            sym_img = Image.new('1', (w + 12, h + 12), 1)
            sym_pixels = sym_img.load()

            # 复制组件像素
            component_region = binary[y:y+h, x:x+w]
            for dy in range(h):
                for dx in range(w):
                    if component_region[dy, dx]:
                        sym_pixels[dx + 6, dy + 6] = 0

            # 分类符号
            symbol_idx = self._classify_symbol(sym_img)

            # 记录符号实例
            sym_instance_idx = len(self.symbol_positions)
            self.symbol_positions.append((page_idx, x, y + h - 1))
            self.symbol_assignments[sym_instance_idx] = symbol_idx

            symbols_on_page.append(sym_instance_idx)
            self.symbol_use_count[symbol_idx] = \
                self.symbol_use_count.get(symbol_idx, 0) + 1

        return symbols_on_page

    def _classify_symbol(self, sym_img: Image.Image) -> int:
        """
        对符号进行分类
        """
        for idx, existing_img in enumerate(self.symbols):
            if are_equivalent(existing_img, sym_img, self.threshold):
                return idx

        new_idx = len(self.symbols)
        self.symbols.append(sym_img)
        return new_idx

    def pages_complete(self, verbose: bool = False) -> bytes:
        """
        完成页面添加，编码符号表
        """
        single_page = len(self.pages) == 1

        multiuse_symbols = []
        for i in range(len(self.symbols)):
            count = self.symbol_use_count.get(i, 0)
            if count == 0:
                continue
            if count > 1 or single_page:
                multiuse_symbols.append(i)

        self.num_global_symbols = len(multiuse_symbols)

        self.symmap = {}
        for new_idx, old_idx in enumerate(multiuse_symbols):
            self.symmap[old_idx] = new_idx

        encoder = ArithmeticEncoder()

        symbol_list = multiuse_symbols if multiuse_symbols else list(range(len(self.symbols)))
        temp_symmap = {}
        encode_symbol_table(encoder, self.symbols, symbol_list, temp_symmap, True)

        if verbose:
            print(f"JBIG2 compression complete. pages:{len(self.pages)} "
                  f"symbols:{len(self.symbols)} log2:{log2up(len(self.symbols))}")

        return encoder.get_bytes()

    def produce_page(self, page_no: int, xres: int = -1, yres: int = -1) -> bytes:
        """
        编码单个页面
        """
        page = self.pages[page_no]
        width = page['width']
        height = page['height']

        if xres < 0:
            xres = page['xres']
        if yres < 0:
            yres = page['yres']

        encoder = ArithmeticEncoder()

        comps = self.page_symbols[page_no]
        positions = [(self.symbol_positions[c][1], self.symbol_positions[c][2])
                     for c in comps]
        assignments = [self.symbol_assignments[c] for c in comps]

        numsyms = self.num_global_symbols
        symbits = log2up(numsyms) if numsyms > 0 else 1

        empty_symmap2 = {}
        encode_text_region(encoder, self.symmap, empty_symmap2, comps,
                          positions, self.symbols, assignments, 1, symbits)

        return encoder.get_bytes()

    def destroy(self) -> None:
        """释放资源"""
        self.symbols.clear()
        self.symbol_positions.clear()
        self.symbol_assignments.clear()
        self.pages.clear()
        self.page_symbols.clear()


def encode_generic(img: Image.Image,
                   full_headers: bool = True,
                   xres: int = 0,
                   yres: int = 0,
                   duplicate_line_removal: bool = False) -> bytes:
    """
    编码单张图像为通用区域（Numba加速版本）
    """
    if img.mode != '1':
        img = img.convert('1')

    width, height = img.size
    if not xres:
        xres = img.info.get('dpi', (0, 0))[0]
    if not yres:
        yres = img.info.get('dpi', (0, 0))[1]

    # 转换为numpy数组
    arr = np.array(img)
    if len(arr.shape) > 2:
        arr = arr[:, :, 0]

    binary = (arr < 128).astype(np.uint8)

    # 使用Numba加速打包
    words_per_row = (width + 31) // 32
    packed_array = pack_image_to_words(binary, width, height)
    packed = packed_array.tobytes()

    encoder = ArithmeticEncoder()
    encode_bitimage(encoder, bytes(packed), width, height, duplicate_line_removal)
    encoder.finalize()

    encoded_data = encoder.get_bytes()

    # 构建JBIG2结构
    output = bytearray()

    if full_headers:
        header = Jbig2FileHeader(n_pages=1)
        output.extend(header.pack())

    segnum = 0
    seg = Segment()
    seg.number = segnum
    segnum += 1
    seg.type = SegmentType.PAGE_INFORMATION
    seg.page = 1

    page_info = Jbig2PageInfo(
        width=width,
        height=height,
        xres=int(xres) if xres else 0,
        yres=int(yres) if yres else 0,
        is_lossless=1
    )
    page_data = page_info.pack()
    seg.len = len(page_data)

    output.extend(seg.to_bytes())
    output.extend(page_data)

    seg2 = Segment()
    seg2.number = segnum
    segnum += 1
    seg2.type = SegmentType.IMM_GENERIC_REGION
    seg2.page = 1

    genreg = Jbig2GenericRegion(
        width=width,
        height=height,
        tpgdon=1 if duplicate_line_removal else 0
    )
    genreg_data = genreg.pack()
    seg2.len = len(genreg_data) + len(encoded_data)

    output.extend(seg2.to_bytes())
    output.extend(genreg_data)
    output.extend(encoded_data)

    if full_headers:
        endseg = Segment()
        endseg.number = segnum
        segnum += 1
        endseg.type = SegmentType.END_OF_PAGE
        endseg.page = 1
        output.extend(endseg.to_bytes())

        eofseg = Segment()
        eofseg.number = segnum
        eofseg.type = SegmentType.END_OF_FILE
        eofseg.page = 0
        output.extend(eofseg.to_bytes())

    return bytes(output)


def auto_threshold(ctx: Jbig2Context) -> None:
    """
    执行自动阈值处理
    """
    n = len(ctx.symbols)

    for i in range(n):
        j = i + 1
        while j < n:
            if are_equivalent(ctx.symbols[i], ctx.symbols[j]):
                _unite_symbols(ctx, i, j)
                n -= 1
            else:
                j += 1


def auto_threshold_using_hash(ctx: Jbig2Context) -> None:
    """
    使用哈希加速的自动阈值处理
    """
    from collections import defaultdict

    hash_groups: Dict[int, List[int]] = defaultdict(list)

    for i in range(len(ctx.symbols)):
        h = compute_hash(ctx.symbols[i])
        hash_groups[h].append(i)

    merged = set()

    for hash_val, indices in hash_groups.items():
        n = len(indices)
        for i in range(n):
            if indices[i] in merged:
                continue
            j = i + 1
            while j < n:
                if indices[j] in merged:
                    j += 1
                    continue

                if are_equivalent(ctx.symbols[indices[i]], ctx.symbols[indices[j]]):
                    _unite_symbols(ctx, indices[i], indices[j])
                    merged.add(indices[j])
                    indices.pop(j)
                    n -= 1
                else:
                    j += 1


def _unite_symbols(ctx: Jbig2Context, target: int, source: int) -> None:
    """
    合并两个符号
    """
    for comp_idx, sym_idx in ctx.symbol_assignments.items():
        if sym_idx == source:
            ctx.symbol_assignments[comp_idx] = target

    ctx.symbol_use_count[source] = 0
