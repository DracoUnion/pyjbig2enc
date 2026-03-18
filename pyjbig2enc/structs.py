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
JBIG2数据结构定义模块

本模块定义了JBIG2文件格式中使用的所有二进制数据结构。
JBIG2是一种用于二值图像（黑白图像）压缩的国际标准。

注意：JBIG2使用大端字节序（Big Endian）存储多字节数据
"""

import struct
from enum import IntEnum
from dataclasses import dataclass
from typing import List


# JBIG2文件魔数 - 每个JBIG2文件都以这8个字节开头
# 0x97是JBIG2特有的文件标识字节
JBIG2_FILE_MAGIC = b'\x97J B2\r\n\x1a\n'


class SegmentType(IntEnum):
    """
    JBIG2段类型枚举

    JBIG2文件由多个段（Segment）组成，每个段有特定的类型。
    这些类型定义了段的内容和用途。
    """
    SYMBOL_TABLE = 0           # 符号表 - 存储文档中使用的符号（字形）
    IMM_GENERIC_REGION = 38    # 立即通用区域 - 直接存储压缩的图像数据
    PAGE_INFORMATION = 48      # 页面信息 - 描述页面尺寸、分辨率等
    IMM_TEXT_REGION = 6        # 立即文本区域 - 存储文本符号的引用和位置
    END_OF_PAGE = 49           # 页面结束标记
    END_OF_FILE = 51           # 文件结束标记


@dataclass
class Jbig2FileHeader:
    """
    JBIG2文件头结构（13字节）

    这是JBIG2文件的开头部分，包含文件标识和基本信息。
    仅在完整的JBIG2文件中存在，PDF嵌入模式不需要。

    字段说明：
    - id: 8字节文件标识，固定为JBIG2_FILE_MAGIC
    - organisation_type: 文件组织类型（0=顺序，1=页式）
    - unknown_n_pages: 页面数是否未知（0=已知，1=未知）
    - reserved: 保留位
    - n_pages: 文件中的总页数（4字节大端）
    """
    id: bytes = JBIG2_FILE_MAGIC  # 8字节文件魔数
    organisation_type: int = 1     # 1=页式组织（随机访问）
    unknown_n_pages: int = 0       # 0=页面数已知
    reserved: int = 0              # 保留位，必须为0
    n_pages: int = 0               # 总页数

    def pack(self) -> bytes:
        """
        将文件头打包为二进制字节

        返回:
            bytes: 13字节的二进制数据，可直接写入文件
        """
        # 第一个字节包含 organisation_type(1位), unknown_n_pages(1位), reserved(6位)
        flags = (self.organisation_type & 0x01) | \
                ((self.unknown_n_pages & 0x01) << 1) | \
                ((self.reserved & 0x3F) << 2)

        return self.id + struct.pack('>B', flags) + struct.pack('>I', self.n_pages)


@dataclass
class Jbig2PageInfo:
    """
    JBIG2页面信息段数据（19字节）

    描述一个页面的基本属性，每个页面必须有一个页面信息段。

    字段说明：
    - width: 页面宽度（像素）
    - height: 页面高度（像素）
    - xres: X方向分辨率（每英寸点数）
    - yres: Y方向分辨率（每英寸点数）
    - is_lossless: 是否无损压缩（1=无损，0=有损）
    - contains_refinements: 是否包含细化编码
    - default_pixel: 默认像素值（0或1）
    - default_operator: 默认组合操作符
    - aux_buffers: 辅助缓冲区标志
    - operator_override: 操作符覆盖标志
    - reserved: 保留位
    - segment_flags: 段标志
    """
    width: int = 0
    height: int = 0
    xres: int = 0
    yres: int = 0
    is_lossless: int = 1
    contains_refinements: int = 0
    default_pixel: int = 0
    default_operator: int = 0
    aux_buffers: int = 0
    operator_override: int = 0
    reserved: int = 0
    segment_flags: int = 0

    def pack(self) -> bytes:
        """打包页面信息为19字节的二进制数据"""
        # 打包标志字节
        flags = (self.is_lossless & 0x01) | \
                ((self.contains_refinements & 0x01) << 1) | \
                ((self.default_pixel & 0x01) << 2) | \
                ((self.default_operator & 0x03) << 3) | \
                ((self.aux_buffers & 0x01) << 5) | \
                ((self.operator_override & 0x01) << 6) | \
                ((self.reserved & 0x01) << 7)

        return struct.pack('>IIII', self.width, self.height, self.xres, self.yres) + \
               struct.pack('BB', flags, self.segment_flags)


@dataclass
class Jbig2GenericRegion:
    """
    JBIG2通用区域段头（17字节 + 8字节自适应模板）

    通用区域编码用于直接压缩二值图像数据。
    使用自适应算术编码，支持模板匹配预测。

    字段说明：
    - width, height: 区域尺寸
    - x, y: 区域在页面中的位置
    - comb_operator: 组合操作符（用于多层合成）
    - mmr: 是否使用MMR编码（0=算术编码，1=MMR）
    - gbtemplate: 通用区域模板（0-3）
    - tpgdon: 是否启用TPGD（典型预测）
    - a1x, a1y, ...: 自适应模板像素偏移
    """
    width: int = 0
    height: int = 0
    x: int = 0
    y: int = 0
    comb_operator: int = 0
    mmr: int = 0          # 0 = 算术编码, 1 = MMR
    gbtemplate: int = 0   # 通用区域模板编号 (0-3)
    tpgdon: int = 0       # 典型预测标志
    reserved: int = 0
    # 自适应模板像素偏移 - 定义预测上下文的位置
    a1x: int = 3
    a1y: int = -1
    a2x: int = -3
    a2y: int = -1
    a3x: int = 2
    a3y: int = -2
    a4x: int = -2
    a4y: int = -2

    def pack(self) -> bytes:
        """打包通用区域头为二进制数据"""
        # 标志字节: mmr(1位), gbtemplate(2位), tpgdon(1位), reserved(4位)
        flags = (self.mmr & 0x01) | \
                ((self.gbtemplate & 0x03) << 1) | \
                ((self.tpgdon & 0x01) << 3) | \
                ((self.reserved & 0x0F) << 4)

        result = struct.pack('>IIIIB', self.width, self.height, self.x, self.y,
                            self.comb_operator)
        result += struct.pack('B', flags)
        # 自适应模板偏移
        result += struct.pack('bbbbbbbb', self.a1x, self.a1y, self.a2x, self.a2y,
                              self.a3x, self.a3y, self.a4x, self.a4y)
        return result


@dataclass
class Jbig2SymbolDict:
    """
    JBIG2符号字典段头（22字节 + 自适应模板）

    符号字典存储文档中重复使用的符号（如字符字形）。
    文本区域通过引用符号字典中的条目来表示文本内容。

    编码标志（sdhuff等）控制使用算术编码还是霍夫曼编码。
    """
    # 编码控制标志
    sdhuff: int = 0           # 使用霍夫曼编码
    sdrefagg: int = 0         # 引用聚集标志
    sdhuffdh: int = 0         # 增量高度霍夫曼编码
    sdhuffdw: int = 0         # 增量宽度霍夫曼编码
    sdhuffbmsize: int = 0     # 位图大小霍夫曼编码
    sdhuffagginst: int = 0    # 聚集实例霍夫曼编码
    bmcontext: int = 0        # 位图上下文
    bmcontextretained: int = 0  # 保留上下文
    sdtemplate: int = 0       # 符号字典模板 (0-3)
    sdrtemplate: int = 0      # 细化模板
    reserved: int = 0

    # 自适应模板偏移
    a1x: int = 3
    a1y: int = -1
    a2x: int = -3
    a2y: int = -1
    a3x: int = 2
    a3y: int = -2
    a4x: int = -2
    a4y: int = -2

    # 符号数量
    exsyms: int = 0    # 导出符号数
    newsyms: int = 0   # 新符号数

    def pack(self) -> bytes:
        """打包符号字典头为二进制数据"""
        # 第一个标志字节
        flags1 = (self.sdhuff & 0x01) | \
                 ((self.sdrefagg & 0x01) << 1) | \
                 ((self.sdhuffdh & 0x03) << 2) | \
                 ((self.sdhuffdw & 0x03) << 4) | \
                 ((self.sdhuffbmsize & 0x01) << 6) | \
                 ((self.sdhuffagginst & 0x01) << 7)

        # 第二个标志字节
        flags2 = (self.bmcontext & 0x01) | \
                 ((self.bmcontextretained & 0x01) << 1) | \
                 ((self.sdtemplate & 0x03) << 2) | \
                 ((self.sdrtemplate & 0x01) << 4) | \
                 ((self.reserved & 0x07) << 5)

        result = struct.pack('BB', flags1, flags2)
        result += struct.pack('bbbbbbbb', self.a1x, self.a1y, self.a2x, self.a2y,
                              self.a3x, self.a3y, self.a4x, self.a4y)
        result += struct.pack('>II', self.exsyms, self.newsyms)
        return result


@dataclass
class Jbig2TextRegion:
    """
    JBIG2文本区域段头（16字节）

    文本区域通过引用符号字典中的符号来表示文本内容。
    每个符号实例指定了符号ID和位置信息。

    字段说明：
    - width, height: 文本区域尺寸
    - x, y: 区域位置
    - comb_operator: 组合操作符
    - sbcombop2, sbdefpixel: 子区域组合操作和默认像素
    - sbdsoffset: 深度偏移
    - sbrtemplate: 细化模板
    - sbhuff: 使用霍夫曼编码
    - sbrefine: 启用细化
    - logsbstrips: 条带宽度对数（0=1px, 1=2px, 2=4px, 3=8px）
    - refcorner: 参考角（0=左下, 1=右下, 2=左上, 3=右上）
    - transposed: 是否转置
    - sbcombop1: 子区域组合操作1
    """
    width: int = 0
    height: int = 0
    x: int = 0
    y: int = 0
    comb_operator: int = 0

    sbcombop2: int = 0
    sbdefpixel: int = 0
    sbdsoffset: int = 0
    sbrtemplate: int = 0
    sbhuff: int = 0
    sbrefine: int = 0
    logsbstrips: int = 0
    refcorner: int = 0
    transposed: int = 0
    sbcombop1: int = 0

    def pack(self) -> bytes:
        """打包文本区域头为二进制数据"""
        # 打包两个标志字节
        flags = (self.sbcombop2 & 0x01) | \
                ((self.sbdefpixel & 0x01) << 1) | \
                ((self.sbdsoffset & 0x1F) << 2)

        flags2 = (self.sbrtemplate & 0x01) | \
                 ((self.sbhuff & 0x01) << 1) | \
                 ((self.sbrefine & 0x01) << 2) | \
                 ((self.logsbstrips & 0x03) << 3) | \
                 ((self.refcorner & 0x03) << 5) | \
                 ((self.transposed & 0x01) << 7)

        flags3 = self.sbcombop1 & 0x01

        return struct.pack('>IIIIB', self.width, self.height, self.x, self.y,
                          self.comb_operator) + \
               struct.pack('BBB', flags, flags2, flags3)


@dataclass
class Jbig2TextRegionAtflags:
    """
    文本区域自适应模板标志（4字节）

    定义细化编码时使用的自适应模板像素偏移。
    """
    a1x: int = -1
    a1y: int = -1
    a2x: int = -1
    a2y: int = -1

    def pack(self) -> bytes:
        """打包为4字节"""
        return struct.pack('bbbb', self.a1x, self.a1y, self.a2x, self.a2y)


@dataclass
class Jbig2TextRegionSyminsts:
    """
    文本区域符号实例信息（4字节）

    存储该文本区域中的符号实例数量。
    """
    sbnuminstances: int = 0

    def pack(self) -> bytes:
        """打包为4字节大端整数"""
        return struct.pack('>I', self.sbnuminstances)


def log2up(v: int) -> int:
    """
    计算编码v个符号所需的最小位数（向上取整的对数）

    例如：
    - log2up(1) = 0  （需要0位）
    - log2up(2) = 1  （需要1位：0, 1）
    - log2up(3) = 2  （需要2位：00, 01, 10）
    - log2up(4) = 2  （需要2位：00, 01, 10, 11）
    - log2up(5) = 3  （需要3位）

    参数:
        v: 要编码的符号数量

    返回:
        int: 所需的最小二进制位数
    """
    if v <= 0:
        return 0
    # 检查是否为2的幂
    is_pow_of_2 = (v & (v - 1)) == 0

    r = 0
    v -= 1  # 因为2^n需要n位，但2^n-1也只需要n位
    while v > 0:
        v >>= 1
        r += 1

    return r
