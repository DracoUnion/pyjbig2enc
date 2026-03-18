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
JBIG2算术编码器模块

算术编码是一种熵编码方法，可以达到接近信息理论极限的压缩率。
JBIG2使用自适应算术编码，根据已编码的数据动态调整概率模型。

编码过程：
1. 维护一个区间[A, C)，初始为[0, 1)
2. 根据符号的概率将区间细分
3. 选择对应符号的子区间作为新的当前区间
4. 当区间足够小时，输出固定位数并扩展区间

本模块实现了JBIG2标准（ISO/IEC 14492）中指定的MQ编码器。
"""

from typing import List, Optional
import array

from numba import njit, uint8, uint32, int32
import numpy as np

# 最大上下文数 - JBIG2最多支持65536个不同的上下文
JBIG2_MAX_CTX = 65536

# 输出缓冲区大小 - 每个块的大小
JBIG2_OUTPUTBUFFER_SIZE = 20 * 1024


# 整数编码过程编号
# 用于标识不同类型的整数编码（高度、宽度、X偏移等）
class IntProc:
    """整数编码过程类型枚举"""
    IAAI = 0   # 聚集实例数
    IADH = 1   # 增量高度
    IADS = 2   # 增量S值
    IADT = 3   # 增量T值（条带位置）
    IADW = 4   # 增量宽度
    IAEX = 5   # 导出符号数
    IAFS = 6   # 第一个S值
    IAIT = 7   # 增量T（条带内）
    IARDH = 8  # 细化增量高度
    IARDW = 9  # 细化增量宽度
    IARDX = 10 # 细化X偏移
    IARDY = 11 # 细化Y偏移
    IARI = 12  # 细化标志


# MQ编码器状态表（来自JBIG2标准表E.1）
# 每个状态包含：
# - qe: 概率估计值（16位整数）
# - mps: 编码MPS（更可能符号）后的下一个状态
# - lps: 编码LPS（不太可能符号）后的下一个状态
# 状态46-91是0-45的"切换"版本（MPS翻转）
_MQ_STATES = [
    # 标准状态表（0-45）
    (0x5601, 1, 46),   # 0
    (0x3401, 2, 6),    # 1
    (0x1801, 3, 9),    # 2
    (0x0ac1, 4, 12),   # 3
    (0x0521, 5, 29),   # 4
    (0x0221, 38, 33),  # 5
    (0x5601, 7, 46+6), # 6
    (0x5401, 8, 14),   # 7
    (0x4801, 9, 14),   # 8
    (0x3801, 10, 14),  # 9
    (0x3001, 11, 17),  # 10
    (0x2401, 12, 18),  # 11
    (0x1c01, 13, 20),  # 12
    (0x1601, 29, 21),  # 13
    (0x5601, 15, 46+14), # 14
    (0x5401, 16, 14),  # 15
    (0x5101, 17, 15),  # 16
    (0x4801, 18, 16),  # 17
    (0x3801, 19, 17),  # 18
    (0x3401, 20, 18),  # 19
    (0x3001, 21, 19),  # 20
    (0x2801, 22, 19),  # 21
    (0x2401, 23, 20),  # 22
    (0x2201, 24, 21),  # 23
    (0x1c01, 25, 22),  # 24
    (0x1801, 26, 23),  # 25
    (0x1601, 27, 24),  # 26
    (0x1401, 28, 25),  # 27
    (0x1201, 29, 26),  # 28
    (0x1101, 30, 27),  # 29
    (0x0ac1, 31, 28),  # 30
    (0x09c1, 32, 29),  # 31
    (0x08a1, 33, 30),  # 32
    (0x0521, 34, 31),  # 33
    (0x0441, 35, 32),  # 34
    (0x02a1, 36, 33),  # 35
    (0x0221, 37, 34),  # 36
    (0x0141, 38, 35),  # 37
    (0x0111, 39, 36),  # 38
    (0x0085, 40, 37),  # 39
    (0x0049, 41, 38),  # 40
    (0x0025, 42, 39),  # 41
    (0x0015, 43, 40),  # 42
    (0x0009, 44, 41),  # 43
    (0x0005, 45, 42),  # 44
    (0x0001, 45, 43),  # 45
    # 切换状态（46-91）- 状态0-45的LPS切换版本
    (0x5601, 47, 1),   # 46
    (0x3401, 48, 6),   # 47
    (0x1801, 49, 9),   # 48
    (0x0ac1, 50, 12),  # 49
    (0x0521, 51, 29),  # 50
    (0x0221, 52, 33),  # 51
    (0x5601, 53, 6),   # 52
    (0x5401, 54, 14),  # 53
    (0x4801, 55, 14),  # 54
    (0x3801, 56, 14),  # 55
    (0x3001, 57, 17),  # 56
    (0x2401, 58, 18),  # 57
    (0x1c01, 59, 20),  # 58
    (0x1601, 60, 21),  # 59
    (0x5601, 61, 14),  # 60
    (0x5401, 62, 14),  # 61
    (0x5101, 63, 15),  # 62
    (0x4801, 64, 16),  # 63
    (0x3801, 65, 17),  # 64
    (0x3401, 66, 18),  # 65
    (0x3001, 67, 19),  # 66
    (0x2801, 68, 19),  # 67
    (0x2401, 69, 20),  # 68
    (0x2201, 70, 21),  # 69
    (0x1c01, 71, 22),  # 70
    (0x1801, 72, 23),  # 71
    (0x1601, 73, 24),  # 72
    (0x1401, 74, 25),  # 73
    (0x1201, 75, 26),  # 74
    (0x1101, 76, 27),  # 75
    (0x0ac1, 77, 28),  # 76
    (0x09c1, 78, 29),  # 77
    (0x08a1, 79, 30),  # 78
    (0x0521, 80, 31),  # 79
    (0x0441, 81, 32),  # 80
    (0x02a1, 82, 33),  # 81
    (0x0221, 83, 34),  # 82
    (0x0141, 84, 35),  # 83
    (0x0111, 85, 36),  # 84
    (0x0085, 86, 37),  # 85
    (0x0049, 87, 38),  # 86
    (0x0025, 88, 39),  # 87
    (0x0015, 89, 40),  # 88
    (0x0009, 90, 41),  # 89
    (0x0005, 91, 42),  # 90
    (0x0001, 91, 43),  # 91
]


class ArithmeticEncoder:
    """
    JBIG2 MQ算术编码器

    MQ编码器是JBIG2使用的自适应二进制算术编码器。
    它根据上下文（context）维护不同的概率模型。

    主要状态变量：
    - a: 区间大小寄存器（16位，0x8000表示满区间）
    - c: 代码寄存器（保存已编码的数据）
    - ct: 计数器（跟踪需要输出的字节数）
    - b: 当前字节缓冲区
    - bp: 字节位置计数器

    属性:
        context: 上下文状态数组，每个上下文有自己的概率状态
        intctx: 整数编码上下文，用于编码各种整数
        iaidctx: IAID编码上下文，用于编码符号ID
    """

    def __init__(self):
        """初始化编码器状态"""
        # 算术编码器核心状态
        self.a: int = 0x8000      # 区间大小，初始为满区间
        self.c: int = 0           # 代码寄存器
        self.ct: int = 12         # 计数器
        self.b: int = 0           # 当前字节
        self.bp: int = -1         # 字节位置（-1表示尚未输出）

        # 输出缓冲区
        self.output_chunks: List[bytearray] = []
        self.outbuf: bytearray = bytearray(JBIG2_OUTPUTBUFFER_SIZE)
        self.outbuf_used: int = 0

        # 上下文状态数组 - 每个条目是一个状态索引（0-91）
        # 图像编码使用的上下文
        self.context: array.array = array.array('B', [0] * JBIG2_MAX_CTX)
        # 整数编码使用的上下文 [13种过程][512个状态]
        self.intctx: array.array = array.array('B', [0] * (13 * 512))
        # IAID编码上下文（动态分配）
        self.iaidctx: Optional[array.array] = None

    def reset(self) -> None:
        """
        重置编码器状态

        用于开始一个新的编码段，但不释放缓冲区。
        保留已编码的数据，重置算术编码状态。
        """
        self.a = 0x8000
        self.c = 0
        self.ct = 12
        self.bp = -1
        self.b = 0
        self.iaidctx = None
        # 重置所有上下文状态为0
        for i in range(len(self.context)):
            self.context[i] = 0
        for i in range(len(self.intctx)):
            self.intctx[i] = 0

    def flush(self) -> None:
        """
        清空当前输出缓冲区

        重置输出位置，但保留已分配的缓冲区空间。
        用于准备新的编码操作。
        """
        self.outbuf_used = 0
        self.bp = -1

    def dealloc(self) -> None:
        """
        释放所有资源

        删除所有输出块和缓冲区。
        编码器不再可用，需要重新创建。
        """
        self.output_chunks.clear()
        self.outbuf = bytearray()
        self.iaidctx = None

    def _emit(self) -> None:
        """
        输出一个字节到缓冲区

        如果当前缓冲区已满，创建新缓冲区。
        这是内部方法，由_byteout调用。
        """
        if self.outbuf_used == JBIG2_OUTPUTBUFFER_SIZE:
            # 缓冲区已满，保存当前块并创建新块
            self.output_chunks.append(self.outbuf)
            self.outbuf = bytearray(JBIG2_OUTPUTBUFFER_SIZE)
            self.outbuf_used = 0

        self.outbuf[self.outbuf_used] = self.b & 0xFF
        self.outbuf_used += 1

    def _byteout(self) -> None:
        """
        执行BYTEOUT过程（MQ编码器核心）

        处理进位传播和字节输出。
        这是MQ编码器的关键部分，处理：
        1. 进位检测和传播
        2. 0xFF字节的特殊处理（防止与标记混淆）
        3. 字节提取和计数器重置
        """
        # 如果当前字节是0xFF，需要特殊处理
        if self.b == 0xFF:
            # 0xFF后面不能直接跟特定值，需要延迟输出
            self._emit()
            self.b = (self.c >> 20) & 0xFF
            self.c &= 0xFFFFF
            self.ct = 7
            self.bp += 1
            return

        # 检查是否有进位
        if self.c < 0x8000000:
            # 无进位 - 左块
            if self.bp >= 0:
                self._emit()
            self.b = (self.c >> 19) & 0xFF
            self.c &= 0x7FFFF
            self.ct = 8
            self.bp += 1
        else:
            # 有进位 - 需要传播
            self.b += 1
            if self.b == 0xFF:
                # 进位导致0xFF，清除高位
                self.c &= 0x7FFFFFF
                if self.bp >= 0:
                    self._emit()
                self.b = (self.c >> 20) & 0xFF
                self.c &= 0xFFFFF
                self.ct = 7
            else:
                if self.bp >= 0:
                    self._emit()
                self.b = (self.c >> 19) & 0xFF
                self.c &= 0x7FFFF
                self.ct = 8
            self.bp += 1

    def _renorme(self) -> None:
        """
        执行重归一化过程（RENORME）

        当区间大小A < 0x8000时，需要左移A和C，
        直到A >= 0x8000。每次左移可能触发字节输出。
        """
        while self.a < 0x8000:
            self.a <<= 1
            self.c <<= 1
            self.ct -= 1
            if self.ct == 0:
                self._byteout()

    def encode_bit(self, ctx: int, bit: int) -> None:
        """
        编码一个二进制位

        这是MQ编码器的核心编码函数。
        根据上下文状态决定如何编码位，并更新状态。

        参数:
            ctx: 上下文索引（0到JBIG2_MAX_CTX-1）
            bit: 要编码的位（0或1）

        过程:
        1. 获取当前上下文状态
        2. 根据MPS/LPS决定如何更新区间
        3. 可能需要重归一化
        4. 更新上下文状态
        """
        state_idx = self.context[ctx]
        qe, mps_next, lps_next = _MQ_STATES[state_idx]

        # MPS（更可能符号）由状态索引决定
        # 状态0-45: MPS=0, 状态46-91: MPS=1
        mps = 1 if state_idx > 45 else 0

        if bit == mps:
            # 编码MPS
            self.a -= qe
            if self.a >= 0x8000:
                # 无需重归一化
                self.c += qe
            else:
                # 需要重归一化
                if self.a < qe:
                    # 区间反转
                    self.a = qe
                else:
                    self.c += qe
                self.context[ctx] = mps_next
                self._renorme()
        else:
            # 编码LPS
            self.a -= qe
            if self.a < qe:
                self.c += qe
            else:
                self.a = qe
            self.context[ctx] = lps_next
            self._renorme()

    def finalize(self) -> None:
        """
        完成编码（FINALIZE过程）

        在编码结束时调用，确保所有数据都被输出。
        设置固定位，输出剩余字节，并添加终止标记。
        """
        # SETBITS - 设置固定位
        tempc = self.c + self.a
        self.c |= 0xFFFF
        if self.c >= tempc:
            self.c -= 0x8000

        # 输出最终字节
        self.c <<= self.ct
        self._byteout()
        self.c <<= self.ct
        self._byteout()
        self._emit()

        # 如果不是0xFF，再输出0xFF 0xAC终止序列
        if self.b != 0xFF:
            self.b = 0xFF
            self._emit()
        self.b = 0xAC
        self._emit()

    def datasize(self) -> int:
        """
        获取已编码数据的总字节数

        返回:
            int: 所有输出块和当前缓冲区的总字节数
        """
        return JBIG2_OUTPUTBUFFER_SIZE * len(self.output_chunks) + self.outbuf_used

    def to_buffer(self, buffer: bytearray) -> None:
        """
        将编码数据复制到缓冲区

        参数:
            buffer: 目标字节数组，必须足够大（至少datasize()字节）
        """
        pos = 0
        # 复制已完成的块
        for chunk in self.output_chunks:
            buffer[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
        # 复制当前缓冲区
        buffer[pos:pos + self.outbuf_used] = self.outbuf[:self.outbuf_used]

    def get_bytes(self) -> bytes:
        """
        获取编码数据作为字节串

        返回:
            bytes: 所有编码数据的副本
        """
        result = bytearray(self.datasize())
        self.to_buffer(result)
        return bytes(result)


# TPGD上下文索引 - 用于典型预测
TPGDCTX = 0x9B25


@njit(cache=True)
def _encode_image_numba(image_data: uint8[:], mx: int32, my: int32,
                        duplicate_line_removal: bool) -> uint32[:, :]:
    """
    Numba加速的图像编码核心循环

    返回形状为 (N, 2) 的数组，每行包含 (context, bit)
    """
    # 估算最大输出数
    max_output = mx * my + my  # 每行可能有一个TPGD位
    outputs = np.zeros((max_output, 2), dtype=np.uint32)
    out_idx = 0

    ltp = 0  # 典型预测状态

    for y in range(my):
        # 初始化上下文
        c1 = ((_get_pixel_numba(image_data, 0, y - 2, mx, my) << 2) |
              (_get_pixel_numba(image_data, 1, y - 2, mx, my) << 1) |
              _get_pixel_numba(image_data, 2, y - 2, mx, my))
        c2 = ((_get_pixel_numba(image_data, 0, y - 1, mx, my) << 3) |
              (_get_pixel_numba(image_data, 1, y - 1, mx, my) << 2) |
              (_get_pixel_numba(image_data, 2, y - 1, mx, my) << 1) |
              _get_pixel_numba(image_data, 3, y - 1, mx, my))
        c3 = 0

        # 检查是否与前一行相同（典型预测）
        if y > 0 and duplicate_line_removal:
            same = True
            for x in range(mx):
                if (_get_pixel_numba(image_data, x, y, mx, my) !=
                        _get_pixel_numba(image_data, x, y - 1, mx, my)):
                    same = False
                    break
            sltp = ltp ^ 1
            ltp = 1 if same else 0
            outputs[out_idx, 0] = TPGDCTX
            outputs[out_idx, 1] = sltp
            out_idx += 1
            if ltp:
                continue  # 跳过整行

        for x in range(mx):
            # 计算13位上下文索引
            tval = (c1 << 11) | (c2 << 4) | c3
            v = _get_pixel_numba(image_data, x, y, mx, my)

            outputs[out_idx, 0] = tval
            outputs[out_idx, 1] = v
            out_idx += 1

            # 更新上下文（移位并加入新值）
            c1 = ((c1 << 1) | _get_pixel_numba(image_data, x + 3, y - 2, mx, my)) & 0x1F
            c2 = ((c2 << 1) | _get_pixel_numba(image_data, x + 4, y - 1, mx, my)) & 0x7F
            c3 = ((c3 << 1) | v) & 0x0F

    return outputs[:out_idx]


def encode_image(encoder: ArithmeticEncoder, data: bytes, mx: int, my: int,
                 duplicate_line_removal: bool = False) -> None:
    """
    编码二值图像

    使用JBIG2的通用区域编码算法压缩二值图像。
    使用模板0（13位上下文）进行预测。

    参数:
        encoder: 算术编码器实例
        data: 图像数据，每行mx字节（ unpacked，每像素1位）
        mx: 图像宽度（像素）
        my: 图像高度（像素）
        duplicate_line_removal: 是否启用TPGD（典型预测）

    编码过程：
    1. 对于每个像素，收集上下文（周围像素的值）
    2. 使用上下文索引选择概率模型
    3. 编码当前像素值
    4. 更新上下文
    """
    # 转换为numpy数组以使用Numba
    image_array = np.frombuffer(data, dtype=np.uint8)

    # 使用Numba加速的核心循环
    outputs = _encode_image_numba(image_array, int32(mx), int32(my), duplicate_line_removal)

    # 编码结果
    for i in range(len(outputs)):
        ctx = outputs[i, 0]
        bit = outputs[i, 1]
        encoder.encode_bit(int(ctx), int(bit))


@njit(cache=True)
def _encode_bitimage_numba(data: uint8[:], mx: int32, my: int32,
                           duplicate_line_removal: bool) -> uint32[:, :]:
    """
    Numba加速的打包图像编码核心循环

    返回形状为 (N, 2) 的数组，每行包含 (context, bit)
    """
    words_per_row = (mx + 31) // 32
    max_output = mx * my + my
    outputs = np.zeros((max_output, 2), dtype=np.uint32)
    out_idx = 0

    ltp = 0

    for y in range(my):
        # 读取当前行和前两行的数据
        w1 = w2 = w3 = uint32(0)
        if y >= 2:
            row_start = (y - 2) * words_per_row * 4
            w1 = uint32((data[row_start + 0]) | (data[row_start + 1] << 8) |
                       (data[row_start + 2] << 16) | (data[row_start + 3] << 24))
        if y >= 1:
            row_start = (y - 1) * words_per_row * 4
            w2 = uint32((data[row_start + 0]) | (data[row_start + 1] << 8) |
                       (data[row_start + 2] << 16) | (data[row_start + 3] << 24))

            # 典型预测检查
            if duplicate_line_removal:
                curr_row_start = y * words_per_row * 4
                prev_row_start = (y - 1) * words_per_row * 4
                same = True
                for i in range(words_per_row * 4):
                    if data[curr_row_start + i] != data[prev_row_start + i]:
                        same = False
                        break
                sltp = ltp ^ 1
                ltp = 1 if same else 0
                outputs[out_idx, 0] = TPGDCTX
                outputs[out_idx, 1] = sltp
                out_idx += 1
                if ltp:
                    continue

        row_start = y * words_per_row * 4
        w3 = uint32((data[row_start + 0]) | (data[row_start + 1] << 8) |
                   (data[row_start + 2] << 16) | (data[row_start + 3] << 24))

        # 初始化上下文
        c1 = (w1 >> 29) & 0x1F
        c2 = (w2 >> 28) & 0x7F
        c3 = 0
        w1 <<= 3
        w2 <<= 4

        for x in range(mx):
            tval = (c1 << 11) | (c2 << 4) | c3
            v = (w3 >> 31) & 1
            outputs[out_idx, 0] = tval
            outputs[out_idx, 1] = v
            out_idx += 1

            c1 = ((c1 << 1) | ((w1 >> 31) & 1)) & 0x1F
            c2 = ((c2 << 1) | ((w2 >> 31) & 1)) & 0x7F
            c3 = ((c3 << 1) | v) & 0x0F
            w1 <<= 1
            w2 <<= 1
            w3 <<= 1

            # 每32像素重新加载字
            m = x % 32
            if m == 28 and y >= 2:
                wordno = (x // 32) + 1
                if wordno < words_per_row:
                    row_start = (y - 2) * words_per_row * 4 + wordno * 4
                    w1 = uint32((data[row_start + 0]) | (data[row_start + 1] << 8) |
                               (data[row_start + 2] << 16) | (data[row_start + 3] << 24))
            if m == 27 and y >= 1:
                wordno = (x // 32) + 1
                if wordno < words_per_row:
                    row_start = (y - 1) * words_per_row * 4 + wordno * 4
                    w2 = uint32((data[row_start + 0]) | (data[row_start + 1] << 8) |
                               (data[row_start + 2] << 16) | (data[row_start + 3] << 24))
            if m == 31:
                wordno = (x // 32) + 1
                if wordno < words_per_row:
                    row_start = y * words_per_row * 4 + wordno * 4
                    w3 = uint32((data[row_start + 0]) | (data[row_start + 1] << 8) |
                               (data[row_start + 2] << 16) | (data[row_start + 3] << 24))

    return outputs[:out_idx]


def encode_bitimage(encoder: ArithmeticEncoder, data: bytes, mx: int, my: int,
                    duplicate_line_removal: bool = False) -> None:
    """
    编码打包的二值图像（Leptonica格式）

    与encode_image类似，但输入数据是打包格式：
    每行以32位字（4字节）为单位存储，每个字包含32个像素。
    这是Leptonica库使用的标准格式。

    参数:
        encoder: 算术编码器实例
        data: 打包的图像数据（每行4字节对齐）
        mx: 图像宽度（像素）
        my: 图像高度（像素）
        duplicate_line_removal: 是否启用TPGD
    """
    # 转换为numpy数组以使用Numba
    image_array = np.frombuffer(data, dtype=np.uint8)

    # 使用Numba加速的核心循环
    outputs = _encode_bitimage_numba(image_array, int32(mx), int32(my), duplicate_line_removal)

    # 编码结果
    for i in range(len(outputs)):
        ctx = outputs[i, 0]
        bit = outputs[i, 1]
        encoder.encode_bit(int(ctx), int(bit))


@njit(cache=True)
def _get_pixel_numba(data: uint8[:], x: int32, y: int32, mx: int32, my: int32) -> int32:
    """
    Numba加速的像素获取（边界安全）

    如果坐标超出边界，返回0（边界外视为白色）。
    """
    if y < 0 or y >= my or x < 0 or x >= mx:
        return 0
    return int32(data[y * mx + x] & 1)


def get_pixel(data: bytes, x: int, y: int, mx: int, my: int) -> int:
    """
    安全地获取图像像素值

    如果坐标超出边界，返回0（边界外视为白色）。
    这是JBIG2标准要求的边界处理方式。

    参数:
        data: 图像数据
        x: X坐标
        y: Y坐标
        mx: 图像宽度
        my: 图像高度

    返回:
        int: 像素值（0或1）
    """
    if y < 0 or y >= my or x < 0 or x >= mx:
        return 0
    return data[y * mx + x] & 1


# 整数编码范围表
# 定义了不同数值范围的编码方式
# 每个条目包含：(最小值, 最大值, 前缀数据, 前缀位数, 增量, 整数位数)
_INTENC_RANGES = [
    (0, 3, 0, 2, 0, 2),           # 0-3: 2位前缀，2位整数
    (-1, -1, 9, 4, 0, 0),         # OOB标记
    (-3, -2, 5, 3, 2, 1),         # -3到-2
    (4, 19, 2, 3, 4, 4),          # 4-19
    (-19, -4, 3, 3, 4, 4),        # -19到-4
    (20, 83, 6, 4, 20, 6),        # 20-83
    (-83, -20, 7, 4, 20, 6),      # -83到-20
    (84, 339, 14, 5, 84, 8),      # 84-339
    (-339, -84, 15, 5, 84, 8),    # -339到-84
    (340, 4435, 30, 6, 340, 12),  # 340-4435
    (-4435, -340, 31, 6, 340, 12), # -4435到-340
    (4436, 2000000000, 62, 6, 4436, 32),  # 大正数
    (-2000000000, -4436, 63, 6, 4436, 32), # 大负数
]


def encode_int(encoder: ArithmeticEncoder, proc: int, value: int) -> None:
    """
    编码一个整数（IAxI过程）

    JBIG2使用多种不同的整数编码过程（IAEX, IADH等），
    每种过程有自己的概率模型。

    编码过程：
    1. 根据数值范围选择编码方式
    2. 编码前缀位标识范围
    3. 编码剩余的整数值

    参数:
        encoder: 算术编码器
        proc: 过程编号（IntProc中的常量）
        value: 要编码的整数值
    """
    context = encoder.intctx
    base_idx = proc * 512

    # 查找适用的编码范围
    range_idx = 0
    for i, (bot, top, _, _, _, _) in enumerate(_INTENC_RANGES):
        if bot <= value <= top:
            range_idx = i
            break

    r = _INTENC_RANGES[range_idx]
    _, _, prefix_data, prefix_bits, delta, int_bits = r

    # 处理负值
    if value < 0:
        value = -value
    value -= delta

    # 编码前缀位
    prev = 1
    data = prefix_data
    for _ in range(prefix_bits):
        v = data & 1
        encoder.encode_bit(base_idx + prev, v)
        data >>= 1
        # 更新上下文索引
        if prev & 0x100:
            prev = (((prev << 1) | v) & 0x1FF) | 0x100
        else:
            prev = (prev << 1) | v

    # 编码整数值
    if int_bits:
        value <<= (32 - int_bits)
        for _ in range(int_bits):
            v = (value >> 31) & 1
            encoder.encode_bit(base_idx + prev, v)
            value <<= 1
            if prev & 0x100:
                prev = (((prev << 1) | v) & 0x1FF) | 0x100
            else:
                prev = (prev << 1) | v


def encode_oob(encoder: ArithmeticEncoder, proc: int) -> None:
    """
    编码OOB（越界）标记

    OOB用于标记列表的结束（如符号列表、高度类等）。
    编码为特定的位模式（1, 0, 0, 0）。

    参数:
        encoder: 算术编码器
        proc: 过程编号
    """
    base_idx = proc * 512
    encoder.encode_bit(base_idx + 1, 1)
    encoder.encode_bit(base_idx + 3, 0)
    encoder.encode_bit(base_idx + 6, 0)
    encoder.encode_bit(base_idx + 12, 0)


def encode_iaid(encoder: ArithmeticEncoder, symcodelen: int, value: int) -> None:
    """
    编码IAID（符号ID）

    IAID使用独立的上下文数组，每个位位置有自己的上下文。
    符号ID的长度（位数）取决于符号表的大小。

    参数:
        encoder: 算术编码器
        symcodelen: 符号ID的位数（log2(符号数)）
        value: 符号ID值
    """
    # 延迟分配IAID上下文
    if encoder.iaidctx is None:
        encoder.iaidctx = array.array('B', [0] * (1 << symcodelen))

    mask = (1 << (symcodelen + 1)) - 1
    if symcodelen:
        value <<= (32 - symcodelen)

    prev = 1
    for _ in range(symcodelen):
        tval = prev & mask
        v = (value >> 31) & 1
        encoder.encode_bit(tval, v)
        prev = (prev << 1) | v
        value <<= 1
