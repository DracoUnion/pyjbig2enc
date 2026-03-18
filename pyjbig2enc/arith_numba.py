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
Numba加速的算术编码和图像处理函数

本模块提供使用Numba JIT编译优化的核心计算函数，
可以显著提高编码性能。
"""

import numpy as np
from numba import njit, uint32, uint16, uint8, int32, boolean

# MQ编码器状态表（编译为Numba可用的格式）
_MQ_QE = np.array([
    0x5601, 0x3401, 0x1801, 0x0ac1, 0x0521, 0x0221,
    0x5601, 0x5401, 0x4801, 0x3801, 0x3001, 0x2401,
    0x1c01, 0x1601, 0x5601, 0x5401, 0x5101, 0x4801,
    0x3801, 0x3401, 0x3001, 0x2801, 0x2401, 0x2201,
    0x1c01, 0x1801, 0x1601, 0x1401, 0x1201, 0x1101,
    0x0ac1, 0x09c1, 0x08a1, 0x0521, 0x0441, 0x02a1,
    0x0221, 0x0141, 0x0111, 0x0085, 0x0049, 0x0025,
    0x0015, 0x0009, 0x0005, 0x0001,
    0x5601, 0x3401, 0x1801, 0x0ac1, 0x0521, 0x0221,
    0x5601, 0x5401, 0x4801, 0x3801, 0x3001, 0x2401,
    0x1c01, 0x1601, 0x5601, 0x5401, 0x5101, 0x4801,
    0x3801, 0x3401, 0x3001, 0x2801, 0x2401, 0x2201,
    0x1c01, 0x1801, 0x1601, 0x1401, 0x1201, 0x1101,
    0x0ac1, 0x09c1, 0x08a1, 0x0521, 0x0441, 0x02a1,
    0x0221, 0x0141, 0x0111, 0x0085, 0x0049, 0x0025,
    0x0015, 0x0009, 0x0005, 0x0001,
], dtype=np.uint16)

_MQ_MPS = np.array([
    1, 2, 3, 4, 5, 38, 7, 8, 9, 10, 11, 12, 13, 29, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 45,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 84, 85, 86, 87, 88, 89, 90, 91, 91,
], dtype=np.uint8)

_MQ_LPS = np.array([
    46, 6, 9, 12, 29, 33, 52, 14, 14, 14, 17, 18, 20, 21, 60, 14, 15, 16, 17, 18,
    19, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43,
    1, 6, 9, 12, 29, 33, 6, 14, 14, 14, 17, 18, 20, 21, 14, 14, 15, 16, 17, 18,
    19, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43,
], dtype=np.uint8)


@njit(cache=True)
def _get_mps(state_idx):
    """获取状态的MPS值"""
    return 1 if state_idx > 45 else 0


@njit(cache=True)
def _encode_bit_numba(a, c, ct, context_val, bit):
    """
    Numba加速的位编码核心函数

    参数:
        a: 区间大小（传入引用）
        c: 代码寄存器（传入引用）
        ct: 计数器（传入引用）
        context_val: 当前上下文状态索引
        bit: 要编码的位

    返回:
        (new_a, new_c, new_ct, new_context_val)
    """
    qe = _MQ_QE[context_val]
    mps = _get_mps(context_val)

    if bit == mps:
        # 编码MPS
        a -= qe
        if a & 0x8000:
            # 无需重归一化
            c += qe
            new_context = context_val
        else:
            if a < qe:
                a = qe
            else:
                c += qe
            new_context = _MQ_MPS[context_val]
            # 重归一化
            while (a & 0x8000) == 0:
                a <<= 1
                c <<= 1
                ct -= 1
    else:
        # 编码LPS
        a -= qe
        if a < qe:
            c += qe
        else:
            a = qe
        new_context = _MQ_LPS[context_val]
        # 重归一化
        while (a & 0x8000) == 0:
            a <<= 1
            c <<= 1
            ct -= 1

    return a, c, ct, new_context


@njit(cache=True)
def _renorme_numba(a, c, ct):
    """Numba加速的重归一化"""
    while (a & 0x8000) == 0:
        a <<= 1
        c <<= 1
        ct -= 1
    return a, c, ct


@njit(cache=True)
def pack_image_to_words(image_data, width, height):
    """
    将二值图像打包为32位字（Leptonica格式）

    参数:
        image_data: 二值图像数组 (height, width)
        width: 图像宽度
        height: 图像高度

    返回:
        打包后的数据数组
    """
    words_per_row = (width + 31) // 32
    packed = np.zeros((height, words_per_row), dtype=np.uint32)

    for y in range(height):
        for x in range(width):
            word_idx = x // 32
            bit_idx = 31 - (x % 32)  # 大端位序
            if image_data[y, x]:
                packed[y, word_idx] |= (1 << bit_idx)

    return packed


@njit(cache=True)
def encode_image_numba(image_data, width, height, context, output_bits):
    """
    Numba加速的图像编码

    参数:
        image_data: 二值图像数组
        width: 图像宽度
        height: 图像高度
        context: 上下文状态数组（会被修改）
        output_bits: 输出位列表（用于调试）

    返回:
        编码的位列表
    """
    ltp = 0  # 典型预测状态

    for y in range(height):
        # 初始化上下文
        c1 = ((get_pixel_numba(image_data, 0, y - 2, width, height) << 2) |
              (get_pixel_numba(image_data, 1, y - 2, width, height) << 1) |
              get_pixel_numba(image_data, 2, y - 2, width, height))

        c2 = ((get_pixel_numba(image_data, 0, y - 1, width, height) << 3) |
              (get_pixel_numba(image_data, 1, y - 1, width, height) << 2) |
              (get_pixel_numba(image_data, 2, y - 1, width, height) << 1) |
              get_pixel_numba(image_data, 3, y - 1, width, height))

        c3 = 0

        # 典型预测检查
        if y > 0:
            same = True
            for x in range(width):
                if (get_pixel_numba(image_data, x, y, width, height) !=
                    get_pixel_numba(image_data, x, y - 1, width, height)):
                    same = False
                    break

            sltp = ltp ^ 1
            ltp = 1 if same else 0
            # 编码TPGD位（这里简化处理，实际需要编码器）
            if ltp:
                continue  # 跳过整行

        for x in range(width):
            tval = (c1 << 11) | (c2 << 4) | c3
            v = get_pixel_numba(image_data, x, y, width, height)

            output_bits.append(tval)
            output_bits.append(v)

            # 更新上下文
            c1 = ((c1 << 1) | get_pixel_numba(image_data, x + 3, y - 2, width, height)) & 0x1F
            c2 = ((c2 << 1) | get_pixel_numba(image_data, x + 4, y - 1, width, height)) & 0x7F
            c3 = ((c3 << 1) | v) & 0x0F

    return output_bits


@njit(cache=True)
def get_pixel_numba(data, x, y, mx, my):
    """
    Numba加速的像素获取（边界安全）

    参数:
        data: 图像数组
        x: X坐标
        y: Y坐标
        mx: 图像宽度
        my: 图像高度

    返回:
        像素值（0或1），边界外返回0
    """
    if y < 0 or y >= my or x < 0 or x >= mx:
        return 0
    return data[y, x] & 1


@njit(cache=True, parallel=True)
def compute_xor_analysis_numba(first_arr, second_arr, width, height):
    """
    Numba加速的XOR差异分析

    用于符号比较器，计算两个符号的差异分布。

    参数:
        first_arr: 第一个符号数组
        second_arr: 第二个符号数组
        width: 宽度
        height: 高度

    返回:
        (diff_count, first_count, grid_counts)
    """
    xor = first_arr ^ second_arr

    diff_count = 0
    first_count = 0

    for y in range(height):
        for x in range(width):
            if xor[y, x]:
                diff_count += 1
            if first_arr[y, x]:
                first_count += 1

    return diff_count, first_count


@njit(cache=True)
def flood_fill_numba(binary, visited, start_x, start_y, width, height):
    """
    Numba加速的洪水填充算法

    用于连通组件分析。

    参数:
        binary: 二值图像
        visited: 访问标记数组
        start_x, start_y: 起始位置
        width, height: 图像尺寸

    返回:
        (pixels_count, min_x, max_x, min_y, max_y)
    """
    # 使用简单的数组作为栈
    max_size = width * height
    stack_x = np.zeros(max_size, dtype=np.int32)
    stack_y = np.zeros(max_size, dtype=np.int32)
    stack_ptr = 0

    # 压入起始点
    stack_x[stack_ptr] = start_x
    stack_y[stack_ptr] = start_y
    stack_ptr += 1

    min_x = max_x = start_x
    min_y = max_y = start_y
    pixel_count = 0

    while stack_ptr > 0:
        stack_ptr -= 1
        x = stack_x[stack_ptr]
        y = stack_y[stack_ptr]

        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if visited[y, x] or not binary[y, x]:
            continue

        visited[y, x] = True
        pixel_count += 1

        # 更新边界
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

        # 压入8个邻居
        if stack_ptr + 8 < max_size:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx != 0 or dy != 0:
                        stack_x[stack_ptr] = x + dx
                        stack_y[stack_ptr] = y + dy
                        stack_ptr += 1

    return pixel_count, min_x, max_x, min_y, max_y


@njit(cache=True, parallel=True)
def connected_components_numba(binary, min_size=10):
    """
    Numba加速的连通组件分析

    参数:
        binary: 二值图像数组
        min_size: 最小组件大小

    返回:
        组件列表，每个组件为 (x, y, w, h, pixel_count)
    """
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=np.bool_)

    # 预分配组件数组（最大可能数量）
    max_components = min(10000, height * width // min_size)
    components_x = np.zeros(max_components, dtype=np.int32)
    components_y = np.zeros(max_components, dtype=np.int32)
    components_w = np.zeros(max_components, dtype=np.int32)
    components_h = np.zeros(max_components, dtype=np.int32)
    components_count = np.zeros(max_components, dtype=np.int32)
    num_components = 0

    for y in range(height):
        for x in range(width):
            if binary[y, x] and not visited[y, x]:
                # 找到一个新组件
                count, min_x, max_x, min_y, max_y = flood_fill_numba(
                    binary, visited, x, y, width, height
                )

                if count >= min_size and num_components < max_components:
                    components_x[num_components] = min_x
                    components_y[num_components] = min_y
                    components_w[num_components] = max_x - min_x + 1
                    components_h[num_components] = max_y - min_y + 1
                    components_count[num_components] = count
                    num_components += 1

    return num_components, components_x[:num_components], components_y[:num_components], \
           components_w[:num_components], components_h[:num_components], components_count[:num_components]


# 编译时预热（可选）
def _warmup():
    """预编译关键函数"""
    # 创建小测试数据触发编译
    test_img = np.zeros((10, 10), dtype=np.uint8)
    test_img[3:7, 3:7] = 1

    # 触发编译
    _ = pack_image_to_words(test_img, 10, 10)

    # 触发连通组件编译
    visited = np.zeros_like(test_img, dtype=np.bool_)
    _ = flood_fill_numba(test_img, visited, 5, 5, 10, 10)


# 导入时执行预热
# _warmup()
