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
符号比较器模块（Numba加速版本）

本模块提供符号（字形）等价性判断功能。
用于自动阈值处理，合并视觉上相似的符号。

比较算法基于论文：
"Compression of Bi-level Images with JBIG2"
通过分析两个符号的差异分布来判断是否等价。
"""

import numpy as np
from PIL import Image
from typing import Tuple
import math
from numba import njit, int32, uint8, boolean, float64


@njit(cache=True)
def _are_equivalent_numba(first_arr: uint8[:, :], second_arr: uint8[:, :],
                          threshold: float64) -> boolean:
    """
    Numba加速的符号等价性判断

    参数:
        first_arr: 第一个符号数组 (H, W)
        second_arr: 第二个符号数组 (H, W)
        threshold: 差异阈值

    返回:
        bool: 如果符号等价返回True
    """
    h, w = first_arr.shape

    # 计算XOR差异
    diff_count = 0
    first_count = 0
    for i in range(h):
        for j in range(w):
            if first_arr[i, j]:
                first_count += 1
            if first_arr[i, j] ^ second_arr[i, j]:
                diff_count += 1

    if first_count == 0:
        first_count = 1

    threshold_count = int(first_count * threshold)
    if diff_count > threshold_count:
        return False

    if diff_count == 0:
        return True

    # 网格分析参数
    divider = 9

    vertical_part = h // divider
    horizontal_part = w // divider

    if vertical_part < 1 or horizontal_part < 1:
        return diff_count <= threshold_count

    # 计算阈值
    a = max(horizontal_part, vertical_part) / 2
    b = min(horizontal_part, vertical_part) / 2
    point_thresh = a * b * math.pi * 0.5
    vline_thresh = (vertical_part * (horizontal_part // 2)) * 0.9
    hline_thresh = (horizontal_part * (vertical_part // 2)) * 0.9

    # 创建差异计数网格
    parsed_pix_counts = np.zeros((divider, divider), dtype=np.int32)
    horizontal_counts = np.zeros((divider * 2, divider), dtype=np.int32)
    vertical_counts = np.zeros((divider, divider * 2), dtype=np.int32)

    h_counter = 0
    for hi in range(divider):
        h_start = horizontal_part * hi + h_counter
        if hi == divider - 1:
            h_counter = 0
            h_end = w
        else:
            if (w - h_counter) % divider > 0:
                h_end = h_start + horizontal_part + 1
                h_counter += 1
            else:
                h_end = h_start + horizontal_part

        v_counter = 0
        for vi in range(divider):
            v_start = vertical_part * vi + v_counter
            if vi == divider - 1:
                v_counter = 0
                v_end = h
            else:
                if (h - v_counter) % divider > 0:
                    v_end = v_start + vertical_part + 1
                    v_counter += 1
                else:
                    v_end = v_start + vertical_part

            # 计算区域差异
            region_count = 0
            left_count = 0
            right_count = 0
            up_count = 0
            down_count = 0

            mid_h = (h_start + h_end) // 2
            mid_v = (v_start + v_end) // 2

            for i in range(v_start, v_end):
                for j in range(h_start, h_end):
                    if i < h and j < w:
                        if first_arr[i, j] ^ second_arr[i, j]:
                            region_count += 1
                            if j < mid_h:
                                left_count += 1
                            else:
                                right_count += 1
                            if i < mid_v:
                                up_count += 1
                            else:
                                down_count += 1

            parsed_pix_counts[hi, vi] = region_count
            horizontal_counts[hi * 2, vi] = left_count
            horizontal_counts[hi * 2 + 1, vi] = right_count
            vertical_counts[hi, vi * 2] = up_count
            vertical_counts[hi, vi * 2 + 1] = down_count

    # 检查水平线条
    for i in range(divider * 2 - 1):
        for j in range(divider - 1):
            h_sum = 0
            for ii in range(2):
                for jj in range(2):
                    if i + ii < divider * 2 and j + jj < divider:
                        h_sum += horizontal_counts[i + ii, j + jj]
            if h_sum >= hline_thresh:
                return False

    # 检查垂直线条
    for i in range(divider - 1):
        for j in range(divider * 2 - 1):
            v_sum = 0
            for ii in range(2):
                for jj in range(2):
                    if i + ii < divider and j + jj < divider * 2:
                        v_sum += vertical_counts[i + ii, j + jj]
            if v_sum >= vline_thresh:
                return False

    # 检查交叉线条
    for i in range(divider - 2):
        for j in range(divider - 2):
            left_cross = 0
            right_cross = 0
            for x in range(3):
                if i + x < divider and j + x < divider:
                    left_cross += parsed_pix_counts[i + x, j + x]
                if i + x < divider and j + 2 - x >= 0 and j + 2 - x < divider:
                    right_cross += parsed_pix_counts[i + x, j + 2 - x]
            if left_cross >= hline_thresh or right_cross >= hline_thresh:
                return False

    # 检查2x2网格的累积差异
    for i in range(divider - 1):
        for j in range(divider - 1):
            grid_sum = 0
            for ii in range(2):
                for jj in range(2):
                    if i + ii < divider and j + jj < divider:
                        grid_sum += parsed_pix_counts[i + ii, j + jj]
            if grid_sum >= point_thresh:
                return False

    return True


def are_equivalent(first: Image.Image, second: Image.Image,
                   threshold: float = 0.25) -> bool:
    """
    判断两个符号是否视觉上等价

    这是JBIG2自动阈值处理的核心函数，用于合并相似符号。

    参数:
        first: 第一个符号图像（1位PIL图像）
        second: 第二个符号图像（1位PIL图像）
        threshold: 差异阈值（默认0.25，即25%）

    返回:
        bool: 如果符号等价返回True，否则返回False
    """
    # 快速检查：尺寸必须相同
    if first.size != second.size:
        return False

    w, h = first.size

    # 转换为numpy数组
    first_arr = np.array(first.convert('1')).astype(np.uint8)
    second_arr = np.array(second.convert('1')).astype(np.uint8)

    # 确保是二值图像
    if len(first_arr.shape) > 2:
        first_arr = first_arr[:, :, 0] // 255
        second_arr = second_arr[:, :, 0] // 255
    else:
        first_arr = first_arr // 255
        second_arr = second_arr // 255

    # 使用Numba加速的核心比较
    return _are_equivalent_numba(first_arr, second_arr, float64(threshold))


def compute_hash(img: Image.Image) -> int:
    """
    计算符号的哈希值

    用于快速分类符号，减少需要详细比较的符号对数。
    哈希基于：孔洞数、高度、宽度

    参数:
        img: 符号图像

    返回:
        int: 哈希值
    """
    w, h = img.size

    # 转换为numpy数组
    arr = np.array(img.convert('1')).astype(np.uint8)
    if len(arr.shape) > 2:
        arr = arr[:, :, 0] // 255
    else:
        arr = arr // 255

    # 计算孔洞数（使用Numba加速）
    holes = _count_holes_simple_numba(arr)

    return (holes + 10 * h + 10000 * w) % 10000000


@njit(cache=True)
def _count_holes_simple_numba(arr: uint8[:, :]) -> int32:
    """
    Numba加速的简单孔洞计数

    对小的二值图像使用简化算法。
    """
    h, w = arr.shape

    # 计算4连通的组件数
    visited = np.zeros((h, w), dtype=boolean)
    components = 0

    # 使用简单数组作为栈
    max_stack = h * w
    stack_i = np.zeros(max_stack, dtype=int32)
    stack_j = np.zeros(max_stack, dtype=int32)

    for i in range(h):
        for j in range(w):
            if arr[i, j] and not visited[i, j]:
                components += 1
                # BFS
                stack_ptr = 0
                stack_i[stack_ptr] = i
                stack_j[stack_ptr] = j
                stack_ptr += 1
                visited[i, j] = True

                while stack_ptr > 0:
                    stack_ptr -= 1
                    ci = stack_i[stack_ptr]
                    cj = stack_j[stack_ptr]

                    # 4连通邻居
                    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
                    for di, dj in neighbors:
                        ni = ci + di
                        nj = cj + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if arr[ni, nj] and not visited[ni, nj]:
                                visited[ni, nj] = True
                                stack_i[stack_ptr] = ni
                                stack_j[stack_ptr] = nj
                                stack_ptr += 1

    # 简化估计：假设大多数字符有0-2个孔洞
    # 根据欧拉特征粗略估计
    result = components - 1
    if result < 0:
        result = 0
    return result


def _count_holes_simple(arr: np.ndarray) -> int:
    """
    简单的孔洞计数

    对小的二值图像使用简化算法。
    """
    return _count_holes_simple_numba(arr)
