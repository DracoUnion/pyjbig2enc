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
算术编码器测试

测试 MQ 算术编码器的核心功能。
"""

import unittest
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyjbig2enc.arith import (
    ArithmeticEncoder,
    encode_image,
    encode_int,
    encode_oob,
    IntProc,
)


class TestArithmeticEncoder(unittest.TestCase):
    """测试算术编码器的基本功能"""

    def test_init(self):
        """测试编码器初始化"""
        encoder = ArithmeticEncoder()

        # 检查初始状态
        self.assertEqual(encoder.a, 0x8000)  # 满区间
        self.assertEqual(encoder.c, 0)
        self.assertEqual(encoder.ct, 12)
        self.assertEqual(encoder.bp, -1)

    def test_simple_encode(self):
        """测试简单位编码"""
        encoder = ArithmeticEncoder()

        # 编码一些位
        encoder.encode_bit(0, 0)
        encoder.encode_bit(0, 1)
        encoder.encode_bit(0, 0)
        encoder.encode_bit(0, 1)

        # 完成编码
        encoder.finalize()

        # 检查有输出
        self.assertGreater(encoder.datasize(), 0)

        # 获取数据
        data = encoder.get_bytes()
        self.assertIsInstance(data, bytes)
        self.assertGreater(len(data), 0)

    def test_context_isolation(self):
        """测试不同上下文的隔离"""
        encoder = ArithmeticEncoder()

        # 在不同上下文中编码
        for ctx in range(10):
            encoder.encode_bit(ctx, ctx % 2)

        encoder.finalize()

        # 应该有输出
        self.assertGreater(encoder.datasize(), 0)

    def test_multiple_finalize(self):
        """测试多次编码和完成"""
        encoder = ArithmeticEncoder()

        # 第一次编码
        encoder.encode_bit(0, 0)
        encoder.finalize()
        size1 = encoder.datasize()

        # 重置并再次编码
        encoder.reset()
        encoder.encode_bit(0, 1)
        encoder.finalize()
        size2 = encoder.datasize()

        # 两次应该有输出
        self.assertGreater(size1, 0)
        self.assertGreater(size2, 0)


class TestIntegerEncoding(unittest.TestCase):
    """测试整数编码功能"""

    def test_encode_small_integers(self):
        """测试小整数编码"""
        encoder = ArithmeticEncoder()

        # 编码一些小整数
        for i in range(10):
            encode_int(encoder, IntProc.IADT, i)

        encoder.finalize()
        self.assertGreater(encoder.datasize(), 0)

    def test_encode_negative_integers(self):
        """测试负整数编码"""
        encoder = ArithmeticEncoder()

        # 编码一些负整数
        for i in range(-10, 10):
            encode_int(encoder, IntProc.IADH, i)

        encoder.finalize()
        self.assertGreater(encoder.datasize(), 0)

    def test_encode_oob(self):
        """测试 OOB 标记编码"""
        encoder = ArithmeticEncoder()

        encode_int(encoder, IntProc.IADW, 10)
        encode_oob(encoder, IntProc.IADW)
        encode_int(encoder, IntProc.IADW, 20)

        encoder.finalize()
        self.assertGreater(encoder.datasize(), 0)


class TestImageEncoding(unittest.TestCase):
    """测试图像编码功能"""

    def test_encode_small_image(self):
        """测试小图像编码"""
        encoder = ArithmeticEncoder()

        # 创建一个 8x8 的测试图像数据
        # 每行 8 位 = 1 字节
        data = bytes([
            0b10101010,
            0b01010101,
            0b11111111,
            0b00000000,
            0b10101010,
            0b01010101,
            0b11111111,
            0b00000000,
        ])

        encode_image(encoder, data, 8, 8, False)
        encoder.finalize()

        self.assertGreater(encoder.datasize(), 0)

    def test_encode_with_tpgd(self):
        """测试带 TPGD 的图像编码"""
        encoder = ArithmeticEncoder()

        # 创建一个 8x8 的测试图像
        data = bytes([
            0b10101010,
            0b10101010,  # 与上一行相同（TPGD 应该能检测到）
            0b01010101,
            0b01010101,  # 与上一行相同
            0b11111111,
            0b00000000,
            0b11111111,
            0b00000000,
        ])

        encode_image(encoder, data, 8, 8, True)
        encoder.finalize()

        self.assertGreater(encoder.datasize(), 0)


class TestDataIntegrity(unittest.TestCase):
    """测试数据完整性"""

    def test_data_consistency(self):
        """测试编码数据的一致性"""
        encoder = ArithmeticEncoder()

        # 编码固定模式
        for i in range(100):
            encoder.encode_bit(i % 16, i % 2)

        encoder.finalize()

        # 获取数据两次应该相同
        data1 = encoder.get_bytes()
        data2 = encoder.get_bytes()

        self.assertEqual(data1, data2)

    def test_buffer_output(self):
        """测试缓冲区输出"""
        encoder = ArithmeticEncoder()

        encoder.encode_bit(0, 1)
        encoder.encode_bit(0, 0)
        encoder.finalize()

        # 使用 get_bytes
        data1 = encoder.get_bytes()

        # 使用 to_buffer
        buf = bytearray(encoder.datasize())
        encoder.to_buffer(buf)

        self.assertEqual(data1, bytes(buf))


if __name__ == '__main__':
    unittest.main()
