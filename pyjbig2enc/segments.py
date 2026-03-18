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
JBIG2段头处理模块

JBIG2文件由多个段（Segment）组成，每个段包含一个头部和数据。
本模块定义了段头的结构和序列化方法。

段头格式：
- 段号（4字节）
- 段类型和标志（1字节）
- 保留段计数（1字节）
- 引用段列表（可变长度）
- 页关联（1或4字节）
- 数据长度（4字节）
"""

import struct
from typing import List
from .structs import SegmentType


class Segment:
    """
    JBIG2段头类

    表示JBIG2文件中的一个段的头部信息。
    每个段都有一个唯一的段号，可以引用之前定义的段。

    属性说明：
    - number: 段号（唯一标识）
    - type: 段类型（见SegmentType枚举）
    - deferred_non_retain: 延迟非保留标志
    - retain_bits: 保留位（决定段是否保留在内存中）
    - referred_to: 引用的段号列表（本段依赖的段）
    - page: 关联的页号（0表示全局段）
    - len: 段数据长度（字节）
    """

    def __init__(self):
        """初始化一个空的段头"""
        self.number: int = 0          # 段号
        self.type: int = 0            # 段类型
        self.deferred_non_retain: int = 0  # 延迟非保留标志
        self.retain_bits: int = 0     # 保留位
        self.referred_to: List[int] = []   # 引用的段号列表
        self.page: int = 0            # 页号（0=全局）
        self.len: int = 0             # 数据长度

    def reference_size(self) -> int:
        """
        计算引用段号的字段大小

        根据段号的大小，引用可以使用1、2或4字节存储：
        - 段号 <= 256: 使用1字节
        - 段号 <= 65536: 使用2字节
        - 否则: 使用4字节

        返回:
            int: 每个引用段号所需的字节数
        """
        if self.number <= 256:
            return 1
        elif self.number <= 65536:
            return 2
        else:
            return 4

    def page_size(self) -> int:
        """
        计算页关联字段的大小

        页号<=255时使用1字节，否则使用4字节

        返回:
            int: 页关联字段的字节数（1或4）
        """
        return 1 if self.page <= 255 else 4

    def size(self) -> int:
        """
        计算整个段头的字节数

        段头大小 = 固定头部(6字节) + 引用列表 + 页关联 + 长度字段(4字节)

        返回:
            int: 段头的总字节数
        """
        refsize = self.reference_size()
        pagesize = self.page_size()
        # 6 = sizeof(jbig2_segment) = 4(number) + 1(flags) + 1(count)
        return 6 + refsize * len(self.referred_to) + pagesize + 4

    def write(self, buf: bytearray, offset: int = 0) -> int:
        """
        将段头序列化为二进制数据

        将本段头的所有字段写入到提供的缓冲区中。
        使用大端字节序。

        参数:
            buf: 目标字节数组
            offset: 起始偏移量

        返回:
            int: 写入的字节数
        """
        # 计算引用大小
        refsize = self.reference_size()
        pagesize = self.page_size()

        # 构建第一个标志字节
        # 位0-5: 段类型
        # 位6: page_assoc_size (页关联大小，0=1字节，1=4字节)
        # 位7: deferred_non_retain
        seg_flags = (self.type & 0x3F)
        if pagesize == 4:
            seg_flags |= 0x40  # 设置page_assoc_size位
        if self.deferred_non_retain:
            seg_flags |= 0x80

        # 构建第二个标志字节
        # 位0-4: retain_bits
        # 位5-7: 引用段数（如果少于8个）
        ref_count = len(self.referred_to)
        seg_flags2 = (self.retain_bits & 0x1F)
        if ref_count < 8:
            seg_flags2 |= (ref_count << 5)

        # 写入段号（4字节大端）
        struct.pack_into('>I', buf, offset, self.number)
        offset += 4

        # 写入标志字节
        buf[offset] = seg_flags
        offset += 1
        buf[offset] = seg_flags2
        offset += 1

        # 写入引用段号
        if ref_count >= 8:
            # 引用数>=8时，使用5位无法存储，需要额外编码
            # 这里简化处理，实际应该使用扩展格式
            pass

        for ref in self.referred_to:
            if refsize == 4:
                struct.pack_into('>I', buf, offset, ref)
            elif refsize == 2:
                struct.pack_into('>H', buf, offset, ref)
            else:
                buf[offset] = ref & 0xFF
            offset += refsize

        # 写入页关联
        if pagesize == 4:
            struct.pack_into('>I', buf, offset, self.page)
        else:
            buf[offset] = self.page & 0xFF
        offset += pagesize

        # 写入数据长度
        struct.pack_into('>I', buf, offset, self.len)
        offset += 4

        return offset

    def to_bytes(self) -> bytes:
        """
        将段头转换为字节串

        返回:
            bytes: 序列化后的段头数据
        """
        buf = bytearray(self.size())
        self.write(buf)
        return bytes(buf)

    def __repr__(self) -> str:
        """返回段的字符串表示，便于调试"""
        return (f"Segment(number={self.number}, type={self.type}, "
                f"page={self.page}, len={self.len}, refs={self.referred_to})")
