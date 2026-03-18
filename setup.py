#!/usr/bin/env python3
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
JBIG2 Encoder - Python重写版本安装脚本

JBIG2是一种用于二值图像（1bpp）的高效压缩国际标准，
通常用于PDF文档中的黑白图像压缩。

特性：
- 完整的JBIG2编码器实现
- Numba加速支持（可选，可提升10-50倍性能）
- 纯Python回退实现（无需编译）

原始C++版本: https://github.com/agl/jbig2enc
"""

from setuptools import setup, find_packages
import os

# 读取README文件
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'JBIG2 Encoder - Python rewrite of jbig2enc'

setup(
    # 基本信息
    name='pyjbig2enc',
    version='0.29.0',
    description='JBIG2 Encoder - Python rewrite of jbig2enc with required Numba JIT acceleration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='wizardforcel',
    author_email='',
    url='https://github.com/agl/jbig2enc',
    license='MIT',

    # 包配置
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    # 依赖
    install_requires=[
        'Pillow>=9.0.0',  # 图像处理
        'numpy>=1.20.0',  # 数组处理
        'numba>=0.56.0',  # JIT编译加速（必需）
    ],

    # 可选依赖
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'mypy>=1.0.0',
        ],
    },

    # 入口点
    entry_points={
        'console_scripts': [
            'pyjbig2enc=pyjbig2enc.cli:main',
        ],
    },

    # 分类信息
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Python版本要求
    python_requires='>=3.8',

    # 关键词
    keywords='jbig2 image compression pdf encoder numba',
)
