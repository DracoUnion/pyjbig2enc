import sys
import os
from pathlib import Path
from typing import List, Optional, Callable
from io import BytesIO
from PIL import Image

from .encoder import Jbig2Context, encode_generic_single, auto_threshold, auto_threshold_using_hash
from .structs import *
from .pdf import make_jb2_pdf
from .adathres import adathres

def img2jb2pdf(imgs: bytes | str | Callable) -> bytes:
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