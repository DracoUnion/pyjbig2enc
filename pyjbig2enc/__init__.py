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
JBIG2 Encoder - Python rewrite of jbig2enc

JBIG2 encodes bi-level (1 bpp) images using a number of clever tricks to get
better compression than G4. This encoder can:
   * Generate JBIG2 files, or fragments for embedding in PDFs
   * Generic region encoding
   * Perform symbol extraction, classification and text region coding
   * Perform refinement coding and,
   * Compress multipage documents
"""

__version__ = "0.29"

from .encoder import (
    Jbig2Context,
    encode_generic,
    auto_threshold,
    auto_threshold_using_hash,
)

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
)

from .segments import Segment

__all__ = [
    'Jbig2Context',
    'encode_generic',
    'auto_threshold',
    'auto_threshold_using_hash',
    'JBIG2_FILE_MAGIC',
    'SegmentType',
    'Jbig2FileHeader',
    'Jbig2PageInfo',
    'Jbig2GenericRegion',
    'Jbig2SymbolDict',
    'Jbig2TextRegion',
    'Jbig2TextRegionAtflags',
    'Jbig2TextRegionSyminsts',
    'Segment',
    '__version__',
]