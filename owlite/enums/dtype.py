from enum import Enum

import numpy as np


# pylint: disable=invalid-name
class DType(Enum):
    """The enum for specifying available np.dtype objects"""

    FLOAT16 = np.dtype("float16")
    FLOAT32 = np.dtype("float32")
    FLOAT64 = np.dtype("float64")

    UINT8 = np.dtype("uint8")
    INT8 = np.dtype("int8")
    UINT16 = np.dtype("uint16")
    INT16 = np.dtype("int16")
    UINT32 = np.dtype("uint32")
    INT32 = np.dtype("int32")
    UINT64 = np.dtype("uint64")
    INT64 = np.dtype("int64")

    COMPLEX64 = np.dtype("complex64")
    COMPLEX128 = np.dtype("complex128")

    BOOL = np.dtype("bool")
