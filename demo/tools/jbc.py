"""The container of the problem file format (document 11), standard library only.

read(path) -> {name: (dtype, shape, array.array)} and write(path, [(name, dtype, shape, values)]).
dtype is the format's code: 1 u8, 2 i32, 3 i64, 4 f32, 5 f64. Used for the demo's capture
files (.jbc, sections in demo/README.md).
"""

import array
import struct
import sys

MAGIC = b"\x89JBA\r\n\x1a\n"
U8, I32, I64, F32, F64 = 1, 2, 3, 4, 5
_TYPECODE = {U8: "B", I32: "i", I64: "q", F32: "f", F64: "d"}
_SIZE = {U8: 1, I32: 4, I64: 8, F32: 4, F64: 8}


def _array(dtype, values=()):
    a = array.array(_TYPECODE[dtype], values)
    assert a.itemsize == _SIZE[dtype], "unexpected item size on this platform"
    return a


def _pad(n):
    return (n + 7) // 8 * 8


def read(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not a file of this format")
    version, count = struct.unpack_from("<II", data, 8)
    if version != 1:
        raise ValueError(f"{path}: unknown version {version}")
    pos, out = 16, {}
    for _ in range(count):
        (n,) = struct.unpack_from("<I", data, pos)
        pos += 4
        name = data[pos:pos + n].decode()
        pos += n
        dtype, ndim = data[pos], data[pos + 1]
        pos += 4
        if dtype not in _SIZE:
            raise ValueError(f"{path}: unknown data type in {name}")
        shape = struct.unpack_from(f"<{ndim}Q", data, pos)
        pos = _pad(pos + 8 * ndim)
        count_items = 1
        for d in shape:
            count_items *= d
        nbytes = count_items * _SIZE[dtype]
        if pos + nbytes > len(data):
            raise ValueError(f"{path}: truncated")
        a = _array(dtype)
        a.frombytes(data[pos:pos + nbytes])
        if sys.byteorder != "little":
            a.byteswap()
        pos = _pad(pos + nbytes)
        if name in out:
            raise ValueError(f"{path}: section {name} appears twice")
        out[name] = (dtype, tuple(shape), a)
    if pos != len(data):
        raise ValueError(f"{path}: data after the last section")
    return out


def write(path, sections):
    out = bytearray(MAGIC)
    out += struct.pack("<II", 1, len(sections))
    for name, dtype, shape, values in sections:
        a = values if isinstance(values, array.array) and values.typecode == _TYPECODE[dtype] else _array(dtype, values)
        count_items = 1
        for d in shape:
            count_items *= d
        if count_items != len(a):
            raise ValueError(f"section {name}: shape {shape} does not match {len(a)} values")
        if sys.byteorder != "little":
            a = _array(dtype, a)
            a.byteswap()
        encoded = name.encode()
        out += struct.pack("<I", len(encoded)) + encoded + struct.pack("<BBH", dtype, len(shape), 0)
        out += struct.pack(f"<{len(shape)}Q", *shape)
        out += bytes(_pad(len(out)) - len(out))
        out += a.tobytes()
        out += bytes(_pad(len(out)) - len(out))
    with open(path, "wb") as f:
        f.write(out)
