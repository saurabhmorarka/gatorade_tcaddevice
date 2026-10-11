"""VTK XML file I/O in pure numpy: .vtu unstructured grids and .pvd
collections, so every result opens in ParaView (and later in tcadview)
without adding VTK/meshio to the pip environment.

Format choices:
  * UnstructuredGrid, VTKFile version 1.0, header_type UInt64, little endian.
  * Arrays go in one raw appended block (no base64), so large 3D states cost
    their binary size. Optional zlib compression ("vtkZLibDataCompressor",
    standard library zlib), which pays off for meshes/ids more than for
    densities.
  * String arrays (names) are FieldData, written inline as ASCII codes with
    a 0 terminator per string - VTK's own encoding for type="String".

read_vtu() reads back what write_vtu() writes (any compression setting); it
is meant for tests and for our own tools, not as a general VTK reader.
"""
import os
import struct
import xml.etree.ElementTree as ET
import zlib
from xml.sax.saxutils import quoteattr

import numpy as np

# VTK cell type ids
VTK_VERTEX = 1
VTK_LINE = 3
VTK_TRIANGLE = 5
VTK_QUAD = 9
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13

_VTK_TYPES = {
    np.dtype(np.float64): "Float64", np.dtype(np.float32): "Float32",
    np.dtype(np.int64): "Int64", np.dtype(np.int32): "Int32", np.dtype(np.int16): "Int16",
    np.dtype(np.int8): "Int8", np.dtype(np.uint64): "UInt64", np.dtype(np.uint32): "UInt32",
    np.dtype(np.uint16): "UInt16", np.dtype(np.uint8): "UInt8",
}
_NP_TYPES = {v: k for k, v in _VTK_TYPES.items()}
_BLOCK = 1 << 20     # zlib block size (uncompressed bytes)


def _as_vtk_array(a):
    a = np.asarray(a)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    elif a.dtype.kind == "f" and a.dtype not in _VTK_TYPES:
        a = a.astype(np.float64)
    elif a.dtype.kind in "iu" and a.dtype not in _VTK_TYPES:
        a = a.astype(np.int64)
    return np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))


def _encode(a, compress):
    raw = a.tobytes()
    if not compress:
        return struct.pack("<Q", len(raw)) + raw
    blocks = [raw[i:i + _BLOCK] for i in range(0, len(raw), _BLOCK)] or [b""]
    comp = [zlib.compress(b, 6) for b in blocks]
    last = len(blocks[-1]) if len(raw) else 0
    head = struct.pack(f"<{3 + len(comp)}Q", len(comp), _BLOCK, last, *map(len, comp))
    return head + b"".join(comp)


def _decode(buf, pos, dtype, compressed):
    if not compressed:
        (nbytes,) = struct.unpack_from("<Q", buf, pos)
        start = pos + 8
        return np.frombuffer(buf, dtype=dtype, count=nbytes // dtype.itemsize, offset=start)
    nblocks, _, _ = struct.unpack_from("<3Q", buf, pos)
    sizes = struct.unpack_from(f"<{nblocks}Q", buf, pos + 24)
    p = pos + 24 + 8 * nblocks
    out = []
    for s in sizes:
        out.append(zlib.decompress(buf[p:p + s]))
        p += s
    return np.frombuffer(b"".join(out), dtype=dtype)


def _string_array_xml(name, strings, indent):
    codes = []
    for s in strings:
        codes.extend(str(s).encode("utf-8"))
        codes.append(0)
    return (f'{indent}<DataArray type="String" Name={quoteattr(name)} NumberOfTuples="{len(strings)}" '
            f'format="ascii">\n{indent}  {" ".join(map(str, codes))}\n{indent}</DataArray>\n')


def write_vtu(path, points, cells, cell_types, point_data=None, cell_data=None, field_data=None,
              compress=False):
    """Write an unstructured grid.

    points      (N,3) (or (N,2), padded with z=0), cm or any unit
    cells       list of (M_k, n_k) int connectivity arrays, one per cell type
                block, or a single (M, n) array
    cell_types  VTK type id per block (int or list of ints matching `cells`)
    point_data  {name: (N,) or (N,c) array}
    cell_data   {name: (M,) or (M,c) array}, M = total cells over all blocks
    field_data  {name: array or list of str} - dataset-wide values (names,
                biases, units); a list of str becomes a VTK String array
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[1] == 2:
        pts = np.c_[pts, np.zeros(len(pts))]
    if not isinstance(cells, (list, tuple)):
        cells, cell_types = [cells], [cell_types]
    elif np.ndim(cell_types) == 0:
        cell_types = [cell_types] * len(cells)
    conn = np.concatenate([np.asarray(c, dtype=np.int64).ravel() for c in cells]) if cells else \
        np.zeros(0, np.int64)
    sizes = np.concatenate([np.full(len(c), np.shape(c)[1], dtype=np.int64) for c in cells])
    offsets = np.cumsum(sizes)
    types = np.concatenate([np.full(len(c), t, dtype=np.uint8) for c, t in zip(cells, cell_types)])
    n_pts, n_cells = len(pts), len(types)

    blobs = []
    offset = [0]

    def darray(name, a, indent):
        a = _as_vtk_array(a)
        ncomp = 1 if a.ndim == 1 else a.shape[1]
        blob = _encode(a, compress)
        attr = f'type="{_VTK_TYPES[a.dtype.newbyteorder("=")]}"'
        if name is not None:
            attr += f" Name={quoteattr(name)}"
        xml = (f'{indent}<DataArray {attr} NumberOfComponents="{ncomp}" format="appended" '
               f'offset="{offset[0]}"/>\n')
        blobs.append(blob)
        offset[0] += len(blob)
        return xml

    def check(name, a, n, what):
        if len(a) != n:
            raise ValueError(f"{what} array {name!r} has length {len(a)}, expected {n}")

    out = ['<?xml version="1.0"?>\n',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64"'
           + (' compressor="vtkZLibDataCompressor"' if compress else "") + ">\n",
           "  <UnstructuredGrid>\n"]
    if field_data:
        out.append("    <FieldData>\n")
        for name, v in field_data.items():
            if isinstance(v, (list, tuple)) and v and all(isinstance(s, str) for s in v):
                out.append(_string_array_xml(name, v, "      "))
            elif isinstance(v, str):
                out.append(_string_array_xml(name, [v], "      "))
            else:
                a = np.atleast_1d(np.asarray(v))
                x = darray(name, a, "      ")
                out.append(x.replace('format="appended"', f'NumberOfTuples="{len(a)}" format="appended"'))
        out.append("    </FieldData>\n")
    out.append(f'    <Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cells}">\n')
    out.append("      <PointData>\n")
    for name, a in (point_data or {}).items():
        check(name, a, n_pts, "point")
        out.append(darray(name, a, "        "))
    out.append("      </PointData>\n      <CellData>\n")
    for name, a in (cell_data or {}).items():
        check(name, a, n_cells, "cell")
        out.append(darray(name, a, "        "))
    out.append("      </CellData>\n      <Points>\n")
    out.append(darray("Points", pts, "        "))
    out.append("      </Points>\n      <Cells>\n")
    out.append(darray("connectivity", conn, "        "))
    out.append(darray("offsets", offsets, "        "))
    out.append(darray("types", types, "        "))
    out.append("      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n")
    out.append('  <AppendedData encoding="raw">\n   _')
    tail = "\n  </AppendedData>\n</VTKFile>\n"
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write("".join(out).encode("utf-8"))
        for b in blobs:
            f.write(b)
        f.write(tail.encode("utf-8"))
    os.replace(tmp, path)


def write_pvd(path, entries):
    """Write a .pvd collection. entries: list of dicts with keys
    file (path relative to the .pvd's folder), timestep (float: the swept
    bias, or a step index), and optional name (ParaView shows it as a block
    name). ParaView animates over timestep - for a bias sweep that is the
    bias value, so the time slider reads in volts."""
    lines = ['<?xml version="1.0"?>\n',
             '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">\n', "  <Collection>\n"]
    for e in entries:
        name = f" name={quoteattr(str(e['name']))}" if e.get("name") is not None else ""
        lines.append(f'    <DataSet timestep="{float(e["timestep"]):.10g}" group="" part="0"{name} '
                     f"file={quoteattr(e['file'])}/>\n")
    lines.append("  </Collection>\n</VTKFile>\n")
    with open(path, "w") as f:
        f.write("".join(lines))


def read_pvd(path):
    """[(timestep, absolute file path, name or None)] from a .pvd."""
    root = ET.parse(path).getroot()
    base = os.path.dirname(os.path.abspath(path))
    return [(float(d.get("timestep")), os.path.join(base, d.get("file")), d.get("name"))
            for d in root.iter("DataSet")]


def _parse_strings(text):
    codes = [int(c) for c in text.split()]
    out, cur = [], []
    for c in codes:
        if c == 0:
            out.append(bytes(cur).decode("utf-8"))
            cur = []
        else:
            cur.append(c)
    return out


def read_vtu(path):
    """Read a file written by write_vtu. Returns dict with points (N,3),
    connectivity, offsets, types, point_data, cell_data, field_data."""
    with open(path, "rb") as f:
        buf = f.read()
    i = buf.find(b"<AppendedData")
    if i < 0:
        raise ValueError(f"{path}: no AppendedData block (not written by core.vtk_io?)")
    start = buf.index(b"_", i) + 1
    root = ET.fromstring(buf[:i] + b"</VTKFile>")
    compressed = root.get("compressor") is not None
    data = buf[start:]

    def load(el):
        dtype = _NP_TYPES[el.get("type")].newbyteorder("<")
        a = _decode(data, int(el.get("offset")), dtype, compressed).astype(dtype.newbyteorder("="))
        ncomp = int(el.get("NumberOfComponents", "1"))
        return a.reshape(-1, ncomp) if ncomp > 1 else a

    grid = root.find("UnstructuredGrid")
    piece = grid.find("Piece")
    out = {"field_data": {}, "point_data": {}, "cell_data": {}}
    fd = grid.find("FieldData")
    if fd is not None:
        for el in fd.findall("DataArray"):
            out["field_data"][el.get("Name")] = _parse_strings(el.text) if el.get("type") == "String" \
                else load(el)
    for tag, key in (("PointData", "point_data"), ("CellData", "cell_data")):
        sec = piece.find(tag)
        for el in (sec.findall("DataArray") if sec is not None else []):
            out[key][el.get("Name")] = load(el)
    out["points"] = load(piece.find("Points/DataArray"))
    for el in piece.find("Cells").findall("DataArray"):
        out[el.get("Name")] = load(el)
    return out
