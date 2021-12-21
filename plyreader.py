import numpy as np

PLY_PROPERTY_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}

class PLYReader:
    def __init__(self) -> None:
        self.format = ""
        self.format_v = 1.0
        self.vertex = []
        self.faces = []
        self.num_vertex = 0
        self.num_faces = 0
        self.num_triangles = 0
        self.meta_vertex = []
        self.meta_faces = []
        self.vertex_block_size = 0
        self.endian = ""
        self._vxyz_idx = [0, 1, 2]
        self._vertex_idx = 0
        self._rgba_idx = [-1, -1, -1, -1]

    def read_ply(self, filename: str):
        with open(filename, 'rb') as ply:
            assert ply.read(4) == b'ply\n', "not a ply FILE"
            _format = ply.readline().decode().strip().split(' ')
            assert _format[0] == "format"
            self.format = _format[1]
            self.format_v = float(_format[2])

            if self.format == "binary_little_endian":
                self.endian = "<"  # numpy little endian "<"
            elif self.format == "binary_big_endian":
                self.endian = ">"  # numpy big endian ">"

            self.meta_vertex.clear()
            self.meta_faces.clear()
            meta_x = None
            meta_xord = []
            while True:
                line = ply.readline().decode().strip().split(' ')
                if line[0] == "end_header":
                    break
                elif line[0] == "element":
                    if line[1] == "vertex":
                        meta_x = self.meta_vertex
                        self.num_vertex = int(line[2])
                        meta_xord.append(
                            (meta_x, self.vertex, self.num_vertex))
                    elif line[1] == "face":
                        meta_x = self.meta_faces
                        self.num_faces = int(line[2])
                        meta_xord.append((meta_x, self.faces, self.num_faces))
                    else:
                        raise ValueError("format unrecgonized %s" % line[1])
                elif line[0] == "property":
                    if line[1] == "list":
                        _dtype = "l" + self.endian + \
                            PLY_PROPERTY_DTYPE[line[2]] + \
                            self.endian + PLY_PROPERTY_DTYPE[line[3]]
                        meta_x.append((line[4], _dtype))
                    else:
                        _dtype = self.endian + PLY_PROPERTY_DTYPE[line[1]]
                        meta_x.append((line[2], _dtype))
                elif line[0] == "comment":
                    continue
                else:
                    raise ValueError("format unrecognized %s" % line[0])

            for meta in meta_xord:
                if self.endian == "":  # ascii
                    self._read_ascii(ply, *meta)
                else:
                    self._read_binary(ply, *meta)

        for i, tag in enumerate(self.meta_vertex):
            t = tag[0]
            if t == 'x':
                self._vxyz_idx[0] = i
            elif t == 'y':
                self._vxyz_idx[1] = i
            elif t == 'z':
                self._vxyz_idx[2] = i

        for i, tag in enumerate(self.meta_faces):
            t = tag[0]
            if t == "vertex_indices":
                self._vertex_idx = i
            elif t == "red":
                self._rgba_idx[0] = i
            elif t == "green":
                self._rgba_idx[1] = i
            elif t == "blue":
                self._rgba_idx[2] = i
            elif t == "alpha":
                self._rgba_idx[3] = i

        # count triangles
        self.num_triangles = 0
        for fi in self.faces:
            self.num_triangles += len(fi[self._vertex_idx]) - 2

    def _read_binary(self, ply, meta: list, dst: list, count: int):
        for i in range(count):
            item = []
            # meta has (name, dtype| bytes)
            for _, d in meta:
                if d[0] == 'l':  # list
                    # l>u1>i4
                    # 0123456
                    _raw = ply.read(int(d[3]))
                    _len = np.frombuffer(
                        _raw, dtype=np.dtype(d[1:4]), count=1)[0]
                    _dsize = int(d[6])

                    _raw = ply.read(_len * _dsize)

                    # TRIANGLE_FAN

                    item.append(
                        np.frombuffer(_raw, dtype=np.dtype(d[4:7]), count=_len)
                    )
                else:
                    # >i4
                    # 012
                    _raw = ply.read(int(d[2]))
                    item.append(
                        np.frombuffer(_raw, dtype=np.dtype(d), count=1)
                    )
            dst.append(item)

    def _read_ascii(self, ply, meta: list, dst: list, count: int):
        for i in range(count):
            line = ply.readline().decode().split()
            # meta has (name, dtype| bytes)
            item = []
            j = 0
            for _, d in meta:
                if d[0] == 'l': # list
                    # lu1i4
                    # 01234
                    _len = int(line[j])
                    # read _len items
                    j += 1
                    if d[3] == "i" or d[3] == 'u':
                        _faces = [int(line[j+_k]) for _k in range(_len)]
                    else:
                        _faces = [float(line[j+_k]) for _k in range(_len)]
                    j += _len
                    item.append(np.asarray(_faces, dtype=np.dtype(d[3:5])))
                elif d[0] == 'i' or d[0] == 'u':
                    item.append(int(line[j]))
                    j+=1
                else:
                    # just numbers
                    item.append(float(line[j]))
                    j+=1
            dst.append(item)

    def vertex_tags(self):
        return [t[0] for t in self.meta_vertex]

    def face_tags(self):
        return [t[0] for t in self.meta_faces]

    def get_vertex(self, idx, tag=None):
        if tag is None:
            return self.vertex[idx]

        for i, t in enumerate(self.meta_vertex):
            if t[0] == tag:
                return self.vertex[idx][i]
        return None

    def get_vertex_xyz(self, idx):
        return np.asarray([
            self.vertex[idx][self._vxyz_idx[0]],
            self.vertex[idx][self._vxyz_idx[1]],
            self.vertex[idx][self._vxyz_idx[2]],
        ]).reshape(-1)

    def get_rgba(self, idx):
        ans = np.ones(4, dtype=np.float32)
        for i, ci in enumerate(self._rgba_idx):
            if ci < 0:
                continue
            _t = self.faces[idx][ci]
            if _t > 1.0:
                _t = _t / 255.0
            ans[i] = _t
        return ans

    def get_face(self, idx, tag=None):
        if tag is None:
            return self.faces[idx]

        for i, t in enumerate(self.meta_faces):
            if t[0] == tag:
                return self.faces[idx][i]

        return None

    def vertex_xyz_np(self):
        out = np.zeros((self.num_vertex, 3), dtype=np.float32)
        for i in range(self.num_vertex):
            out[i] = self.get_vertex_xyz(i)
        return out

    def triangles_idx(self):
        out = np.zeros((self.num_faces, 3), dtype=np.int32)
        for i, fi in enumerate(self.faces):
            for i, fi in enumerate(self.faces):
                out[i, 0] = fi[self._vertex_idx][0]
                out[i, 1] = fi[self._vertex_idx][1]
                out[i, 2] = fi[self._vertex_idx][2]
        return out

    def face_iter(self):
        for i in range(self.num_faces):
            vtex = self.faces[i][self._vertex_idx]

            rgba = self.get_rgba(i)
            a = self.get_vertex_xyz(vtex[0])

            # triangle fan
            #   0 1 2 3 4 ...
            # > 0 1 2
            # > 0 2 3
            # > 0 2 4
            # ... 
            for i in range(1, len(vtex)-1):
                b = self.get_vertex_xyz(vtex[i])
                c = self.get_vertex_xyz(vtex[i+1])
                yield a, b, c, rgba

