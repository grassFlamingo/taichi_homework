# PLY file format reader
# naive implementation
# only support ascii
#
# struct point
# {
# 	float x, y, z;		// coordinates
# 	float nx, ny, nz;	// surface normal
# 	float s, t;		// texture coords
# };
# we won't check the order of x y z nx ny nz s t
# ref http://www.paulbourke.net/dataformats/ply/

from os import sep
import numpy as np


class NPLYReader:
    def __init__(self, filename) -> None:
        ply = open(filename, "r")

        line = ply.readline()
        assert line == "ply\n", "not a ply file"

        line = ply.readline()
        line.index("ascii")

        while True:
            line = ply.readline()
            if line.startswith("element"):
                break

        _, tag, count = line.strip().split()
        assert tag == "vertex"
        numVertex = int(count)
        p = 0
        while True:
            line = ply.readline()
            if line.startswith("element"):
                break
            else:
                p += 1
        assert p == 8, "only support x,y,z,nx,ny,nz,s,t"
        _, tag, count = line.strip().split()
        assert tag == "face"
        numFace = int(count)

        while True:
            line = ply.readline()
            if line.startswith("end_header"):
                break

        self.numVertex = numVertex
        self.vertex = []
        
        # read vertex
        # print("read vertex")
        for i in range(numVertex):
            line = ply.readline()
            self.vertex.append(
                np.fromstring(line, dtype=np.float32, sep=" ", count=p)
            )
        
        self.vertex = np.asarray(self.vertex, dtype=np.float32)

        # print(self.vertex)

        self.faces = []
        # read faces
        # print("read faces")
        for f in range(numFace):
            line = ply.readline()
            items = line.strip().split()
            items = [int(o) for o in items]
            # triangle fan
            # 1 2 3 4 5
            # -> 1 2 3; 1 3 4; 1 4 5
            for c in range(1, items[0] - 1):
                self.faces.append((
                    items[1],
                    items[c+1],
                    items[c+2],
                ))
        self.numFaces = len(self.faces)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        # print(self.faces)
        ply.close()

    def iter_points(self):
        r"""
        iter [x,y,z,nx,ny,nz,s,t] of 0, 1, 2
        ```
             0
            / \
           /   \
          /     \
        1 ------ 2
        ```
        """
        for a, b, c in self.faces:
            x = self.vertex[a]
            y = self.vertex[b]
            z = self.vertex[c]
            yield x, y, z

    def it_points_3(self):
        r"""
        iter x,y,z of 0, 1, 2
        ```
             0
            / \
           /   \
          /     \
        1 ------ 2
        ```
        """
        for a, b, c in self.faces:
            x = np.asarray(self.vertex[a][0:3])
            y = np.asarray(self.vertex[b][0:3])
            z = np.asarray(self.vertex[c][0:3])
            yield x, y, z

    def it_points_3st(self):
        r"""
        iter x,y,z,s,t of 0, 1, 2
        ```
             0
            / \
           /   \
          /     \
        1 ------ 2
        ```
        """
        for a, b, c in self.faces:
            x = self.vertex[a][(0, 1, 2)]
            y = self.vertex[b][(0, 1, 2)]
            z = self.vertex[c][(0, 1, 2)]
            xst = self.vertex[a][(6,7)]
            yst = self.vertex[b][(6,7)]
            zst = self.vertex[c][(6,7)]
            yield x, y, z, xst, yst, zst
