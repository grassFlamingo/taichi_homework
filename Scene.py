from locale import NOEXPR
import taichi as ti
import numpy as np
import math
import warnings

from taichi.lang.ops import bit_xor


def _var_as_tuple(x, length):
    if isinstance(x, (tuple, list)):
        if len(x) >= length:
            return tuple([x[i] for i in range(length)])
        else:
            return tuple([x for _ in range(length)])
    else:
        return tuple([x for _ in range(length)])

def _ensure_ti_field(x):
    if x is None:
        return None
    elif isinstance(x, np.ndarray):
        if x.dtype == np.float32:
            tty = ti.f32
        elif x.dtype == np.int32:
            tty = ti.i32
        elif x.dtype == np.int8:
            tty = ti.i8
        elif x.dtype == np.uint8:
            tty = ti.u8
        else:
            raise ValueError(f"unsupported dtype {x.dtype}")
        out = ti.field(tty, x.shape)
        out.from_numpy(x)
        return out
    else:
        return x

# these 8 functions below are copied from course examples


@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])


@ti.func
def random_in_unit_sphere():
    p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    return p


@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()


@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point


@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal


@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel


@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)


M_unknown = 0
M_light_source = 1
M_metal = 2
M_fuzzy_metal = 5
M_dielectric = 3
M_diffuse = 4


@ti.data_oriented
class Ray:
    """A ray
    ray equation: origin + t * direction
    """

    def __init__(self, origin, direction) -> None:
        self.origin = origin
        self.direction = direction / direction.norm()

    @ti.func
    def at(self, t: float):
        return self.origin + t * self.direction

    def __str__(self) -> str:
        return f"Ray(start={self.origin}, dir={self.direction})"


@ti.func
def _sphere_hit(x, r, ray, time_min: float, time_max: float):
    """ray sphere intersection.
    refernece:
    Fundamentals of Computer Graphics 4-th Edition, Chapter 4, section 4.4.1; page 76
    """
    is_hit = 0
    hit_time = time_max
    hit_point = ti.Vector([0.0, 0.0, 0.0])
    hit_normal = ti.Vector([0.0, 1.0, 0.0])
    is_inside = 0

    # the direction of ray is normed to 1
    oc = ray.origin - x

    doc = ray.direction.dot(oc)

    t = doc * doc - (oc.dot(oc) - r * r)
    # t < 0: no intersection
    if t > time_min:  # have intersection
        _t = ti.sqrt(t)
        if _t < hit_time:
            # - doc +/- sqrt(t)
            t1 = -doc + _t
            t2 = -doc - _t

            if t1 < 0:
                if t2 > 0:
                    # ray is inside this sphere
                    # ( <-- t1 - O-> --- t2 ---> )
                    is_hit = 1
                    hit_time = t2
                    is_inside = 1
                # else:  # no solution
                #     # ( <--t2-- ) <-- t1 -- O->
                #     print("no solution", t1, t2)
            else:
                if t2 > 0:
                    # O-> -- t1 -->( --- t2 --> )
                    # O-> -- t2 -->( --- t1 --> )
                    is_hit = 1
                    hit_time = ti.min(t1, t2)
                else:
                    # ray is inside this sphere
                    # ( <-- t2 -- O-> -- t1 --> )
                    is_hit = 1
                    is_inside = 1
                    hit_time = t2

    # compute normal
    if is_hit == 1:
        hit_point = ray.at(hit_time)
        if is_inside:
            # normal vector is oppositive
            hit_normal = (x - hit_point) / r
        else:
            hit_normal = (hit_point - x) / r

    return is_hit, hit_time, hit_point, hit_normal, is_inside


@ti.data_oriented
class Sphere:
    """A Sphere
    sphere function: ||x - center||_F^2 = radius^2
    """

    def __init__(self, center, radius, material, color) -> None:
        self.center = center
        self.radius = radius
        self.material = material
        self.color = color

    @ti.func
    def ray_intersect(self, ray, time_min: float, time_max: float):
        """ray sphere intersection.
        refernece:
        Fundamentals of Computer Graphics 4-th Edition, Chapter 4, section 4.4.1; page 76
        """
        is_hit, hit_time, hit_point, hit_normal, is_inside = _sphere_hit(
            self.center, self.radius, ray, time_min, time_max)

        return is_hit, hit_time, hit_point, hit_normal, self.material, self.color, is_inside

    def __str__(self) -> str:
        return f"Sphere(center={self.center}, radius={self.radius}, material={self.material}, color={self.color})"


@ti.data_oriented
class Plane:
    """An infinity large plane

    plane equaltion: normal dot (x - center)
    """

    def __init__(self, center, normal, material, color) -> None:
        self.center = center
        self.normal = normal / normal.norm(1e-8)
        self.material = material
        self.color = color

    @ti.func
    def ray_intersect(self, ray, time_min, time_max):
        is_hit = 0
        hit_time = time_max
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0

        # > 0 same direction; never hit
        # = 0 orthogonal; never hit
        if ray.direction.dot(self.normal) < 0:
            # n (p + td - c) = 0
            # t = n(c - p) / (nd)
            hit_time = self.normal.dot(self.center - ray.origin) / \
                self.normal.dot(ray.direction)
            if hit_time < time_min:
                is_hit = 0
            elif hit_time < time_max:
                is_hit = 1
                hit_normal = self.normal
                hit_point = ray.at(hit_time)

        return is_hit, hit_time, hit_point, hit_normal, self.material, self.color, is_inside

    def __str__(self) -> str:
        return "Plane()"

@ti.pyfunc
def _parallelogram_init(a, x, b):
    """
    ```
       a
      /
     /
    x --- b
    ```
    """
    ax = a - x
    bx = b - x
    normal = bx.cross(ax)
    area = normal.norm()  # what is the norm is 0?
    # assert area >= 1e-12, "not a parallelogram"
    detidx = 0
    idet = 0.0
    if area < 1e-12:
        # not a parallglogram
        normal = ti.Vector([0.0, 0.0, 0.0])
    else:
        normal = normal / area

        # | i     j     k     |
        # | ax[0] ax[1] ax[2] |
        # | bx[0] bx[1] bx[2] |
        # try to compute its 2D determinant

        det = ax[1] * bx[2] - ax[2] * bx[1]
        if ti.abs(det) < 1e-8:
            # previous det is too small; try another one
            det = ax[0] * bx[2] - ax[2] * bx[0]
            detidx = 1

        if ti.abs(det) < 1e-8:
            # previous dets are too small; try another one
            det = ax[0] * bx[1] - ax[1] * bx[0]
            detidx = 2

        idet = 1.0 / det

    # assert(ti.abs(det) > 1e-8)  # "this could not happen, just in case"
    return ax, bx, normal, area, detidx, idet


@ti.func
def _parallelogram_alpha_beta(px, ax, bx, idetidx, idet):
    """
    check point is in parallelogram; 
    please convinced that this point is in the plane of this parallelogram.
    p = x + alpha (a-x) + beta (b-x)
    p - x = alpha (a-x) + beta (b-x)

    px[0] = alpha ax[0] + beta bx[0]
    px[1] = alpha ax[1] + beta bx[1]
    px[2] = alpha ax[2] + beta bx[2]

    detA0 = ax[1] bx[2] - ax[2] bx[1]
    detA1 = ax[0] bx[2] - ax[2] bx[0]
    detA2 = ax[0] bx[1] - ax[1] bx[0]

    ```
    | a b | -> inverse | d -c | 
    | c d |            | -b a | / detA
    ```

    if idetidx == 0:
        alpha = 1/detA (px[0] ax[1] + px[1] ax[2])
        beta  = 1/detA (px[0] bx[1] + px[1] bx[2])
    if idetidx == 1:
        alpha = 1/detA (px[0] ax[0] + px[1] ax[2])
        beta  = 1/detA (px[0] bx[0] + px[1] bx[2])
    if idetidx == 2:
        alpha = 1/detA (px[0] ax[0] + px[1] ax[1])
        beta  = 1/detA (px[0] bx[0] + px[1] bx[1])
    """

    alpha, beta = -1.0, -1.0

    if idetidx == 0:  # a1 a2 | b1 b2
        alpha = px[1] * bx[2] - px[2] * bx[1]
        beta = px[2] * ax[1] - px[1] * ax[2]
    elif idetidx == 1:  # a0 a2 | b0 b2
        alpha = px[0] * bx[2] - px[2] * bx[0]
        beta = px[2] * ax[0] - px[0] * ax[2]
    else:  # 2 a0 a1 | b0 b1
        alpha = px[0] * bx[1] - px[1] * bx[0]
        beta = px[1] * ax[0] - px[0] * ax[1]

    return alpha * idet, beta * idet

# reference: https://stackoverflow.com/questions/2563849/ray-box-intersection-theory


@ti.func
def ray_intersect_solid_box_face(uhit, omin, omax, axis):
    """
    - uhit: [t * rx, t * ry, r * rz] hit point
    - omin: [min.x, min.y, min.z]
    - omax: [max.x, max.y, max.z]
    - axis: the axis to ignore
    """
    ans = 1
    for i in ti.static(range(3)):
        if i != axis:
            if uhit[i] < omin[i] or uhit[i] > omax[i]:
                ans = 0
    # return umin <= uhit and uhit <= umax and vmin <= vhit and vhit <= vmax
    return ans


@ti.func
def ray_intersect_solid_box(ray, cmin, cmax, time_min: float, time_max: float):
    """
    intersect ray with solid box;
    reference: Computer Graphic Book

    ray hit the plane and hit point in the plane

    n (x - (o + t d)) = 0
    => n (x - o) = t n d
    n = ( 1, 0, 0), (0,  1, 0), (0, 0,  1)
        (-1, 0, 0), (0, -1, 0), (0, 0, -1)
    => ti = (x - o)[i] / d[i]
    """

    omin = cmin - ray.origin
    omax = cmax - ray.origin

    t = 0.0
    tmin = time_max
    for i in ti.static(range(3)):
        rdi = ray.direction[i]
        if rdi != 0:
            t = omin[i] / rdi
            if t > time_min and t < tmin:
                hpoint = ray.at(t)
                ishit = ray_intersect_solid_box_face(hpoint, cmin, cmax, i)
                if ishit == 1:
                    tmin = t

            t = omax[i] / rdi
            if t > time_min and t < tmin:
                hpoint = ray.at(t)
                ishit = ray_intersect_solid_box_face(hpoint, cmin, cmax, i)
                if ishit == 1:
                    tmin = t

    is_hit = 0
    if tmin > time_min and tmin < time_max:
        is_hit = 1
    return is_hit, tmin


@ti.func
def solid_box_box_intersection(amin, amax, bmin, bmax):
    ans = 1
    for i in ti.static(range(3)):
        if ans == 1 and max(amin[i], bmin[i]) > min(amax[i], bmax[i]):
            ans = 0
    return ans


class KDTreeNode:
    def __init__(self) -> None:
        self.items = None
        self.cmin = None
        self.cmax = None
        self.left = None
        self.mid = None
        self.right = None
        self.lritems = None


@ti.data_oriented
class KDTree:
    """KD Tree; only works for TriangleSoup and ParallelogramSoup
    """

    def __init__(self, meshes) -> None:
        self.count = meshes.shape[0]
        self.meshes = meshes
        self.root = None
        # cmin, cmax
        self.meshboxes = ti.Vector.field(3, ti.f32, (self.count, 2))
        self._compute_obj_boxes()

    def build_tree(self):
        if self.count <= 0:
            return
        self.root = self._split_kd([i for i in range(self.count)], 0, 128)

    def flat_tree_to_taichi(self, kdflat, kdcmin, kdcmax):
        self._flat_tree_to_taichi(kdflat, kdcmin, kdcmax, 0, 0, self.root)
        # kdcmin[0] = self.root.cmin
        # kdcmax[0] = self.root.cmax
        # kdflat[0] = 0
        # kdflat[1] = -1
        # kdflat[2] = self.count
        # for i in range(self.count):
        #     kdflat[3+i] = i

    def _flat_tree_to_taichi(self, kdflat, kdcmin, kdcmax, fidx, cidx, root):
        if root is None:
            # write -1, 0 to kdflat
            kdcmin[cidx] = ti.Vector([0.0, 0.0, 0.0])
            kdcmax[cidx] = ti.Vector([0.0, 0.0, 0.0])
            kdflat[fidx] = cidx
            kdflat[fidx + 1] = -1
            kdflat[fidx + 2] = 0
            return 3, 1

        # [kdflat]
        # 0: corner index;
        # 1: mark
        # if mark >= 0:
        #   1: left index
        #   2: right index
        #   3: mid index
        #   4： lr data index
        # else:
        #   2: data len (n)
        #   3+0: triangle index 0
        #   3+1: triangle index 1
        #   ...: .....
        #   3+n-1: triangle index n-1
        #
        # next block

        kdcmin[cidx] = root.cmin
        kdcmax[cidx] = root.cmax
        kdflat[fidx] = cidx

        fadd, cadd = 1, 1

        if root.items is None:  # have childs
            fadd += 4  # left right mid, lrdata
            for i, child in enumerate([root.left, root.right, root.mid]):
                kdflat[fidx + 1 + i] = fidx + fadd
                _dxf, _dxc = self._flat_tree_to_taichi(
                    kdflat, kdcmin, kdcmax, fidx + fadd, cidx + cadd, child)
                fadd += _dxf
                cadd += _dxc

            kdflat[fidx + 1 + 3] = fidx + fadd
            kdflat[fidx + fadd] = cidx  # use the same box as root
            fadd += 1
            kdflat[fidx + fadd] = -1
            fadd += 1
            # data len
            kdflat[fidx + fadd] = len(root.lritems)
            fadd += 1
            for i in root.lritems:
                kdflat[fidx + fadd] = i
                fadd += 1
        else:
            # write to kdflat
            # left index set to self; no more left node
            kdflat[fidx + fadd] = -1
            fadd += 1
            # data length
            kdflat[fidx + fadd] = len(root.items)
            fadd += 1
            for i in root.items:
                kdflat[fidx + fadd] = i
                fadd += 1

        return fadd, cadd

    @ti.kernel
    def _compute_obj_boxes(self):
        # num faces is self.count <- static alert
        for i in range(self.count):
            di = self.meshes[i]
            x = di.x
            a = di.ax + x  # ax = a - x
            b = di.bx + x  # bx = b - x

            for j in ti.static(range(3)):
                self.meshboxes[i, 0][j] = min(x[j], a[j], b[j])
                self.meshboxes[i, 1][j] = max(x[j], a[j], b[j])

    @ti.kernel
    def _compute_box(self, cmm: ti.any_arr(), indexes: ti.any_arr()):

        # init
        cmin = ti.Vector([0.0, 0.0, 0.0])
        cmax = ti.Vector([0.0, 0.0, 0.0])
        Ex = ti.Vector([0.0, 0.0, 0.0])
        Exx = ti.Vector([0.0, 0.0, 0.0])

        for i in indexes:
            ii = indexes[i]
            di = self.meshes[ii]
            x = di.x
            a = di.ax + x  # ax = a - x
            b = di.bx + x  # bx - b - x

            for j in ti.static(range(3)):
                _cmin = min(x[j], a[j], b[j])
                _cmax = max(x[j], a[j], b[j])
                ti.atomic_min(cmin[j], _cmin)
                ti.atomic_max(cmax[j], _cmax)
                _sv = abs(_cmax - _cmin)
                Ex[j] += _sv
                Exx[j] += _sv * _sv

        # cmm: cmin, cmax, size var
        cmm[0] = cmin
        cmm[1] = cmax
        # V(x) = E(x^2) - E^2(x)
        Ex /= indexes.shape[0]
        Exx /= indexes.shape[0]
        cmm[2] = Exx - Ex * Ex

    @ti.kernel
    def _align_kd_box(
            self,
            dst: ti.any_arr(),
            objlist: ti.any_arr(),
            axis: ti.i32,
            left: ti.f32,
            mx: ti.f32,
            right: ti.f32):
        # 0; not left nor right
        # 1: left
        # 2: right
        # 3: left and right

        for i in objlist:
            # cmin, cmax, center
            obox0 = 0.0
            obox1 = 0.0
            for j in ti.static(range(3)):
                if j == axis:
                    obox0 = self.meshboxes[objlist[i], 0][j]
                    obox1 = self.meshboxes[objlist[i], 1][j]
            _t = 0
            if max(left, obox0) <= min(obox1, mx):
                _t += 1
            if max(mx, obox0) <= min(obox1, right):
                _t += 2

            dst[i] = _t

    def _split_kd(self, objlist: list, depth, max_depth=128):
        # print("_split_oct", left, right)
        # (left, right]

        objlen = len(objlist)
        objlist = np.asarray(objlist, dtype=np.int32)

        if len(objlist) == 0:
            return None

        root = KDTreeNode()
        # cmm: cmin, cmax, size var
        cmm = ti.Vector.ndarray(3, ti.f32, shape=(3,))
        self._compute_box(cmm, objlist)
        root.cmin = cmm[0]
        root.cmax = cmm[1]

        if objlen < 8 or depth >= max_depth:
            root.items = objlist
            return root

        # the axis of max size
        axis = 0
        bsize = cmm[2]
        if bsize[0] > bsize[1]:
            if bsize[0] > bsize[2]:  # 0 > 1; 0 > 2
                axis = 0
            else:  # 2 > 0; 0 > 1;
                axis = 2
        else:  # 1 > 0
            if bsize[1] > bsize[2]:  # 1 > 0; 1 > 2
                axis = 1
            else:  # 2 > 1; 1 > 0;
                axis = 2

        dst = ti.ndarray(ti.i32, objlen)

        ll = float(cmm[0][axis])
        rr = float(cmm[1][axis])

        mx = self.search_split_line(objlist, axis, ll, rr)
        self._align_kd_box(dst, objlist, axis, ll, mx, rr)

        left = []
        right = []
        mid = []
        for i in range(objlen):
            if dst[i] == 1:
                left.append(objlist[i])
            elif dst[i] == 2:
                right.append(objlist[i])
            elif dst[i] == 3:
                mid.append(objlist[i])

        if len(mid) == objlen:
            # all is in the left
            root.items = objlist
            return root

        root.mid = self._split_kd(mid, depth+1, depth+1)  # max depth = depth
        root.left = self._split_kd(left, depth+1, max_depth)
        root.right = self._split_kd(right, depth+1, max_depth)
        root.lritems = np.asarray(left + right, dtype=np.int32)

        return root

    def search_split_line(self, objlist: np.ndarray, axis: int, ll, rr):
        """
        grid search split line along axis
        """
        best_mx = 0.0
        best_cnt = objlist.shape[0]

        alpha = 0.0

        while alpha < 1.0:
            mx = alpha * ll + (1.0 - alpha) * rr
            cnt = self._kd_box_count_x(objlist, axis, ll, mx, rr)
            if cnt < best_cnt:
                best_mx = mx
                best_cnt = cnt
            alpha += 0.05

        return best_mx

    @ti.kernel
    def _kd_box_count_x(
            self,
            objlist: ti.any_arr(),
            axis: ti.i32,
            left: ti.f32,
            mx: ti.f32,
            right: ti.f32) -> ti.f32:
        lcnt = 0.0
        rcnt = 0.0
        mcnt = 0.0
        for i in objlist:
            # cmin, cmax, center
            obox0 = 0.0
            obox1 = 0.0
            for j in ti.static(range(3)):
                if j == axis:
                    obox0 = self.meshboxes[objlist[i], 0][j]
                    obox1 = self.meshboxes[objlist[i], 1][j]
            hL = max(left, obox0) <= min(obox1, mx)
            hR = max(mx, obox0) <= min(obox1, right)

            if hL and hR:
                mcnt += 1.0
            elif hL:
                lcnt += 1.0
            else:
                rcnt += 1.0

        # just guess
        return abs(lcnt - rcnt) + 0.5 * mcnt


@ti.data_oriented
class TriangleSoup:
    r""" A bunch of Triangles
    ```
       a
      / \
    |/   \
    x --- b

    n (normal vector is up)
    ```

    Require: 
    - vertex: 3d points (x,y,z) or or (x,y,z,s,t) or (x,y,z,nx,ny,nz,s,t)
        - if vertex contains (s,t), texture is enabled
    - faces: the faces
    - matrial: material
    - texture: 
    """

    def __init__(self, vertex, faces, material=M_unknown, texture=None) -> None:
        self.vertex = _ensure_ti_field(vertex)
        self.faces = _ensure_ti_field(faces)

        if texture is not None:
            if isinstance(texture, np.ndarray):
                if texture.dtype == np.uint8:
                    texture = np.asarray(texture, np.float32)
                    texture /= 255.0
                
        self.texture = _ensure_ti_field(texture)
        self.has_texture = texture is not None

        if vertex.shape[1] == 8:
            self.vertex_s = 6
            self.vertex_t = 7
        else:
            if texture is not None:
                warnings.warn("vertex do not contains (s,t), ignore texture")
            texture = None

        self.count = self.faces.shape[0]

        self.meshes = ti.Struct.field({
            "x": ti.types.vector(3, ti.f32),
            "ax": ti.types.vector(3, ti.f32),
            "bx": ti.types.vector(3, ti.f32),
            "n": ti.types.vector(3, ti.f32),
            "idetidx": ti.i32,
            "idet": ti.f32,
        }, self.count)

        self.material = material

        # [kdflat]
        # 0: corner index;
        # 1: mark
        # if mark >= 0:
        #   1: left index
        #   2: right index
        #   3: mid index
        #   4： lr data index
        # else:
        #   2: data len (n)
        #   3+0: triangle index 0
        #   3+1: triangle index 1
        #   ...: .....
        #   3+n-1: triangle index n-1
        #
        # next block
        self.kdflat = ti.field(ti.i32)
        ti.root.pointer(ti.i, int(np.ceil(self.count*self.count / 64)))\
            .dense(ti.i, 64).place(self.kdflat)
        self.kdcmin = ti.Vector.field(3, ti.f32)
        self.kdcmax = ti.Vector.field(3, ti.f32)
        ti.root.pointer(ti.i, int(np.ceil(2*self.count / 64)))\
            .dense(ti.i, 64).place(self.kdcmin, self.kdcmax)
        
        self._init_meshes(self.count)

    @ti.kernel
    def _init_meshes(self, n:int):
        for i in range(n):
            p0, p1, p2 = self.faces[i,0], self.faces[i,1], self.faces[i,2]
            a = ti.Vector([self.vertex[p0,0], self.vertex[p0,1], self.vertex[p0,2]])
            x = ti.Vector([self.vertex[p1,0], self.vertex[p1,1], self.vertex[p1,2]])
            b = ti.Vector([self.vertex[p2,0], self.vertex[p2,1], self.vertex[p2,2]])

            info = _parallelogram_init(a, x, b)
            # ax, bx, normal, area, detidx, idet

            # print("mesh", i, info[2], info[4], info[5])

            self.meshes[i].x = x
            self.meshes[i].ax = info[0]
            self.meshes[i].bx = info[1]
            self.meshes[i].n = info[2]
            self.meshes[i].idetidx = info[4]
            self.meshes[i].idet = info[5]

    def build_tree(self):
        print("start build tree")
        octt = KDTree(self.meshes)
        octt.build_tree()
        octt.flat_tree_to_taichi(self.kdflat, self.kdcmin, self.kdcmax)
        self.dump_tree()
        # print(self.meshes)
        del octt
        print("end build tree")
        # exit(0)

    def dump_tree(self):
        nlist = [(0, 'O')]
        # print(self.kdflat)
        while len(nlist) > 0:
            idx, tag = nlist.pop(-1)
            # cidx = self.kdflat[idx]
            mark = self.kdflat[idx + 1]
            if mark == -1:  # they are triangles
                n = self.kdflat[idx + 2]
                # print("triangle", idx, self.kdcmin[cidx], self.kdcmax[cidx])
                print("triangle", idx, end="")
                print(f" ({tag}) data#{n}", end=" ")
                for i in range(n):
                    print(self.kdflat[idx+3+i], end=" ")
                print("")
            else:
                left = mark
                right = self.kdflat[idx + 2]
                mid = self.kdflat[idx + 3]
                # lrd = self.kdflat[idx + 4]
                # print("[node]", left, right, mid, self.kdcmin[cidx], self.kdcmax[cidx])
                print("[node]", left, right, mid)
                nlist.append((mid, 'M'))
                nlist.append((right, 'R'))
                nlist.append((left, 'L'))

    @ti.func
    def _get_color(self, meshidx, alpha:float, beta:float):
        # idx is the index of triangle in self.meshes
        texshape = ti.Vector([
            float(self.texture.shape[0] - 1), 
            float(self.texture.shape[1] - 1), 
        ]) 
        ans = ti.Vector([1.0, 1.0, 1.0])
        if meshidx >= 0 and ti.static(self.has_texture):
            # have texture map
            # compute i,j
            #    a
            #   /
            #  /
            # x ----- b
            p0, p1, p2 = self.faces[meshidx, 0], self.faces[meshidx, 1], self.faces[meshidx, 2]
            a_st = ti.Vector([self.vertex[p0, self.vertex_s], self.vertex[p0, self.vertex_t]])
            x_st = ti.Vector([self.vertex[p1, self.vertex_s], self.vertex[p1, self.vertex_t]])
            b_st = ti.Vector([self.vertex[p2, self.vertex_s], self.vertex[p2, self.vertex_t]])

            # print("alpha, beta", alpha, beta)

            ax_st = a_st - x_st
            bx_st = b_st - x_st

            pij = (x_st + alpha * ax_st + beta * bx_st) * texshape

            rpij = ti.round(pij)

            pi = int(rpij[0])
            pj = int(rpij[1])
            
            ans = ti.Vector([
                self.texture[pi,pj,0],
                self.texture[pi,pj,1],
                self.texture[pi,pj,2],
            ])
        return ans

    @ti.func
    def _check_ab(self, alpha, beta):
        ans = 1
        if alpha < 0.0 or beta < 0.0 or alpha + beta > 1.0:
            ans = 0
        return ans

    @ti.func
    def _iter_objects(self, ray, idx, time_min, time_max):
        is_hit = 0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0

        hit_meshidx = -1
        hit_alpha = -1.0
        hit_beta = -1.0

        _n = self.kdflat[idx + 2]
        # print("iter", _n)
        for _oi in range(_n):
            meshidx = self.kdflat[idx+3+_oi]
            item = self.meshes[meshidx]
            # print("hit item", item.x)
            rdn = ray.direction.dot(item.n)
            if rdn < 0:
                # ray hit plane
                t = item.n.dot(item.x - ray.origin) / \
                    item.n.dot(ray.direction)

                if t > time_min and t < time_max:
                    p = ray.at(t)

                    alpha, beta = _parallelogram_alpha_beta(
                        p - item.x, item.ax, item.bx, item.idetidx, item.idet)

                    if self._check_ab(alpha, beta):
                        # in the triangle | paralleogram
                        is_hit = 1
                        hit_meshidx = meshidx
                        hit_alpha = alpha
                        hit_beta = beta
                        time_max = t  # !! update
                        hit_point = p
                        if rdn > 0:
                            hit_normal = -1 * item.n
                            is_inside = 1
                        # rdn < 0; outside
                        else:
                            hit_normal = item.n
                            is_inside = 0

        return is_hit, hit_meshidx, hit_alpha, hit_beta, time_max, hit_point, hit_normal, is_inside

    @ti.func
    def ray_intersect(self, ray, time_min: float, time_max: float):
        is_hit = 0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0

        idx = 0
        flag, _ = ray_intersect_solid_box(
            ray, self.kdcmin[0], self.kdcmax[0], time_min, time_max)

        hit_meshidx = -1
        hit_alpha = -1.0
        hit_beta = -1.0

        while flag:
            mark = self.kdflat[idx+1]
            if mark < 0:  # triangles
                flag = 0
                info = self._iter_objects(ray, idx, time_min, time_max)
                if info[0] == 1:# and info[4] > time_min and info[4] < time_max:
                    is_hit = 1
                    hit_meshidx = info[1]
                    hit_alpha = info[2]
                    hit_beta = info[3]
                    time_max = info[4]
                    hit_point = info[5]
                    hit_normal = info[6]
                    is_inside = info[7]
            else:  # objects
                _l = mark
                _r = self.kdflat[idx+2]
                _m = self.kdflat[idx+3]
                _lr = self.kdflat[idx+4]

                _li = self.kdflat[_l]
                _ri = self.kdflat[_r]
                _mi = self.kdflat[_m]

                _hitm, _ = ray_intersect_solid_box(
                    ray, self.kdcmin[_mi], self.kdcmax[_mi], time_min, time_max)
                if _hitm == 1:
                    # iter mid
                    info = self._iter_objects(ray, _m, time_min, time_max)
                    if info[0] == 1:# and info[4] > time_min and info[4] < time_max:
                        is_hit = 1
                        hit_meshidx = info[1]
                        hit_alpha = info[2]
                        hit_beta = info[3]
                        time_max = info[4]
                        hit_point = info[5]
                        hit_normal = info[6]
                        is_inside = info[7]

                _hitl, _ = ray_intersect_solid_box(
                    ray, self.kdcmin[_li], self.kdcmax[_li], time_min, time_max
                )
                _hitr, _ = ray_intersect_solid_box(
                    ray, self.kdcmin[_ri], self.kdcmax[_ri], time_min, time_max
                )

                if _hitl == 1 and _hitr == 1:
                    # we should iter both left and right
                    idx = _lr
                elif _hitl == 1:
                    idx = _l
                elif _hitr == 1:
                    idx = _r
                else:  # not hit
                    flag = 0
        

        color = self._get_color(hit_meshidx, hit_alpha, hit_beta)

        return is_hit, time_max, hit_point, hit_normal, self.material, color, is_inside

    def __str__(self) -> str:
        items = []
        for i in range(self.info[None]):
            fi = self.meshes[i]
            items.append(
                f"x={fi.x}, ax={fi.ax}, bx={fi.bx}, n={fi.n}, material={fi.material}, color={fi.color}")
        items = "\n  ".join(items)
        return f"""{self.__class__.__name__}(\n  {items})"""


@ti.data_oriented
class ParallelogramSoup(TriangleSoup):
    r""" A bunch of Parallelograms
    ```
       a ------
      /       /
    |/       /
    x ----- b

    n (normal vector is up)

    a -> x -> b | and right hand, 
    ```
    All x, a, b are 3-dimensional vectors
    """

    def __init__(self, count, material=M_unknown, color=None) -> None:
        super().__init__(count, material, color)

    # only need to rewrite this function
    @ti.func
    def _check_ab(self, alpha, beta):
        ans = 1
        if alpha < 0.0 or beta < 0.0 or alpha > 1.0 or beta > 1.0:
            ans = 0
        return ans


@ti.data_oriented
class Box(ParallelogramSoup):
    """A Box, that has six Parallelogrom.

    ```
       a ------ f
      /        / |
     /        /  |
    x ------ b   g
    |        |  /
    |        | /
    c ------ d

    a---f
    | 0 |
    x---b---f---a---x
    | 2 | 1 | 3 | 4 |
    c---d---g---e---c
    | 5 |
    e---g
    ```

    - d = x + (c-x) + (b-x) = b + c - x;
    - e = x + (a-x) + (c-x) = a + c - x;
    - f = x + (b-x) + (a-x) = b + a - x;
    - g = c + (d-c) + (e-c) = d + e - c;

    if normal_out:
        all normal vectors are point outside
    else:
        all normal vectors are point inside
    """

    def __init__(self, a, x, b, c, material, color, normal_out=True) -> None:
        super().__init__(6)
        # self.volume = ti.abs((a - x).dot((b-x).cross((c-x))))

        material = _var_as_tuple(material, 6)
        color = _var_as_tuple(color, 6)

        d = b + c - x
        e = a + c - x
        f = b + a - x
        # g = d + e - c

        # print(x, a, b, c, d, e, f, g)
        if normal_out:
            self.append(a, x, b, material[0], color[0])
            self.append(f, b, d, material[1], color[1])
            self.append(b, x, c, material[2], color[2])
            self.append(e, a, f, material[3], color[3])
            self.append(c, x, a, material[4], color[4])
            self.append(d, c, e, material[5], color[5])
        else:
            self.append(b, x, a, material[0], color[0])
            self.append(d, b, f, material[1], color[1])
            self.append(c, x, b, material[2], color[2])
            self.append(f, a, e, material[3], color[3])
            self.append(a, x, c, material[4], color[4])
            self.append(e, c, d, material[5], color[5])

        self.build_tree()

    @staticmethod
    def new(center, size, material, color, normal_out=True):
        halfs = size / 2.0

        x = ti.Vector([
            center[0] + halfs[0],
            center[1] - halfs[1],
            center[2] + halfs[2]])
        a = ti.Vector([
            center[0] - halfs[0],
            center[1] - halfs[1],
            center[2] + halfs[2]])
        b = ti.Vector([
            center[0] + halfs[0],
            center[1] + halfs[1],
            center[2] + halfs[2]])
        c = ti.Vector([
            center[0] + halfs[0],
            center[1] - halfs[1],
            center[2] - halfs[2]])

        return Box(a, x, b, c, material, color, normal_out)


@ti.data_oriented
class SphereSoup:
    def __init__(self, count, material=M_diffuse, color=None) -> None:
        if color is None:
            color = ti.Vector([1.0, 1.0, 1.0])

        self.num_sphere = count

        self.spheres = ti.Struct.field({
            "x": ti.types.vector(3, ti.f32),
            "r": ti.f32,
            "material": ti.i32,
            "color": ti.types.vector(3, ti.f32),
        }, count)

        self.scount = 0
        self.material = material
        self.color = color

    def append(self, x, r, material=None, color=None):
        if self.scount >= self.num_sphere:
            print("too much spere")
            return

        if material is None:
            material = self.material
        if color is None:
            color = self.color

        self.spheres[self.scount].x = x
        self.spheres[self.scount].r = r
        self.spheres[self.scount].material = material
        self.spheres[self.scount].color = color

        self.scount += 1

    @ti.func
    def ray_intersect(self, ray, time_min, time_max):
        is_hit = 0
        hit_time = time_max
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0
        material = 0
        color = self.color

        for idx in range(self.scount):
            item = self.spheres[idx]

            info = _sphere_hit(item.x, item.r, ray, time_min, hit_time)
            # is_hit, hit_time, hit_point, hit_normal, is_inside

            if info[0] == 0 or info[1] > hit_time:
                continue
            is_hit = info[0]
            hit_time = info[1]
            hit_point = info[2]
            hit_normal = info[3]
            is_inside = info[4]
            material = item.material
            color = item.color

        return is_hit, hit_time, hit_point, hit_normal, material, color, is_inside

    def __str__(self) -> str:
        items = []
        for i in range(self.face_count):
            fi = self.faces[i]
            items.append(
                f"x={fi.x}, ax={fi.ax}, bx={fi.bx}, n={fi.n}, material={fi.material}, color={fi.color}")
        items = "\n  ".join(items)
        return f"""{self.__class__.__name__}(\n  {items})"""


@ti.data_oriented
class Scene:
    def __init__(self):
        self.objList = []
        self.objName = []

    def append(self, obj, name=None):
        if name is None:
            name = "obj%d" % len(self.objList)
        # print(name, obj)
        new_name = True
        for i, n in enumerate(self.objName):
            if n != name:
                continue
            print("overwrite", name)
            self.objList[i] = obj
            new_name = False
            break
        if new_name:
            self.objName.append(name)
            self.objList.append(obj)

    def remove_obj(self, name):
        ni = -1
        for i, n in enumerate(self.objName):
            if n != name:
                continue
            ni = i
            break
        if ni != -1:
            del self.objList[ni]
            del self.objName[ni]

    def clear_objs(self):
        self.objList.clear()
        self.objName.clear()

    @ti.func
    def ray_intersect(self, ray, time_min=1e-8, time_max=1e+8):
        is_hit = 0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0
        material = 0
        color = ti.Vector([0.0, 0.0, 0.0])

        for i in ti.static(range(len(self.objList))):
            info = self.objList[i].ray_intersect(ray, time_min, time_max)
            # print("hit obj", i, info, time_max)
            if info[0] == 1 and info[1] > time_min and info[1] < time_max:
                is_hit = 1
                time_max = info[1]
                hit_point = info[2]
                hit_normal = info[3]
                material = info[4]
                color = info[5]
                is_inside = info[6]

        return is_hit, time_max, hit_point, hit_normal, material, color, is_inside


@ti.data_oriented
class Camera:
    # codes are copied from course examples
    def __init__(self, fov=60, aspect_ratio=1.0) -> None:
        # Camera parameters
        self.lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vup = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fov = fov
        self.aspect_ratio = aspect_ratio

        self.cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())

        self.set_look_from(7.0, 2.0, 2.0)
        self.set_look_at(-1.0, 1.0, -0.7)
        self.set_vup(0.0, 0.0, 1.0)

        self.reset()

    def set_look_from(self, x: float, y: float, z: float):
        self.lookfrom[None] = [x, y, z]

    def look_at(self):
        a = self.lookat[None]
        return a[0], a[1], a[2]

    def look_from(self):
        a = self.lookfrom[None]
        return a[0], a[1], a[2]

    def set_look_at(self, x: float, y: float, z: float):
        self.lookat[None] = [x, y, z]

    def set_vup(self, x: float, y: float, z: float):
        vup = ti.Vector([x, y, z])
        self.vup[None] = vup.normalized()

    @ti.kernel
    def reset(self):
        theta = self.fov * (math.pi / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        self.cam_origin[None] = self.lookfrom[None]
        w = (self.lookfrom[None] - self.lookat[None]).normalized()
        u = (self.vup[None].cross(w)).normalized()
        v = w.cross(u)
        self.cam_lower_left_corner[None] = ti.Vector(
            [-half_width, -half_height, -1.0])
        self.cam_lower_left_corner[
            None] = self.cam_origin[None] - half_width * u - half_height * v - w
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    @ti.func
    def get_ray(self, u, v):
        return Ray(self.cam_origin[None], self.cam_lower_left_corner[None] + u * self.cam_horizontal[None] + v * self.cam_vertical[None] - self.cam_origin[None])


@ti.kernel
def render(camera: ti.template(), canvas: ti.template(), scene: ti.template(), samples_per_pixel: float, max_depth: int):
    image_width, image_height = canvas.shape
    # print(image_width, image_height)
    for i, j in canvas:
        cc = ti.Vector([0.0, 0.0, 0.0])
        u = (float(i) + ti.random()) / image_width
        v = (float(j) + ti.random()) / image_height
        ray = camera.get_ray(u, v)
        for _ in range(samples_per_pixel):
            cc += ray_color(ray, scene, max_depth)
        canvas[i, j] += cc / samples_per_pixel

# return info
# 0,      1,        2,         3,          4,        5,     6
# is_hit, hit_time, hit_point, hit_normal, material, color, is_inside


@ti.func
def ray_color(ray, scene, max_depth: int):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.9
    for n in range(max_depth):
        if ti.random() > p_RR:
            # print("break p_RR, depth", n)
            break
        info = scene.ray_intersect(Ray(scattered_origin, scattered_direction))
        is_hit, hit_time, hit_point, hit_normal, material, color, is_inside = info
        if is_hit == 0:
            break
        if material == M_light_source:
            color_buffer = color * brightness
            break
        if material == M_diffuse:
            target = hit_point + hit_normal + random_unit_vector()
            scattered_direction = target - hit_point
            scattered_origin = hit_point
            brightness *= color
        elif material == M_metal or material == M_fuzzy_metal:
            scattered_origin = hit_point
            scattered_direction = reflect(
                scattered_direction.normalized(), hit_normal)

            if material == M_fuzzy_metal:
                scattered_direction += 0.4 * random_unit_vector()

            # do not check normal vector
            brightness *= color
        elif material == M_dielectric:
            refraction_ratio = 1.5
            if is_inside == 0:
                refraction_ratio = 1.0 / 1.5

            scattered_direction = scattered_direction.normalized()
            cos_theta = min(-scattered_direction.dot(hit_point), 1.0)
            sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
            # total internal reflection
            if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                scattered_direction = reflect(
                    scattered_direction, hit_normal)
            else:
                scattered_direction = refract(
                    scattered_direction, hit_normal, refraction_ratio)
            scattered_origin = hit_point
            brightness *= color

        brightness /= p_RR

    return color_buffer
