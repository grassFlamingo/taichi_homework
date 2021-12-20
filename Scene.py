from numpy.core.defchararray import index
import taichi as ti
import numpy as np
import math

HIT_TIME_MIN = 1e-4
HIT_TIME_MAX = 1e+9


def _var_as_tuple(x, length):
    if isinstance(x, (tuple, list)):
        if len(x) >= length:
            return tuple([x[i] for i in range(length)])
        else:
            return tuple([x for _ in range(length)])
    else:
        return tuple([x for _ in range(length)])

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
    normal = normal / area

    # | i     j     k     |
    # | ax[0] ax[1] ax[2] |
    # | bx[0] bx[1] bx[2] |
    # try to compute its 2D determinant

    det = ax[1] * bx[2] - ax[2] * bx[1]
    detidx = 0
    if ti.abs(det) < 1e-8:
        # previous det is too small; try another one
        det = ax[0] * bx[2] - ax[2] * bx[0]
        detidx = 1

    if ti.abs(det) < 1e-8:
        # previous dets are too small; try another one
        det = ax[0] * bx[1] - ax[1] * bx[0]
        detidx = 2

    # assert(ti.abs(det) > 1e-8)  # "this could not happen, just for case"

    return ax, bx, normal, area, detidx, 1.0 / det


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


@ti.data_oriented
class KDTreeNode:
    def __init__(self, max_item_len) -> None:
        self.item_len = ti.field(ti.i32, 2)
        self.item_len[0] = max_item_len
        self.item_len[1] = 0
        self.items = ti.field(ti.i32)
        self.corner = ti.Vector.field(3, ti.f32, (2))
        self.left = None
        self.right = None
        if max_item_len > 64:
            ti.root.pointer(ti.i, (max_item_len+64) // 64).dense(ti.i, 64).place(self.items)
        else:
            ti.root.dense(ti.i, max_item_len).place(self.items)
    
    def set_items(self, items):
        self.item_len[1] = items.shape[0]
        assert self.item_len[1] <= self.item_len[0]
        self._copy_item(items.shape[0], items)
    
    @ti.kernel
    def _copy_item(self, n:int, arr:ti.any_arr()):
        for i in range(n):
            self.items[i] = arr[i]
        
    def set_cmin(self, src):
        self.corner[0][0] = src[0]
        self.corner[0][1] = src[1]
        self.corner[0][2] = src[2]

    def set_cmax(self, src):
        self.corner[1][0] = src[0]
        self.corner[1][1] = src[1]
        self.corner[1][2] = src[2]

@ti.data_oriented
class KDTree:
    """KD Tree; only works for TriangleSoup and ParallelogramSoup
    """

    def __init__(self, count, faces) -> None:
        self.count = count
        self.objs = faces
        self.root = None
        # cmin, cmax, center
        self.objboxes = ti.Vector.field(3, ti.f32, (count, 3))
        self._compute_obj_boxes()

    def build_tree(self, root):
        if self.count <= 0:
            return
        self._split_kd(root, [i for i in range(self.count)])

    @ti.kernel
    def _compute_obj_boxes(self):
        # num faces is self.count <- static alert
        for i in range(self.count):
            di = self.objs[i]
            x = di.x
            a = di.ax + x  # ax = a - x
            b = di.bx + x  # bx = b - x

            for j in ti.static(range(3)):
                self.objboxes[i, 0][j] = min(x[j], a[j], b[j])
                self.objboxes[i, 1][j] = max(x[j], a[j], b[j])

            self.objboxes[i, 2] = 0.5 * \
                (self.objboxes[i, 0] + self.objboxes[i, 1])

    @ti.kernel
    def _compute_box(self, cmm: ti.any_arr(), indexes: ti.any_arr()):
        # init
        for j in ti.static(range(3)):
            cmm[0, j] = self.objs[indexes[0]].x[j]
            cmm[1, j] = self.objs[indexes[0]].x[j]
            cmm[2, j] = 0.0
            cmm[3, j] = 0.0

        # cmm: cmin, cmax, E(x), V(x)

        for i in indexes:
            ii = indexes[i]
            di = self.objs[ii]
            x = di.x
            a = di.ax + x  # ax = a - x
            b = di.bx + x  # bx - b - x

            for j in ti.static(range(3)):
                _t = self.objboxes[ii, 2][j]
                cmm[2, j] += _t
                cmm[3, j] += _t * _t
                ti.atomic_min(cmm[0, j], min(x[j], a[j], b[j]))
                ti.atomic_max(cmm[1, j], max(x[j], a[j], b[j]))

        for j in ti.static(range(3)):
            cmm[2, j] /= indexes.shape[0]  # E(X)
            cmm[3, j] /= indexes.shape[0]  # E(X^2)
            # E(x^2) - E^2(X) = V(X)
            cmm[3, j] = cmm[3, j] - cmm[2, j] * cmm[2, j]

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
        # axis = ti.static(vaxis)

        for i in objlist:
            # cmin, cmax, center
            obox0 = 0.0
            obox1 = 0.0
            for j in ti.static(range(3)):
                if j == axis:
                    obox0 = self.objboxes[objlist[i], 0][j]
                    obox1 = self.objboxes[objlist[i], 1][j]
            dst[i] = 0
            if max(left, obox0) <= min(obox1, mx):
                dst[i] += 1
            if max(mx, obox0) <= min(obox1, right):
                dst[i] += 2

    def _split_kd(self, root, objlist: list):
        # print("_split_oct", left, right)
        # (left, right]

        objlen = len(objlist)
        objlist = np.asarray(objlist, dtype=np.int32)

        if len(objlist) == 0:
            return None

        # cmm: cmin, cmax, E(x), V(x)
        cmm = ti.ndarray(ti.f32, shape=(4, 3))
        self._compute_box(cmm, objlist)
        root.set_cmin(cmm[0])
        root.set_cmax(cmm[1])

        if objlen < 8:
            root.set_items(objlist)
            return

        var = cmm[3]
        if var[0] > var[1]:
            if var[0] > var[2]:
                vaxis = 0
            else:  # 2 > 0; 0 > 1;
                vaxis = 2
        else:
            if var[1] > var[2]:  # 1 > 2; 1 > 0
                vaxis = 1
            else:  # 2 > 1; 1 > 0
                vaxis = 2

        # cmin, cmax, center
        centerList = [self.objboxes[o, 2][vaxis] for o in objlist]
        centerList.sort()

        mx = centerList[objlen//2]

        dst = ti.ndarray(ti.i32, objlen)

        self._align_kd_box(
            dst,
            objlist,
            vaxis,
            float(cmm[0, vaxis]),
            mx,
            float(cmm[1, vaxis]),
        )

        left = []
        right = []
        for i in range(objlen):
            if dst[i] == 1:
                left.append(objlist[i])
            elif dst[i] == 2:
                right.append(objlist[i])
            elif dst[i] == 3:
                left.append(objlist[i])
                right.append(objlist[i])

        if len(left) == objlen or len(right) == objlen:
            # all is in the left
            root.set_items(objlist)
            return

        root.left = KDTreeNode(len(left))
        self._split_kd(root.left, left)

        root.right = KDTreeNode(len(right))
        self._split_kd(root.right, right)


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
    """

    def __init__(self, count, material=M_unknown, color=None) -> None:
        self.faces = ti.Struct.field({
            "x": ti.types.vector(3, ti.f32),
            "ax": ti.types.vector(3, ti.f32),
            "bx": ti.types.vector(3, ti.f32),
            "n": ti.types.vector(3, ti.f32),
            "idetidx": ti.i32,
            "idet": ti.f32,
            "material": ti.i32,
            "color": ti.types.vector(3, ti.f32),
        }, count)

        self.material = material
        if color is None:
            self.color = ti.Vector([1.0, 1.0, 1.0])
        else:
            self.color = color

        # num faces, num nodes
        self.counts = ti.field(ti.i32, (2))
        self.counts[0] = 0
        self.kdtree = KDTreeNode(count)

    def append(self, a, x, b, material=None, color=None):
        if material is None:
            material = self.material
        if color is None:
            color = self.color

        info = _parallelogram_init(a, x, b)

        i = self.counts[0]

        if info[3] > 1e-12:
            # < 1e-12 not a triangle
            self.faces[i].x = x
            self.faces[i].ax = info[0]
            self.faces[i].bx = info[1]
            self.faces[i].n = info[2]
            self.faces[i].idetidx = info[4]
            self.faces[i].idet = info[5]
            self.faces[i].material = material
            self.faces[i].color = color

            self.counts[0] += 1

    def build_tree(self):
        print("start build tree")
        octt = KDTree(self.counts[0], self.faces)
        octt.build_tree(self.kdtree)
        self.dump_tree(self.kdtree)
        del octt
        print("end build tree")
        # exit(0)

    def dump_tree(self, root):
        if root is None:
            return

        print("node:")
        if root.item_len[1] > 0:
            print(root.items)
        else:
            self.dump_tree(root.left)
            self.dump_tree(root.right)

    @ti.func
    def _check_ab(self, alpha, beta):
        ans = 1
        if alpha < 0.0 or beta < 0.0 or alpha + beta > 1.0:
            ans = 0
        return ans

    @ti.func
    def ray_intersect(self, ray, time_min: float, time_max: float):
        is_hit = 0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0
        material = 0
        color = self.color

        root = self.kdtree
        flag, _ = ray_intersect_solid_box(
            ray, root.corner[0], root.corner[1], time_min, time_max)

        while flag:
            if root.item_len[1] > 0: # tirangles
                for rii in range(root.item_len[1]):
                    item = self.faces[root.items[rii]]
                    # print("hit item", item.x)
                    if ray.direction.dot(item.n) < 0:
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
                                time_max = t  # !! update
                                hit_point = p
                                hit_normal = item.n
                                material = item.material
                                color = item.color
            else:
                tleft = root.left
                tright = root.right
                _hit0, _t0 = ray_intersect_solid_box(
                    ray, tleft.corner[0], tleft.corner[1], time_min, time_max
                )
                _hit1, _t1 = ray_intersect_solid_box(
                    ray, tright.corner[0], tright.corner[1], time_min, time_max
                )

                if _hit0 == 1 and _t0 < _t1: # hit left
                    root = tleft
                elif _hit1 == 1 and _t1 < _t0: # hit right
                    root = tright
                else:
                    flag = 0

        return is_hit, time_max, hit_point, hit_normal, material, color, is_inside

    @ti.func
    def ray_intersect_old(self, ray, time_min: float, time_max: float):
        is_hit = 0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0
        material = 0
        color = self.color

        for i in range(self.counts[None]):
            item = self.faces[i]
            if ray.direction.dot(item.n) < 0:
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
                        time_max = t  # !! update
                        hit_point = p
                        hit_normal = item.n
                        material = item.material
                        color = item.color

        return is_hit, time_max, hit_point, hit_normal, material, color, is_inside

    def __str__(self) -> str:
        items = []
        for i in range(self.counts[None]):
            fi = self.faces[i]
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
            if info[0] == 1:
                if info[1] > time_min:
                    is_hit = 1
                    if time_max > info[1]:
                        time_max = info[1]
                        hit_point = info[2]
                        hit_normal = info[3]
                        material = info[4]
                        color = info[5]
                        is_inside = info[6]

        return is_hit, time_max, hit_point, hit_normal, material, color, is_inside

# codes are copied from course examples


@ti.data_oriented
class Camera:
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
