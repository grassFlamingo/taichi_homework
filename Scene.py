from posixpath import join
import taichi as ti
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
    def hit(self, ray):
        """ray sphere intersection.
        refernece:
        Fundamentals of Computer Graphics 4-th Edition, Chapter 4, section 4.4.1; page 76
        """
        is_hit = 0
        hit_time = HIT_TIME_MAX
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0

        # the direction of ray is normed to 1
        oc = ray.origin - self.center

        doc = ray.direction.dot(oc)

        t = doc * doc - (oc.dot(oc) - self.radius * self.radius)
        # t < 0: no intersection
        if t >= 0:  # have intersection
            _t = ti.sqrt(t)
            # - doc +/- sqrt(t)
            t1 = -doc + _t
            t2 = -doc - _t

            if t1 < HIT_TIME_MIN:
                if t2 > HIT_TIME_MIN:
                    # ray is inside this sphere
                    # ( <-- t1 - O-> --- t2 ---> )
                    is_hit = 1
                    hit_time = t2
                    is_inside = 1
                # else:  # no solution
                #     # ( <--t2-- ) <-- t1 -- O->
                #     print("no solution", t1, t2)
            else:
                if t2 > HIT_TIME_MIN:
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
                hit_normal = (self.center - hit_point) / self.radius
            else:
                hit_normal = (hit_point - self.center) / self.radius


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
    def hit(self, ray):
        is_hit = 0
        hit_time = HIT_TIME_MAX
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

            is_hit = 1
            hit_normal = self.normal
            hit_point = ray.at(hit_time)

        return is_hit, hit_time, hit_point, hit_normal, self.material, self.color, is_inside

    def __str__(self) -> str:
        return "Plane()"


@ti.data_oriented
class Parallelogram:
    """
    Parallelogram

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

    def __init__(self, a, x, b, material, color):
        self.x = x
        ax = a - x
        bx = b - x
        self.normal = bx.cross(ax)
        self.area = self.normal.norm()  # what is the norm is 0?
        assert self.area >= 1e-8, "not a parallelogram"
        self.normal = self.normal / self.area

        self.ax = ax
        self.bx = bx

        # | i     j     k     |
        # | ax[0] ax[1] ax[2] |
        # | bx[0] bx[1] bx[2] |
        # try to compute its 2D determinant

        det = ti.abs(ax[1] * bx[2] - ax[2] * bx[1])
        detidx = 0
        if det < 1e-8:
            # previous det is too small; try another one
            det = ti.abs(ax[0] * bx[2] - ax[2] * bx[0])
            detidx = 1

        if det < 1e-8:
            # previous dets are too small; try another one
            det = ti.abs(ax[0] * bx[1] - ax[1] * bx[0])
            detidx = 2

        assert(det > 1e-8)  # "this could not happen, just for case"

        self.det = det
        self.detidx = detidx

        self.material = material
        self.color = color

    @ti.func
    def hit(self, ray):
        is_hit = 0
        hit_time = HIT_TIME_MAX
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0

        # > 0 same direction; never hit
        # = 0 orthogonal; never hit
        if ray.direction.dot(self.normal) < 0:
            # ray hit the plane
            t = self.normal.dot(self.x - ray.origin) / \
                self.normal.dot(ray.direction)

            if t > 0:
                p = ray.at(t)
                if self._is_contains(p) == 1:
                    is_hit = 1
                    hit_time = t
                    hit_point = p
                    hit_normal = self.normal

        return is_hit, hit_time, hit_point, hit_normal, self.material, self.color, is_inside

    @ti.func
    def _is_contains(self, point) -> bool:
        """
        check point is in parallelogram; 
        please convinced that this point is in the plane of this parallelogram.
        p = x + alpha (a-x) + beta (b-x)
        p - x = alpha (a-x) + beta (b-x)

        px[0] = alpha ax[0] + beta bx[0]
        px[1] = alpha ax[1] + beta bx[1]
        px[2] = alpha ax[2] + beta bx[2]

        detA0 = ax[1] bx[2] - ax[2] bx[1]
        detA1 = -1 (ax[0] bx[2] - ax[2] bx[0])
        detA2 = ax[0] bx[1] - ax[1] bx[0]

        if idetidx == 0:
            alpha = 1/detA (px[0] ax[1] + px[1] ax[2])
            beta  = 1/detA (px[0] bx[1] + px[1] bx[2])
        if idetidx == 1:
            alpha = -1/detA (px[0] ax[0] + px[1] ax[2])
            beta  = -1/detA (px[0] bx[0] + px[1] bx[2])
        if idetidx == 2:
            alpha = 1/detA (px[0] ax[0] + px[1] ax[1])
            beta  = 1/detA (px[0] bx[0] + px[1] bx[1])
        """

        px = point - self.x
        ax, bx = self.ax, self.bx

        alpha, beta = -1.0, -1.0

        if self.detidx == 0:
            alpha = px[1] * ax[1] + px[2] * ax[2]
            beta = px[1] * bx[1] + px[2] * bx[2]
        elif self.detidx == 1:
            alpha = px[0] * ax[0] + px[2] * ax[2]
            beta = px[0] * bx[0] + px[2] * bx[2]
            # alpha *= -1
            # beta *= -1
        else:  # 2
            alpha = px[0] * ax[0] + px[1] * ax[1]
            beta = px[0] * bx[0] + px[1] * bx[1]

        return self._check_ab(alpha, beta)

    @ti.func
    def _check_ab(self, alpha: float, beta: float):
        ans = 1
        if alpha < 0 or alpha > self.det or beta < 0 or beta > self.det:
            ans = 0
        return ans

    def __str__(self):
        return f"Parallelogram(a={self.ax + self.x}, x={self.x}, b={self.bx + self.x}, normal={self.normal}, det={self.det:.4f}, detidx={self.detidx}"


@ti.data_oriented
class Triangle(Parallelogram):
    r"""Triangle

    ```
       a
      / \
    |/   \
    x --- b

    n (normal vector is up)
    ```
    """

    def __init__(self, a, x, b, material, color):
        super().__init__(a, x, b, material, color)

    @ti.func
    def _check_ab(self, alpha: float, beta: float) -> bool:
        # only need to rewrite this
        ans = 1
        if alpha < 0 or alpha > self.det or beta < 0 or beta > self.det - alpha:
            ans = 0
        return ans

    def __str__(self) -> str:
        return f"Trangle(a={self.ax + self.x}, x={self.x}, b={self.bx + self.x}, normal={self.normal}, det={self.det:.4f}, detidx={self.detidx:.4f})"


@ti.data_oriented
class Box:
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

    def __init__(self, x, a, b, c, material, color, normal_out=True) -> None:
        self.x = x
        self.material = _var_as_tuple(material, 6)
        self.color = _var_as_tuple(color, 6)

        self.ax = a - x
        self.bx = b - x
        self.cx = c - x

        d = b + c - x
        e = a + c - x
        f = b + a - x
        # g = d + e - c

        # print(x, a, b, c, d, e, f, g)

        faces = []
        if normal_out:
            faces.append(
                Parallelogram(a, x, b, self.material[0], self.color[0]))
            faces.append(
                Parallelogram(f, b, d, self.material[1], self.color[1]))
            faces.append(
                Parallelogram(b, x, c, self.material[2], self.color[2]))
            faces.append(
                Parallelogram(e, a, f, self.material[3], self.color[3]))
            faces.append(
                Parallelogram(c, x, a, self.material[4], self.color[4]))
            faces.append(
                Parallelogram(d, c, e, self.material[5], self.color[5]))
        else:
            faces.append(
                Parallelogram(b, x, a, self.material[0], self.color[0]))
            faces.append(
                Parallelogram(d, b, f, self.material[1], self.color[1]))
            faces.append(
                Parallelogram(c, x, b, self.material[2], self.color[2]))
            faces.append(
                Parallelogram(f, a, e, self.material[3], self.color[3]))
            faces.append(
                Parallelogram(a, x, c, self.material[4], self.color[4]))
            faces.append(
                Parallelogram(e, c, d, self.material[5], self.color[5]))

        self.faces = faces
        self.volume = ti.abs(self.ax.dot(self.bx.cross(self.cx)))

    @staticmethod
    def newbox(center, size, material, color, normal_out=True):
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

        return Box(x, a, b, c, material, color, normal_out)

    @ti.func
    def hit(self, ray):
        is_hit = 0
        hit_time = HIT_TIME_MAX
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0
        material = 0
        color = self.color[0]

        for i in ti.static(range(6)):
            info = self.faces[i].hit(ray)
            if info[0] == 1:
                if info[1] > HIT_TIME_MIN:
                    is_hit = 1
                    if hit_time > info[1]:
                        hit_time = info[1]
                        hit_point = info[2]
                        hit_normal = info[3]
                        material = info[4]
                        color = info[5]
                        is_inside = info[6]

        return is_hit, hit_time, hit_point, hit_normal, material, color, is_inside

    def __str__(self):
        face_str = ',\n'.join(["  " + str(face) for face in self.faces])
        return f"Box(\n{face_str}\n)"


@ti.data_oriented
class Scene:
    def __init__(self):
        self.objList = []
        self.objName = []

    def add_obj(self, obj, name=None):
        if name is None:
            name = "obj%d" % len(self.objList)
        print(name, obj)
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
    def hit(self, ray):
        is_hit = 0
        hit_time = HIT_TIME_MAX
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_normal = ti.Vector([0.0, 1.0, 0.0])
        is_inside = 0
        material = 0
        color = ti.Vector([0.0, 0.0, 0.0])

        for idx in ti.static(range(len(self.objList))):
            info = self.objList[idx].hit(ray)
            if info[0] == 1:
                # print("hit obj", idx, info)
                if info[1] > HIT_TIME_MIN:
                    is_hit = 1
                    if hit_time > info[1]:
                        hit_time = info[1]
                        hit_point = info[2]
                        hit_normal = info[3]
                        material = info[4]
                        color = info[5]
                        is_inside = info[6]

        return is_hit, hit_time, hit_point, hit_normal, material, color, is_inside

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
    
    def set_look_from(self, x:float, y:float, z:float):
        self.lookfrom[None] = [x, y, z]

    def look_at(self):
        a = self.lookat[None]
        return a[0], a[1], a[2]
    
    def look_from(self):
        a = self.lookfrom[None]
        return a[0], a[1], a[2]
    
    def set_look_at(self, x:float, y:float, z:float):
        self.lookat[None] = [x, y, z]

    def set_vup(self, x:float, y:float, z:float):
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
