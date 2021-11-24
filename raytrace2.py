import itertools
import taichi as ti
import Scene
import numpy as np

from plyreader import PLYReader

# ti.init(excepthook=True)
# ti.init(debug=True, excepthook=True, kernel_profiler=True)
ti.init()

# Canvas
aspect_ratio = 1.0
image_width = 512
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# Rendering parameters
samples_per_pixel = 8
max_depth = 64


@ti.kernel
def render_01(camera: ti.template()):
    # for i, j in itertools.product(range(image_width, image_height)):
    for i, j in canvas:
        u = (float(i) + ti.random()) / image_width
        v = (float(j) + ti.random()) / image_height
        ray = camera.get_ray(u, v)
        color = ray_color_01(ray)
        canvas[i, j] += color

@ti.func
def ray_color_01(ray: ti.template()):
    default_color = ti.Vector([0.0, 0.0, 0.0])
    info = scene.hit(ray)
    if info[0] == 1:
        default_color = info[5]
    return default_color


@ti.kernel
def render(camera: ti.template()):
    for i, j in canvas:
        cc = ti.Vector([0.0,0.0,0.0])
        u = (float(i) + ti.random()) / image_width
        v = (float(j) + ti.random()) / image_height
        ray = camera.get_ray(u, v)
        for _ in range(samples_per_pixel):
            cc += ray_color(ray)
        canvas[i, j] += cc / samples_per_pixel 

# return info
# 0,      1,        2,         3,          4,        5,     6
# is_hit, hit_time, hit_point, hit_normal, material, color, is_inside


@ti.func
def ray_color(ray):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.9
    for n in range(max_depth):
        if ti.random() > p_RR:
            # print("break p_RR, depth", n)
            break
        info = scene.hit(Scene.Ray(scattered_origin, scattered_direction))
        is_hit, hit_time, hit_point, hit_normal, material, color, is_inside = info
        if is_hit == 0:
            break
        if material == Scene.M_light_source:
            color_buffer = color * brightness
            break
        if material == Scene.M_diffuse:
            target = hit_point + hit_normal + Scene.random_unit_vector()
            scattered_direction = target - hit_point
            scattered_origin = hit_point
            brightness *= color
        elif material == Scene.M_metal or material == Scene.M_fuzzy_metal:
            scattered_origin = hit_point
            scattered_direction = Scene.reflect(scattered_direction.normalized(), hit_normal)

            if material == Scene.M_fuzzy_metal:
                scattered_direction += 0.4 * Scene.random_unit_vector()

            # do not check normal vector
            brightness *= color
        elif material == Scene.M_dielectric:
            refraction_ratio = 1.5
            if is_inside == 0:
                refraction_ratio = 1.0 / 1.5

            scattered_direction = scattered_direction.normalized()
            cos_theta = min(-scattered_direction.dot(hit_point), 1.0)
            sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
            # total internal reflection
            if refraction_ratio * sin_theta > 1.0 or Scene.reflectance(cos_theta, refraction_ratio) > ti.random():
                scattered_direction = Scene.reflect(
                    scattered_direction, hit_normal)
            else:
                scattered_direction = Scene.refract(
                    scattered_direction, hit_normal, refraction_ratio)
            scattered_origin = hit_point
            brightness *= color

        brightness /= p_RR

    return color_buffer


scene = Scene.Scene()

scene.append(
    Scene.Box.new(
        ti.Vector([0, 5, 2]),
        ti.Vector([5, 0.2, 3]),
        Scene.M_metal,
        ti.Vector([0.2, 0.8, 0.5]),
    )
)

scene.append(
    Scene.Sphere(
        ti.Vector([1, -1, 7]),
        1.0,
        Scene.M_light_source,
        ti.Vector([10.0, 10.0, 10.0])
    )
)

# ssoup = Scene.SphereSoup(8*8*8)

# for i, j, k in itertools.product(range(8), range(8), range(8)):
#     ssoup.append(ti.Vector([i-4, j-4, k-4]), 0.1, Scene.M_metal,
#                  ti.Vector(np.random.rand(3) * 0.8 + 0.1))

# scene.append(ssoup, "sphere soup")

ply = PLYReader()
ply.read_ply("./hidden/tree.ply")
# ply.read_ply("./hidden/teamugblend.ply")

soup = Scene.TriangleSoup(ply.num_triangles, Scene.M_diffuse)

cweig = np.array([0.5, 0.9, 0.6])

for a, b, c, rgba in ply.face_iter():
    # print(a, b, c, rgba)
    soup.append(
        ti.Vector(a),
        ti.Vector(b),
        ti.Vector(c),
        # color=ti.Vector([rgba[0], rgba[1], rgba[2]]))
        color=ti.Vector(np.random.rand(3) * cweig))

scene.append(soup, "tree")

canvas.fill(0)

camera = Scene.Camera()
camera.set_look_from(10.0, 0.0, 8.0)
camera.set_look_at(0.0, 0.0, 1.5)
camera.reset()

gui = ti.GUI("ray tracing", res=(image_width, image_height))
# gui.fps_limit = 3
delta = 0.5
cnt = 0
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        ax, ay, az = camera.look_at()
        fx, fy, fz = camera.look_from()

        if not e.key in "jkliopasdqwe":
            continue

        if e.key == 'j':  # look at x +
            camera.set_look_at(ax-delta, ay, az)
        elif e.key == 'k':  # looa at y +
            camera.set_look_at(ax, ay-delta, az)
        elif e.key == 'l':  # looa at z +
            camera.set_look_at(ax, ay, az-delta)
        elif e.key == 'i':  # look at x -
            camera.set_look_at(ax+delta, ay, az)
        elif e.key == 'o':  # looa at y -
            camera.set_look_at(ax, ay+delta, az)
        elif e.key == 'p':  # looa at z -
            camera.set_look_at(ax, ay, az+delta)
        elif e.key == 'a':  # looa from x +
            camera.set_look_from(fx-delta, fy, fz)
        elif e.key == 's':  # looa from y +
            camera.set_look_from(fx, fy-delta, fz)
        elif e.key == 'd':  # looa from z +
            camera.set_look_from(fx, fy, fz-delta)
        elif e.key == 'q':  # looa from x -
            camera.set_look_from(fx+delta, fy, fz)
        elif e.key == 'w':  # looa from y -
            camera.set_look_from(fx, fy+delta, fz)
        elif e.key == 'e':  # looa from z -
            camera.set_look_from(fx, fy, fz+delta)
        camera.reset()
        canvas.fill(0)  # buggy
        cnt = 0.0
        print("look form", camera.look_from(), "look at", camera.look_at())

    cnt += 1
    render(camera)
    gui.set_image(np.sqrt(canvas.to_numpy() / cnt))

    if cnt % 20 == 0:
        gui.show("imgs/raytrace-%d.jpg" % cnt)
    else:
        gui.show()
