import imageio
import taichi as ti
import Scene
import numpy as np
import itertools

from plyread2 import NPLYReader

# ti.init(excepthook=True)
# ti.init(debug=True, kernel_profiler=True)
ti.init()

# Canvas
aspect_ratio = 1.0
image_width = 512
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# Rendering parameters
samples_per_pixel = 4
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


colors_table = ti.Vector.field(3, ti.f32, 8)


@ti.kernel
def init_color_table():
    for i in range(8):
        colors_table[i] = Scene.rand3()


init_color_table()


@ti.func
def ray_color_01(ray: ti.template()):
    default_color = ti.Vector([0.0, 0.0, 0.0])
    info = scene.ray_intersect(ray)
    # is_hit = 0
    # for i in ti.static(range(8)):
    #     ci = soup.octree[0].child[i]
    #     if is_hit == 0:
    #         if ci > 0:
    #             is_hit, _ = Scene.ray_intersect_solid_box(ray, soup.octree[ci].cmin, soup.octree[ci].cmax, 1e-8, 1e+8)
    #             # print(is_hit, soup.octree[ci].cmin, soup.octree[ci].cmax)
    #             default_color = colors_table[i]

    if info[0] == 1:
        default_color = info[5]
        # print("hit time", info[1])
    return default_color


scene = Scene.Scene()

scene.append(
    Scene.Box.new(
        ti.Vector([5, 5, 2]),
        ti.Vector([5, 0.2, 3]),
        Scene.M_metal,
        color = ti.Vector([0.2, 0.8, 0.5]),
    )
)

scene.append(
    Scene.Sphere(
        ti.Vector([0, 0, 7]),
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
ply = NPLYReader("./hidden/cristal.ply")
# ply.read_ply("./hidden/potted_plant_01_4k_bin.ply")
# ply.read_ply("./hidden/tree2.ply")
# ply = NPLYReader("./hidden/tree2.ply")
# ply.read_ply("./hidden/rock.ply")

soup = Scene.TriangleSoup(
    ply.vertex, 
    ply.faces, 
    Scene.M_diffuse,
    imageio.imread("./hidden/test-small.jpeg")  
)

# exit(0)


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
    render_01(camera)
    # Scene.render(camera, canvas, scene, samples_per_pixel, max_depth)
    gui.set_image(np.sqrt(canvas.to_numpy() / cnt))

    gui.show()
