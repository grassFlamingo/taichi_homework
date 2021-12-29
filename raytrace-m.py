import taichi as ti
import Scene
import numpy as np
import PIL.Image

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

@ti.func
def ray_color_01(ray: ti.template()):
    default_color = ti.Vector([0.0, 0.0, 0.0])
    info = scene.ray_intersect(ray)
    if info[0] == 1:
        default_color = info[5]
        # print("hit time", info[1])
    return default_color

scene = Scene.Scene()

scene.append(
    Scene.Sphere(
        ti.Vector([-2, 2, 8]),
        2.0,
        Scene.M_light_source,
        ti.Vector([10.0, 10.0, 10.0])
    )
)

# scene.append(ssoup, "sphere soup")
ply_main = NPLYReader("./mesh/main-main.ply")
ply_crystal = NPLYReader("./mesh/main-crystal.ply")
texture_img = np.asarray(PIL.Image.open(
    "./mesh/main-2-big.png").rotate(-90), dtype=np.float32) / 255.0

texture = Scene._ensure_ti_field(texture_img)

scene.append(Scene.TriangleSoup(
    ply_main.vertex,
    ply_main.faces,
    Scene.M_diffuse,
    texture
))

scene.append(Scene.TriangleSoup(
    ply_crystal.vertex,
    ply_crystal.faces,
    Scene.M_dielectric,
    texture
))
canvas.fill(0)

camera = Scene.Camera()
camera.set_look_from(16.0, -2.0, 10.0)
camera.set_look_at(0.0, 0.0, 0.0)
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
        # if e.key == gui.LMB:
        #     # left mose click
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
    # render_01(camera)
    Scene.render(camera, canvas, scene, samples_per_pixel, max_depth)
    gui.set_image(np.sqrt(canvas.to_numpy() / cnt))

    gui.show()
