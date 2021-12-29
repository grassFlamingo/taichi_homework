import taichi as ti
from taichi.ui.constants import PRESS
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
        ti.Vector([-8, 8, 58]),
        50.0,
        Scene.M_light_source,
        ti.Vector([10.0, 10.0, 10.0])
    )
)

# scene.append(ssoup, "sphere soup")
ply_main = NPLYReader("./mesh/cristal.ply")
texture_img = np.asarray(PIL.Image.open(
    "./mesh/crystal-main.png").rotate(-90), dtype=np.float32) / 255.0

texture = Scene._ensure_ti_field(texture_img)

scene.append(Scene.TriangleSoup(
    ply_main.vertex,
    ply_main.faces,
    Scene.M_dielectric,
    texture
))

canvas.fill(0)

camera = Scene.Camera()
camera.set_look_from(7.28, -4.16, 4.16)
camera.set_look_at(-2.28, 0.57, 0.04)
camera.reset()

gui = ti.GUI("ray tracing", res=(image_width, image_height))
# gui.fps_limit = 3
cnt = 0
clicked_loc = None
while gui.running:
    for e in gui.get_events(ti.GUI.WHEEL, ti.GUI.PRESS, ti.GUI.RELEASE):

        if not e.key in [ti.GUI.WHEEL, ti.GUI.LMB]:
            continue
        if e.key == ti.GUI.WHEEL:
            fx, fy, fz = camera.look_from()
            lf = np.array([fx, fy, fz])
            lf *= 0.02
            if e.delta[1] > 0:
                lf *= -1
                camera.set_look_from_delta(lf[0], lf[1], lf[2])
            elif e.delta[1] < 0:
                camera.set_look_from_delta(lf[0], lf[1], lf[2])
        elif e.key == ti.GUI.LMB:
            if e.type == ti.GUI.PRESS:
                u, v = gui.get_cursor_pos()
                clicked_loc = camera.get_ray(u, v).direction.to_numpy()
            else:
                u, v = gui.get_cursor_pos()
                # print("click", u, v)
                ray = camera.get_ray(u,v)
                dir = clicked_loc - ray.direction.to_numpy()
                dir *= 5.0
                camera.set_look_at_delta(dir[0], dir[1], dir[2])

        camera.reset()
        canvas.fill(0)  # buggy
        cnt = 0.0
        print("look form", camera.look_from(), "look at", camera.look_at())

    cnt += 1
    # render_01(camera)
    Scene.render(camera, canvas, scene, samples_per_pixel, max_depth)
    gui.set_image(np.sqrt(canvas.to_numpy() / cnt))

    gui.show()
