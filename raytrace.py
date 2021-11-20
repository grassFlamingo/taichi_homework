import taichi as ti
import Scene
import numpy as np

# ti.init(excepthook=True)
ti.init(debug=True)

# Canvas
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# Rendering parameters
samples_per_pixel = 4
max_depth = 64

@ti.kernel
def render_01(camera: ti.template()):
    for i, j in canvas:
        ccolor = canvas[i, j]
        u = (float(i) + ti.random()) / image_width
        v = (float(j) + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        ray = camera.get_ray(u, v)
        for _ in range(samples_per_pixel):
            color += ray_color_01(ray)
        color /= samples_per_pixel
        canvas[i, j] += 0.98 * (color - ccolor)


@ti.func
def ray_color_01(ray):
    default_color = ti.Vector([0.0, 0.0, 0.0])
    info = scene.hit(ray)
    if info[0] == 1:
        default_color = info[5]
    return default_color


@ti.kernel
def render(camera: ti.template(), kn: float):
    for i, j in canvas:
        u = (float(i) + ti.random()) / image_width
        v = (float(j) + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        ray = camera.get_ray(u, v)
        for _ in range(samples_per_pixel):
            color += ray_color(ray)
        color /= samples_per_pixel
        
        cc = canvas[i,j]
        canvas[i, j] += kn * (color - cc)

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
            fuzz = 0.0
            if material == Scene.M_fuzzy_metal:
                fuzz = 0.4
            
            scattered_origin = hit_point
            scattered_direction = Scene.reflect(scattered_direction.normalized(), hit_normal) + fuzz * Scene.random_unit_vector()

            if scattered_direction.dot(hit_normal) < 0:
                break
            else:
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
                scattered_direction = Scene.reflect(scattered_direction, hit_normal)
            else:
                scattered_direction = Scene.refract(scattered_direction, hit_normal, refraction_ratio)
            scattered_origin = hit_point
            brightness *= color
        
        brightness /= p_RR
    
    return color_buffer


scene = Scene.Scene()

scene.add_obj(
    Scene.Box.newbox(
        ti.Vector([4, 4, 4]),
        ti.Vector([8.0, 8.0, 8.0]),
        [Scene.M_diffuse,
         Scene.M_diffuse,
         Scene.M_diffuse,
         Scene.M_diffuse,
         Scene.M_fuzzy_metal,
         Scene.M_diffuse],
        [ti.Vector(np.random.rand(3) * 0.8 + 0.1) for _ in range(6)],
        normal_out=False), "room_walls")

# for i in range(8):
#     bi = bin(i+8) # 0b1000 -> 0b1111
#     veci = [float(bi[3]), float(bi[4]), float(bi[5])]
#     scene.add_obj(
#         Scene.Sphere(
#             ti.Vector(veci),
#             0.1,
#             Scene.M_diffuse,
#             ti.Vector(veci),
#         ),
#         "ball%d"%i
#     )

# scene.add_obj(
#     Scene.Plane(
#         ti.Vector([0.0, 0.0, 0.0]),
#         ti.Vector([1.0, 0.0, 0.0]),
#         Scene.Material.specular,
#         ti.Vector([0.1, 0.0, 0.2])
#     )
# )

# scene.add_obj(
#     Scene.Parallelogram(
#         ti.Vector([0.5, 0.5, 0.5]),
#         ti.Vector([0.5, 0.0, 0.2]),
#         ti.Vector([0.5, 0.2, 0.0]),
#         Scene.Material.specular,
#         ti.Vector([0.0, 0.0, 1.0]),
#     ),
#     "parallelogram"
# )

# scene.add_obj(
#     Scene.Triangle(
#         ti.Vector([-0.3, 0.2, 0.7]),
#         ti.Vector([-0.3, 0.0, 0.7]),
#         ti.Vector([-0.3, 0.0, 0.3]),
#         Scene.Material.specular,
#         ti.Vector([0.0, 1.0, 0.0]),
#     ),
#     "trangle"
# )

scene.add_obj(
    Scene.Sphere(
        ti.Vector([4.0, 4.0, 7.0]),
        1.0,
        Scene.M_light_source,
        ti.Vector([8.0, 8.0, 8.0]),
    ),
    "source"
)

scene.add_obj(
    Scene.Sphere(
        ti.Vector([5.0, 5.0, 2.0]),
        2.0,
        Scene.M_dielectric,
        ti.Vector([1.0, 1.0, 1.0]),
    ),
    "grass"
)

scene.add_obj(
    Scene.Box.newbox(
        ti.Vector([2.0, 2.0, 1.0]),
        ti.Vector([1.0, 1.0, 1.0]),
        [Scene.M_metal,
         Scene.M_metal,
         Scene.M_fuzzy_metal,
         Scene.M_dielectric,
         Scene.M_dielectric,
         Scene.M_diffuse],
        [ti.Vector(np.random.rand(3) * 0.8 + 0.2) for _ in range(6)]),
    "box")

canvas.fill(0)

camera = Scene.Camera()
camera.set_look_from(5.5, -12.5, 3.0)
camera.set_look_at(-7.5, 89.5, 1.3)
camera.reset()

gui = ti.GUI("ray tracing", res=(image_width, image_height))
# gui.fps_limit = 3
delta = 0.5
cnt = 0.0
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
        canvas.fill(0) # buggy
        cnt = 0.0
        print("look form", camera.look_from(), "look at", camera.look_at())

    cnt += 1.0
    render(camera, 1.0 / cnt)
    gui.set_image(np.sqrt(canvas.to_numpy()))

    if cnt == 20.0:
        gui.show("imgs/raytrace-%d.jpg"%cnt)
    else:
        gui.show()
