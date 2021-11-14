# some codes are copied from taichi zoo

import taichi as ti

ti.init()

SO2x = ti.Matrix.rotation2d(0.5)


@ti.func
def sigmoid(x: float) -> float:
    return 1.0 / (1 + ti.exp(-1*x))


@ti.func
def hard_tanh(x: float, lower: float, upper: float) -> float:
    out = x
    if x < lower:
        out = lower
    elif x > upper:
        out = upper
    return out


@ti.func
def fract(i: float) -> float:
    """Just return the decimal"""
    return i - ti.floor(i)


@ti.func
def lerp(l, r, frac):
    return l + frac * (r - l)


@ti.func
def linear_mix(x: float, y: float, a: float) -> float:
    return x * (1.0-a) + y * a


@ti.func
def random_gradient(i: float, j: float):
    random_val = 2920 * ti.sin(i * 21942.0 + j * 171324.0 + 8912.0) * \
        ti.cos(i * 23157.0 * j * 217832.0 + 9758.0)
    return ti.Vector([ti.cos(random_val), ti.sin(random_val)])


@ti.func
def dot_grid_gradient(ipx: float, ipy: float, fpx: float, fpy: float):
    grad = random_gradient(ipx, ipy)
    return grad.x * (fpx-ipx) + grad.y * (fpy-ipy)  # dot(grad, frac)


@ti.func
def quintic_interpolate(l, r, t):
    t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return lerp(l, r, t)

# Perlin noise


@ti.func
def perlin(i, j):
    ipx = ti.floor(i)
    ipy = ti.floor(j)
    ll = dot_grid_gradient(ipx, ipy, i, j)
    lr = dot_grid_gradient(ipx + 1.0, ipy, i, j)
    ul = dot_grid_gradient(ipx, ipy + 1.0, i, j)
    ur = dot_grid_gradient(ipx + 1.0, ipy + 1.0, i, j)
    lerpxl = quintic_interpolate(ll, lr, fract(i))
    lerpxu = quintic_interpolate(ul, ur, fract(i))
    return quintic_interpolate(lerpxl, lerpxu, fract(j)) * 0.5 + 0.5


@ti.func
def prtlin_fbm(i: float, j: float, octaves=8) -> float:
    value = 0.0
    amplitude = 0.5
    # frequency = 0.0

    # loop of octaves
    for _ in range(octaves):
        # value += amplitude * perlin(i, j)
        value += amplitude * perlin(i, j)**2  # sharp
        i *= 2.0
        j *= 2.0
        amplitude *= 0.5

    return value


@ti.func
def rot_fbm(i: float, j: float, octaves=5) -> float:
    value = 0.0
    amplitude = 0.7

    st = ti.Vector([i, j])
    # loop of octaves
    for _ in range(octaves):
        value += amplitude * perlin(st[0], st[1])
        st = SO2x @ st
        st *= 2.0
        amplitude *= 0.5

    return value


@ti.func
def random_gradient(i, j):
    random_val = 2920.0 * ti.sin(i * 21942.0 + j * 171324.0 + 8912.0) * \
        ti.cos(i * 23157.0 * j * 217832.0 + 9758.0)
    return ti.Vector([ti.cos(random_val), ti.sin(random_val)])


@ti.func
def dot_grid_gradient(ipx, ipy, fpx, fpy):
    grad = random_gradient(ipx, ipy)
    return grad.x * (fpx-ipx) + grad.y * (fpy-ipy)  # dot(grad, frac)


@ti.kernel
def generate_pattern(img: ti.template()):
    s0 = float(img.shape[0])
    s1 = float(img.shape[1])
    m1 = [ti.random() for _ in ti.static(range(3))]
    m2 = [ti.random() for _ in ti.static(range(3))]
    for i, j in img:
        img[i, j] = [
            # prtlin_fbm(float(i) / s0 + m1[0], float(j) / s1 + m2[0]),
            0.0,
            hard_tanh(rot_fbm(float(i) / s0 + \
                              m1[1], float(j) / s1 + m2[1]), 0.0, 1.0),
            hard_tanh(rot_fbm(float(i) / s0 + \
                              m1[2], float(j) / s1 + m2[2]), 0.0, 1.0),
        ]



@ti.kernel
def generate_pattern_t(img: ti.template(), t: float):
    s0 = float(img.shape[0])
    s1 = float(img.shape[1])
    for i, j in img:
        _i = float(i) / s0
        _j = float(j) / s1
        _p = rot_fbm(_i + 0.1*t, _j + 0.2 + 0.3*t)
        img[i, j] = [
            0.1 * rot_fbm(_i + 0.9 + t, _j  + 0.17 +  0.02*t),
            rot_fbm(_i + 0.1 + 0.2*t, _j + 0.2*t)**2,
            0.8 * rot_fbm(_i + _p + 0.2*t, _j + 0.1*_p + 0.1*t),
        ]

N = 512
timg = ti.Vector.field(3, ti.f32, shape=(N, N))

gui = ti.GUI("smoke", (N, N))
gui.fps_limit = 24

t = 0.0
while gui.running:
    generate_pattern_t(timg, t)
    gui.set_image(timg)
    gui.show()
    t += 0.1
