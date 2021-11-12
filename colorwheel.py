import taichi as ti
import math
import numpy as np

ti.init()


N = 512

TWO_PI = 2 * math.pi
PI_OV_3 = math.pi / 3
img = ti.Vector.field(3, ti.f32)
roll = ti.Vector.field(3, ti.f32)
ti.root.pointer(ti.ij, 8).dense(ti.ij, N//8).place(img, roll)

gui = ti.GUI("color wheel", (N, N))

gui.clear()


@ti.kernel
def draw_picker(dst: ti.template()):
    maxr = float(N) / 2.0
    for i, j in ti.ndrange(N, N):
        fi, fj = (float(i) - maxr) / maxr, (float(j) - maxr) / maxr
        radius = ti.sqrt(fi*fi + fj*fj)
        if radius > 1.0:
            dst[i, j] = [0.0, 0.0, 0.0]
            continue
        angle = ti.acos(fj / radius)  # 0 -> 2 pi

        # hsv (angle, radius, 1.0)
        # hsv to rgb https://en.wikipedia.org/wiki/HSL_and_HSV
        h = angle
        s = radius
        v = ti.log(math.e - s)

        k5 = ti.mod(5.0 + h / PI_OV_3, 6.0)
        k3 = ti.mod(3.0 + h / PI_OV_3, 6.0)
        k1 = ti.mod(1.0 + h / PI_OV_3, 6.0)

        f5 = ti.max(0.0, ti.min(k5, ti.min(4.0-k5, 1.0)))
        f3 = ti.max(0.0, ti.min(k3, ti.min(4.0-k3, 1.0)))
        f1 = ti.max(0.0, ti.min(k1, ti.min(4.0-k1, 1.0)))

        dst[i, j] = v - v*s * ti.Vector([f5, f3, f1])

@ti.kernel
def rotate(dst: ti.template(), src: ti.template(), t: float):
    """
    rotate src to dst
    """
    rot = ti.Matrix.rotation2d(2 * math.pi * t)
    for i, j in ti.ndrange(N,N):
        p = (ti.Vector([i,j]) / N - 0.5)
        if p.norm() > 0.5:
            dst[i,j] = [0.0, 0.0, 0.0]
            continue

        p = rot @ p + 0.5

        ij = int(p * N)

        dst[i, j] = src[ij[0], ij[1]]

draw_picker(img)
draw_picker(roll)
               
t = 0.0
delt = 0.0
while gui.running:
    curx, cury = gui.get_cursor_pos()
    x = int(np.round(curx * N))
    y = int(np.round(cury * N))
    print(f"\rmouse at ({curx:.8f}, {cury:.8f}) color {img[x, y]}", end="")


    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ' ':
            delt = min(delt + 1e-2, 1.0)
        
    t = t + delt

    if delt > 0:
        rotate(roll, img, t)
            
    gui.set_image(roll)
    gui.show()

    delt = max(delt - 1e-3, 0.0)
    