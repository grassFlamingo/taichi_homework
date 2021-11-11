import taichi as ti
import numpy as np

ti.init()

A=0.2
B=0.3
PI=np.pi
TWO_PI = 2*PI
HALF_PI = PI / 2
alpha = 3.0 * TWO_PI
beta= 1.0 * TWO_PI
phase_diff = 0.5 * PI

pos = ti.field(dtype=ti.f32, shape=(256,2))
psize = ti.field(dtype=ti.f32, shape=(256))

@ti.kernel
def init_psize():
    for i in range(psize.shape[0]):
        psize[i] = 0.0
        

@ti.kernel
def run(i: int, t: float):
    pos[i,0] = A * ti.sin(alpha * t + phase_diff) + 0.5
    pos[i,1] = B * ti.sin(beta * t) + 0.5

@ti.kernel
def update_size(i: int):
    for j in range(psize.shape[0]):
        if j == i:
            psize[j] = 5.12
        else:
            psize[j] -= 0.02

gui = ti.GUI("Lissajous", res=(512, 512), background_color=0x1f1f1f)
# gui.fps_limit = 24
gui.clear()
gui.show()

t = 0.0
i = 0
delT = 4e-3
capidx = 0
is_capture = False
capture_per_frame = 16
while gui.running:
    run(i, t)
    update_size(i)
    i = (i + 1) % 256
    t += delT

    print("\r i=%3d, t=%.8f  "%(i,t), end="")
    gui.circles(pos.to_numpy(), radius=psize.to_numpy(), color=0xfa00fa)
    # gui.show() # this-> has clear at the end!
    # gui.core.update()
    # gui.frame += 1
    if is_capture and t > 1 and t < 4:
        if capidx % capture_per_frame == 0:
            gui.show(f"imgs/lissajous-cap-{capidx//capture_per_frame:03d}.png")
        else:
            gui.show()
        capidx += 1
    else:
        gui.show()

    

