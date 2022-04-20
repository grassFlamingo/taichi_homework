
import taichi as ti

import numpy as np

ti.init()

# reference https://anvaka.github.io/fieldplay

N_PARTICLES = 4096
N_TRAJECTORIES = 16

trajectories = ti.Vector.field(2, ti.f32, (N_PARTICLES, N_TRAJECTORIES))
# status
# 0: no trajectories
# 1 - k: trajectority length
trajectories_status = ti.field(ti.i32, N_PARTICLES)

box_kb = ti.Vector.field(2, ti.f32, 2)

# time_step = 0.01

@ti.func
def get_velocity(vec):
    x = vec[0] * box_kb[0][0] +  box_kb[0][1]
    y = vec[1] * box_kb[1][0] +  box_kb[1][1]

    # return Vx, Vy
    # Vx = x * (3 - x - y)
    # Vy = y * (2 - x - y)
    # 
    # Vx = x * (3 - 2*x - y)
    # Vy = y * (2 - x - y)

    # Vx = ti.sin(y)
    # Vy = ti.cos(x)

    # Vx = y
    # Vy = x

    # Vx = 1 + y - ti.exp(-x)
    # Vy = x*x*x - y


    # Vx = ti.sin(y)
    # Vy = x - x*x*x

    # Vx = x*y - 1
    # Vy = x - y*y*y

    Vx = y + y*y
    Vy = -0.5 * x + 0.2 * y - x * y + 6.0/5.0 * y*y

    return ti.Vector([Vx, Vy])


@ti.kernel
def compute_field():
    """
    inplace
    """
    time_step = 0.05 / box_kb[0][0]
    for i in range(N_PARTICLES):
        status = trajectories_status[i]
        if status < 0:
            # kept
            pass
        elif status == 0:
            # no particles
            if ti.random() < 0.05:
                trajectories[i, 0] = ti.Vector([ti.random(), ti.random()])
                status = 1
        elif status < N_TRAJECTORIES:
            # compute next
            vec = time_step * get_velocity(trajectories[i, status-1])
            trajectories[i, status] = trajectories[i, status-1] + vec
            status += 1
        else:
            # particle die
            status = 0        
            
        trajectories_status[i] = status

gui = ti.GUI("Draw vector field", res=666, background_color=0x222222)

gui.fps_limit = 20

trajectories.fill(0)
trajectories_status.fill(0)

box_kb[0] = ti.Vector([10, -5])
box_kb[1] = ti.Vector([10, -5])

while gui.running:

    for e in gui.get_events(ti.GUI.WHEEL):
        if e.key == ti.GUI.WHEEL:
            if e.delta[1] > 0:
                box_kb[0] = 0.9 * box_kb[0]
                box_kb[1] = 0.9 * box_kb[1]
            else:
                box_kb[0] = 1.1 * box_kb[0]
                box_kb[1] = 1.1 * box_kb[1]
            
            print(box_kb[0], box_kb[1])

    compute_field()
    
    # draw field
    gui.clear()
    tnp = trajectories.to_numpy()

    for i in range(N_PARTICLES):
        # print("get traj", i, trajectories_status[i])
        status = float(trajectories_status[i])
        alpha = max(status / N_TRAJECTORIES, 0.5)

        color = 0xf * (1 - alpha) + 0x2 * alpha
        color = int(color) & 0xf
        color = (color << 4) + color # 0xff
        color = (color << 16) + (color << 8) + color

        
        for j in range(trajectories_status[i] - 1):
            gui.line(tnp[i,j], tnp[i,j+1], radius=min(3, 0.2*j), color=color)


    gui.show()
