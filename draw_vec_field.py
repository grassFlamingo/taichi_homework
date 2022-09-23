
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
gui_line_color = ti.field(ti.i32, (N_PARTICLES))

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

    # Vx = y + y*y
    # Vy = -0.5 * x + 0.2 * y - x * y + 6.0/5.0 * y*y

    # divergence vector field = 0;
    # \nabla \cdot B = 0
    Vx = y
    Vy = -x


    return ti.Vector([Vx, Vy])

@ti.func
def status_to_color(status) -> ti.i32:
    alpha = max(float(status) / N_TRAJECTORIES, 0.5)

    fcolor = 0xf * (1 - alpha) + 0x2 * alpha
    color = int(fcolor) & 0xf
    color = (color << 4) + color # 0xff
    color = (color << 16) + (color << 8) + color
    return color

@ti.kernel
def compute_field():
    """
    inplace
    """
    time_step = 0.05 / box_kb[0][0]
    for i in range(N_PARTICLES):
        status = trajectories_status[i]
        color = 0
        if status < 0:
            # kept
            pass
        elif status == 0:
            # no particles
            if ti.random() < 0.05:
                trajectories[i, 0] = ti.Vector([ti.random(), ti.random()])
                status = 1
                color = status_to_color(status)
        elif status < N_TRAJECTORIES:
            # compute next
            vec = time_step * get_velocity(trajectories[i, status-1])
            trajectories[i, status] = trajectories[i, status-1] + vec
            status += 1
            color = status_to_color(status)
        else:
            # particle die
            status = 0        
            
        trajectories_status[i] = status
        gui_line_color[i] = color

gui = ti.GUI("Draw vector field", res=666, background_color=0x222222)

gui.fps_limit = 20

trajectories.fill(0)
trajectories_status.fill(0)

box_kb[0] = ti.Vector([10, -5])
box_kb[1] = ti.Vector([10, -5])

gui_line_radius = np.asarray([min(3.0, 0.2*j) for j in range(N_TRAJECTORIES)])

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
        color = gui_line_color[i]
        traj_i = tnp[i]

        for j in range(trajectories_status[i] - 1):
            gui.line(traj_i[j], traj_i[j+1], radius=gui_line_radius[j], color=color)

    gui.show()
