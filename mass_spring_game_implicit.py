# Tutorials (Chinese):
# - https://www.bilibili.com/video/BV1UK4y177iH
# - https://www.bilibili.com/video/BV1DK411A771
# copied from https://www.github.com/taichi-dev/taichi/master/python/taichi/examples/simulation/mass_spring_game.py

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

spring_Y = ti.field(dtype=ti.f32, shape=())  # Young's modulus
paused = ti.field(dtype=ti.i32, shape=())
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())

max_num_particles = 512
particle_mass = 1.0
dt = 1/60
# substeps = 10

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
guess_x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
cache_q = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
dE = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
dG = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
guess_dx = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
cache_r = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
cache_d = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
ddE = ti.Matrix.field(2, 2, dtype=ti.f32, shape=(
    max_num_particles, max_num_particles))
fixed = ti.field(dtype=ti.i32, shape=max_num_particles)

# rest_length[i, j] == 0 means i and j are NOT connected
rest_length = ti.field(dtype=ti.f32,
                       shape=(max_num_particles, max_num_particles))


@ti.kernel
def compute_force():
    # prepare dE, ddE and tx
    n = num_particles[None]

    I = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])

    # Implicit integration
    # g = 1/2 ||x - (xn - h vn)||_M^2 + h^2 E(x)
    # dg = M (x - (xn + h vn)) + h^2 dE
    # dE = the computed force
    # reset dE, tx
    for i in range(n):
        dE[i] = [0.0, 2.0*particle_mass]  # 2 is g
        guess_x[i] = x[i]

    # compute dE, ddE
    for i in range(n):
        for j in range(i+1, n):
            l0 = rest_length[i, j]
            if l0 <= 0:
                # not connetted
                continue
            r = x[i] - x[j]
            l = r.norm()
            if l < 1e-8:
                ddE[i,j] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                continue

            k = spring_Y[None] * l0
            # k (l - l0) r / l
            # = k (1 - l0/l) r
            f = k * (l - l0) * r / l

            dE[i] += f
            dE[j] -= f

            rrt = ti.Matrix([
                [r[0] * r[0], r[0] * r[1]],
                [r[1] * r[0], r[1] * r[1]]
            ])

            ddE[i, j] = k * (I - (l0/l) * (I - rrt / (l*l)))

@ti.kernel
def compute_dG():
    n = num_particles[None]
    for i in range(n):
        # dG = M(x - (xn + h vn)) + h^2 dE
        dG[i] = particle_mass * \
            (guess_x[i] - (x[i] + dt * v[i])) + dt * dt * dE[i]


@ti.kernel
def blas_dot(a: ti.template(), b: ti.template()) -> ti.f32:
    """ return a dot b """
    n = num_particles[None]
    out = 0.0
    for i in range(n):
        out += a[i].dot(b[i])
    return out


@ti.kernel
def blas_saxpy(_r: ti.template(), alpha: float, _x: ti.template(), _y: ti.template()):
    """r = alpha x + y"""
    n = num_particles[None]
    for i in range(n):
        yi = _y[i]
        _r[i] = alpha * _x[i] + yi

@ti.kernel
def blas_ddg_mv(dst: ti.template(), src: ti.template()):
    # compute dst = ddG @ src
    n = num_particles[None]
    for i in range(n):
        dst[i] = particle_mass * src[i]

    for i in range(n):
        for j in range(i+1, n):
            if rest_length[i, j] <= 0:
                continue
            # ddE is the Hessian matrix for spring force edge i and j
            # ddG = M + dt * dt * ddE
            _tmp = ddE[i, j] @ (src[i] - src[j])
            _tmp = dt * dt * _tmp

            dst[i] += _tmp
            dst[j] -= _tmp

@ti.kernel
def blas_copy(dst: ti.template(), src: ti.template()):
    """dst = src"""
    n = num_particles[None]
    for i in range(n):
        dst[i] = src[i]

@ti.kernel
def update_x():
    n = num_particles[None]
    for i in range(n):
        if fixed[i]:
            continue
        v[i] = drag_damping[None] * (guess_x[i] - x[i])/dt
        x[i] = guess_x[i]
        # Collide with four walls
        for d in ti.static(range(2)):
            # d = 0: treating X (horizontal) component
            # d = 1: treating Y (vertical) component

            if x[i][d] < 0:  # Bottom and left
                x[i][d] = 0  # move particle inside
                v[i][d] = 0  # stop it from moving further

            if x[i][d] > 1:  # Top and right
                x[i][d] = 1  # move particle inside
                v[i][d] = 0  # stop it from moving further

@ti.kernel
def reduce_norm_mean(_x: ti.template()) -> ti.f32:
    n = num_particles[None]
    ans = 0.0
    for i in range(n):
        ans += _x[i].norm()
    return ans / float(n)

def iter_solve():
    # compute dE, ddE, tx
    compute_force()
    # iteration
    n = num_particles[None]
    i_max = n*n
    
    for _ in range(16):
        # conjugate gradient
        # Ax = b; A = ddG; x = guess_dx; b = dG
        compute_dG()
        # ddG x = dG
        blas_copy(guess_dx, dG) # initial guess

        if np.isnan(dG[0][0]):
            print("explosion!!!")
            exit(255)

        # r = b - A @ x
        blas_ddg_mv(cache_q, guess_dx) # A @ x
        blas_saxpy(cache_r, -1, cache_q, dG) # r = - A @ x + b
        # d = r
        blas_copy(cache_d, cache_r)
        # delta_new = r.dot(r)
        deln = blas_dot(cache_r, cache_r)
        del0 = deln

        i = 0
        while i < i_max and deln > 0.1 * del0:
            i += 1
            # q = A @ d
            blas_ddg_mv(cache_q, cache_d)
            # alpha = delta_new / d.dot(q)
            alpha = deln / blas_dot(cache_d, cache_q)
            # x = alpha * d + x
            blas_saxpy(guess_dx, alpha, cache_d, guess_dx)

            # r = b - A @ x
            blas_ddg_mv(cache_q, guess_dx) # A @ x
            # - A @ x + b
            blas_saxpy(cache_r, -1, cache_q, dG)

            delm = deln
            deln = blas_dot(cache_r, cache_r)
            
            beta = deln / delm
            # d = r + beta d
            blas_saxpy(cache_d, beta, cache_d, cache_r)

        # print("i", i, deln/del0)
        # # cache_q should be close to dG
        # blas_ddg_mv(cache_q, guess_dx)
        # print("cg", cache_q[0], dG[0])

        # update x = - 0.1 dx + x
        blas_saxpy(guess_x, -0.1, guess_dx, guess_x)
        if reduce_norm_mean(guess_dx) < 1e-5:
            # converged
            # print("break iter")
            break
    update_x()

@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    # Taichi doesn't support using vectors as kernel arguments yet, so we pass scalars
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    fixed[new_particle_id] = fixed_
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        connection_radius = 0.15
        if dist < connection_radius:
            # Connect the new particle with particle i
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1

@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(num_particles[None]):
        p = ti.Vector([pos_x, pos_y])
        v[i] += -dt * (x[i] - p) * 100


gui = ti.GUI('Implicit Mass Spring System',
             res=(512, 512),
             background_color=0xDDDDDD)

# gui.fps_limit = 24

spring_Y[None] = 1000
drag_damping[None] = 1
dashpot_damping[None] = 100

new_particle(0.3, 0.3, False)
new_particle(0.3, 0.4, False)
new_particle(0.4, 0.4, False)

while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1],
                         int(gui.is_pressed(ti.GUI.SHIFT)))
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 'y':
            if gui.is_pressed('Shift'):
                spring_Y[None] /= 1.1
            else:
                spring_Y[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                drag_damping[None] /= 1.1
            else:
                drag_damping[None] *= 1.1
        elif e.key == 'x':
            if gui.is_pressed('Shift'):
                dashpot_damping[None] /= 1.1
            else:
                dashpot_damping[None] *= 1.1

    if gui.is_pressed(ti.GUI.RMB):
        cursor_pos = gui.get_cursor_pos()
        attract(cursor_pos[0], cursor_pos[1])

    if not paused[None]:
        iter_solve()

    X = x.to_numpy()
    n = num_particles[None]

    # Draw the springs
    for i in range(n):
        for j in range(i + 1, n):
            if rest_length[i, j] != 0:
                gui.line(begin=X[i], end=X[j], radius=2, color=0x444444)

    # Draw the particles
    for i in range(n):
        c = 0xFF0000 if fixed[i] else 0x111111
        gui.circle(pos=X[i], color=c, radius=5)

    gui.text(
        content=f'Left click: add mass point (with shift to fix); Right click: attract',
        pos=(0, 0.99),
        color=0x0)
    gui.text(content=f'C: clear all; Space: pause',
             pos=(0, 0.95),
             color=0x0)
    gui.text(content=f'Y: Spring Young\'s modulus {spring_Y[None]:.1f}',
             pos=(0, 0.9),
             color=0x0)
    gui.text(content=f'D: Drag damping {drag_damping[None]:.2f}',
             pos=(0, 0.85),
             color=0x0)
    gui.text(content=f'X: Dashpot damping {dashpot_damping[None]:.2f}',
             pos=(0, 0.8),
             color=0x0)
    gui.show()
