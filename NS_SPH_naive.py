import numpy as np

import taichi as ti
ti.init()

num_particles = 2048
support_radius = 0.2
particle_radius = 0.02
acc_g = -9.8  # gravity acc
viscosity = 0.1
rho0 = 1000.0 
particle_dim = 2

mass = rho0 * np.pi * particle_radius * particle_radius

stiffness = 1.0

dt = 0.001

valid_particles = ti.field(ti.i32, ())

particles = ti.Struct.field({
    "rho": ti.f32,
    "p": ti.f32, # pressure
    "prr": ti.f32, # p / (rho * rho)
    "dv": ti.types.vector(2, ti.f32),
    "v": ti.types.vector(2, ti.f32),
    "x": ti.types.vector(2, ti.f32)},
    num_particles)

particle_neighbors = ti.field(ti.f32, (num_particles, num_particles))
particle_neighbors.fill(0)

@ti.func
def cubic_kernel(r_norm):
    # copied from example codes sph_base.py
    # value of cubic spline smoothing kernel
    k = 1.0
    h = support_radius
    if particle_dim == 1:
        k = 4 / (3 * h)
    elif particle_dim == 2:
        k = 40 / (7 * np.pi * h * h)
    elif particle_dim == 3:
        k = 8 / (np.pi * h * h * h)
    ans = 0.0
    if r_norm <= h:  # q <= 1
        q = r_norm / h
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            ans = k * (6.0 * q3 - 6.0 * q2 + 1)
        else:
            ans = k * 2 * ti.pow(1 - q, 3.0)
    return ans


@ti.func
def cubic_kernel_derivative(r, r_norm):
    # copied from example codes sph_base.py
    # derivative of cubic spline smoothing kernel
    k = 1.0
    h = support_radius
    if particle_dim == 1:
        k = 4.0 / (3.0 * h)
    elif particle_dim == 2:
        k = 40.0 / (7.0 * np.pi * h * h)
    elif particle_dim == 3:
        k = 8.0 / (np.pi * h * h * h)
    ans = ti.Vector([0.0 for _ in range(particle_dim)])
    if r_norm <= h:  # q <= 1.0:
        q = r_norm / h
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            ans = 6.0 * k * (3.0 * q - 2.0) * q * grad_q
        else:
            iq = 1.0 - q
            ans = -6.0 * k * iq * iq * grad_q
    return ans


@ti.func
def cubic_kernel_laplace(r, r_norm):
    k = 1.0
    h = support_radius
    if particle_dim == 1:
        k = 4 / (3 * h)
    elif particle_dim == 2:
        k = 40 / (7 * np.pi * h * h)
    elif particle_dim == 3:
        k = 8 / (np.pi * h * h * h)
    ans = ti.Vector([0.0 for _ in range(particle_dim)])
    if r_norm <= h:  # q <= 1.0
        q = r_norm / h
        # the hessian matrix is diagnal
        if q <= 0.5:
           rr = r_norm * r_norm
           ans = 6 * k * (3 * q * (rr + r*r)/(rr * h * h) - 2.0 / (h*h))
        else:
            rnh = r_norm * h
            iq = 1.0 - q
            ans = -6 * k * ( iq * (rnh - r*r * h / r_norm) - r * r) / (rnh * rnh)
    return ans


@ti.kernel
def search_neighbors():
    for i in range(num_particles):
        pi = particles[i].x
        for j in range(i+1, num_particles):
            r = pi - particles[j].x
            l = r.norm()
            if l > support_radius:
                continue
            particle_neighbors[i, j] = l
            particle_neighbors[j, i] = l


@ti.kernel
def evaluate_densities():
    # compute densities and pressure
    for i in range(num_particles):
        _xi = particles[i].x
        rho = 0.0
        for j in range(num_particles):
            if particle_neighbors[i, j] < 1e-4:
                continue
            l = (_xi - particles[j].x).norm()
            rho += cubic_kernel(l)
        # print("rho i", i, rho)
        particles[i].rho = rho * mass
        particles[i].p = max(0.0, stiffness * mass * (rho - rho0))
        particles[i].prr = particles[i].p / (particles[i].rho * particles[i].rho)

@ti.kernel
def sph_sample():
    gravity = ti.Vector([0.0, acc_g])
    for i in range(num_particles):
        xi = particles[i]
        vis = ti.Vector([0.0, 0.0])
        press = ti.Vector([0.0, 0.0])
        for j in range(num_particles):
            l = particle_neighbors[i,j]
            if l < 1e-6:
                continue

            xj = particles[j]

            r = xj.x - xi.x

            # 1. density is evaluated
            # 2. evaluate viscosity
            # v laplace(vi) = v sum_j m_j (vj-vi)/rhoj laplace(W_ij)
            vis += r / xj.rho * cubic_kernel_laplace(r, l)

            # print("c kernel lap", cubic_kernel_laplace(r, l))

            # 3. evaluate pressure gradient
            press += (xj.prr + xi.prr) * cubic_kernel_derivative(r, l)
        
        # print("vis", vis)
        # print("press", press)
    
        particles[i].dv = gravity + press * mass + vis * viscosity * mass

@ti.kernel
def update_particles():
    for i in range(num_particles):
        px = particles[i].x
        if px[0] < 0.01:
            particles[i].x[0] = 0.01
            particles[i].v[0] *= -0.3
        elif px[0] > 0.99:
            particles[i].x[0] = 0.99
            particles[i].v[0] *= -0.3
        elif px[1] < 0.01:
            particles[i].x[1] = 0.01
            particles[i].v[1] *= -0.3
        elif px[1] > 0.99:
            particles[i].x[1] = 0.99
            particles[i].v[1] *= -0.3
        else:
            particles[i].v += dt * particles[i].dv
            particles[i].x += dt * particles[i].v

# pipeline
# for each particles:
#  search neighbors j
# for each particles:
#  sample the velocity/density field using SPH
#  compute force/acceration using Navier-Stokes equaiton
# for each particles:
#  update velocity using acceleratin
#  update position velocity


def pipeline():
    particle_neighbors.fill(0.0)
    search_neighbors()
    # print(particle_neighbors)
    evaluate_densities()
    sph_sample()
    update_particles()


@ti.kernel
def system_init():
    for i in range(num_particles):
        _r = ti.random()*0.1
        _a = ti.random() * 2 * np.pi
        _xi = ti.Vector([0.5 + _r * ti.sin(_a) , 0.7 + _r * ti.cos(_a)])
        particles[i].x = _xi
        particles[i].v = [2.0, -10.0]

system_init()

gui = ti.GUI("NS_SPH", (800, 800))

# gui.fps_limit = 5

while gui.running:

    # draw particles
    px = particles.to_numpy()
    gui.circles(px["x"], radius=4, color=0x22ffee)

    gui.show()

    pipeline()

