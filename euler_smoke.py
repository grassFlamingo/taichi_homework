import taichi as ti

# reference 
# - taichi course
# - some codes are copied from static_fuild.py
# 
ti.init()

res = 512
dt = 1 / 24.0

velocity = ti.Vector.field(2, ti.f32, shape=(res, res))
velocity_t = ti.Vector.field(2, ti.f32, shape=(res, res))
velocity_tt = ti.Vector.field(2, ti.f32, shape=(res, res))
velocity_divs = ti.field(ti.f32, shape=(res, res))
pressure = ti.field(ti.f32, shape=(res, res))
valQ = ti.Vector.field(2, ti.f32, shape=(res, res))
valQ_t = ti.Vector.field(2, ti.f32, shape=(res, res))

# use a sparse matrix to solve Poisson's pressure equation.
@ti.kernel
def fill_laplacian_matrix(A: ti.linalg.sparse_matrix_builder()):
    for i, j in ti.ndrange(res, res):
        row = i * res + j
        center = 0.0
        if j != 0:
            A[row, row - 1] += -1.0
            center += 1.0
        if j != res - 1:
            A[row, row + 1] += -1.0
            center += 1.0
        if i != 0:
            A[row, row - res] += -1.0
            center += 1.0
        if i != res - 1:
            A[row, row + res] += -1.0
            center += 1.0
        A[row, row] += center

N = res * res
K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
velocity_div_b = ti.field(ti.f32, shape=N)

fill_laplacian_matrix(K)
L = K.build()
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(L)
solver.factorize(L)

@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl[0] = -vc[0]
        if i == res - 1:
            vr[0] = -vc[0]
        if j == 0:
            vb[1] = -vc[1]
        if j == res - 1:
            vt[1] = -vc[1]
        # delta x = 0.5
        velocity_divs[i, j] = (vr[0] - vl[0] + vt[1] - vb[1]) * 0.5

@ti.kernel
def copy_divergence(div_in: ti.template(), div_out: ti.template()):
    for i, j in div_in:
        div_out[i * res + j] = -div_in[i, j]


@ti.kernel
def apply_pressure(p_in: ti.ext_arr(), p_out: ti.template()):
    for i, j in p_out:
        p_out[i, j] = p_in[i * res + j]

@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = max(0, min(res - 1, I))
    return qf[I]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.func
def bilerp(vf, p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)

# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p

@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p)

@ti.kernel
def apply_force(vf: ti.template(), vff: ti.template()):
    _grav = dt * ti.Vector([0.0, -2.0]) # td * g
    midx = int(res / 2)
    winw = int(0.2*res)
    winh = int(0.1*res)
    for i, j in vf:
        _f = _grav
        if abs(i - midx) < winw and j < winh:
            _f[1] += 4.0 

        vff[i,j] = vf[i,j] + _f # + v nabla^2 vf
        
@ti.kernel
def project_v(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])

def step():
    # step 1 advection
    # q(n+1) = advect(v(n), dt, q(n))
    advect(velocity, valQ, valQ_t)
    valQ.copy_from(valQ_t) # copy
    # v1 = advect(v(n), dt, v(n))
    advect(velocity, velocity, velocity_t)

    # step 2 applying forces
    # v2 = v1 + dt(g + v nabla^2 v1)
    apply_force(velocity_t, velocity_tt)

    # step 3 projection
    # v(n+1) = project(dt, v2)
    divergence(velocity_tt)
    copy_divergence(velocity_divs, velocity_div_b)
    _p = solver.solve(velocity_div_b)
    apply_pressure(_p, pressure)
    project_v(velocity, pressure)

    # return v(n+1), q(n+1)

def init_system():
    velocity.fill(0)
    valQ.fill(0)

@ti.kernel
def generate_spots():
    midx = int(res / 2)
    winw = int(0.2*res)
    winh = int(0.1*res)
    for i, j in valQ:
        if abs(i - midx) < winw and j < winh:
            valQ[i,j] += 1.0

gui = ti.GUI("euler", (res, res))

while gui.running:
    step()

    gui.set_image(valQ.to_numpy())

    gui.show()

