import taichi as ti

ti.init()

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1
PI = 3.141592653

# number of planets
N = 128
# galaxy size
# galaxy_size = 0.4
# planet radius (for rendering)
# planet_radius = 2

# time-step size
h = 1e-4
# substepping
substepping = 5

# pos, vel and force of the planets
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
mass = ti.field(dtype=float, shape=(N))
force = ti.Vector.field(2, ti.f32, N)


@ti.func
def rand(a: float, b:float) -> float:
    if a > b:
        t = b
        b = a
        a = t
    return ti.random() * (b - a) + a

@ti.kernel
def initialize():
    for i in range(N):
        mass[i] = rand(1.0, 10.0) # 1,10
        pos[i] = [rand(0.2, 0.8), rand(0.2, 0.8)]
        vel[i] = [rand(0.2, 0.8), rand(0.2, 0.8)]

@ti.kernel
def compute_force():
    # clear force
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])

    # compute gravitational force
    for i in range(N):
        p = pos[i]
        for j in range(N):# double the computation for a better memory footprint and load balance
            if i == j:  continue
            diff = p-pos[j]
            r = diff.norm(1e-5)

            # gravitational force -(GMm / r^2) * (diff/r) for i
            f = -G * mass[i] * mass[j] * (1.0/r)**3 * diff

            # assign to each particle
            force[i] += f

@ti.kernel
def update():
    dt = h/substepping
    for i in range(N):
        #symplectic euler
        vel[i] += dt*force[i]/mass[i]
        pos[i] += dt*vel[i]
        if pos[i][0] < 0 or pos[i][0] > 1:
            vel[i][0] *= -1
        if pos[i][1] < 0 or pos[i][1] > 1:
            vel[i][1] *= -1

gui = ti.GUI('N-body problem', (512, 512))

initialize()

capidx = 0
is_capture = False
while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in ['c', 'C']:
            is_capture = not is_capture

    for i in range(substepping):
        compute_force()
        update()

    gui.clear(0x112F41)
    gui.circles(pos.to_numpy(), color=0xfafafa, radius=mass.to_numpy())
    if is_capture:
        if capidx % 10 == 0:
            gui.show(f"imgs/Nbodies-cap-{capidx//10:03d}.png")
        else:
            gui.show()
        capidx += 1
    else:
        gui.show()
