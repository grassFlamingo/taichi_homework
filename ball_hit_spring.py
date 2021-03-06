import taichi as ti

ti.init()

N_x, N_y = 18, 2
N_width = 0.025

N_vertex = (N_x + 1) * (N_y + 1)
N_edges = N_x * N_y + N_x * (N_y+1) + N_y * (N_x+1)

vertex = ti.Struct.field({
    'x': ti.types.vector(2, ti.f32),
    'v': ti.types.vector(2, ti.f32),
    'f': ti.types.vector(2, ti.f32),
    'zz': ti.u8, }, # z for sleep
    shape=(N_vertex))

edges = ti.Vector.field(2, ti.i32, N_edges)
spring_length = ti.field(ti.f32, N_edges)  # the initial distance of two vertex

circle = ti.Struct.field({
    'r': ti.f32,
    'm': ti.f32,
    'x': ti.types.vector(2, ti.f32),
    'v': ti.types.vector(2, ti.f32),
    'f': ti.types.vector(2, ti.f32)},
    1)

curser = ti.Vector.field(2, ti.f32, 1)  # [x, y]

# physical quantities
imass = 1
grav_acc = 9.8  # Gravitational acceleration
# grav_acc = 10.0  # Gravitational acceleration
YoungsModulus = ti.field(ti.f32, ())

# time-step size (for simulation, 16.7ms)
# time_h = 16.7e-3
time_h = 1/60
# substepping
substepping = 64
# time-step size (for time integration)
time_dh = time_h/substepping


@ti.func
def ij2index(i: int, j: int, X: int):
    return i*X + j


@ti.kernel
def init_mesh():
    Nxx, Nyy = N_x + 1, N_y + 1
    for i, j in ti.ndrange(Nxx, Nyy):
        vid = ij2index(i, j, Nyy)
        vertex[vid].x = [float(i) * N_width + 0.1, float(j) * N_width + 0.1]
        vertex[vid].v = [0.0, 0.0]
        if j == 0:
            vertex[vid].zz = 1
        else:
            vertex[vid].zz = 0

    # holizontal edges
    ebase = 0
    for i, j in ti.ndrange(N_x, Nyy):
        eid = ebase + ij2index(i, j, Nyy)
        edges[eid] = [ij2index(i, j, Nyy), ij2index(i+1, j, Nyy)]

    ebase = N_x * Nyy
    # vertical edges
    for i, j in ti.ndrange(Nxx, N_y):
        ind = ij2index(i, j, N_y)
        edges[ebase + ind] = [ij2index(i, j, Nyy), ij2index(i, j+1, Nyy)]

    # diagonal edges
    ebase += Nxx * N_y
    for i, j in ti.ndrange(N_x, N_y):
        edges[ebase + ij2index(i, j, N_y)] = [
            ij2index(i+1, j, Nyy), ij2index(i, j+1, Nyy)]

    for i in range(N_edges):
        l = vertex[edges[i][0]].x - vertex[edges[i][1]].x
        spring_length[i] = l.norm()


@ti.kernel
def init_circle():
    circle[0].x = [0.2, 0.3]
    circle[0].v = [0.0, 0.0]
    circle[0].r = 0.04
    circle[0].m = 2.0


@ti.kernel
def compute_gradient():
    # clear gradient
    for i in range(N_vertex):
        vertex[i].f = [0.0, grav_acc]  # mass is 1

    # gradient of elastic potential
    for i in range(N_edges):
        p, q = edges[i][0], edges[i][1]

        r = vertex[p].x - vertex[q].x
        l = r.norm()
        if l < 1e-3:
            continue
        l0 = spring_length[i]

        # stiffness in Hooke's law
        k = YoungsModulus[None] * l0

        force = k * (l - l0) * r / l

        vertex[p].f += force
        vertex[q].f -= force


@ti.kernel
def update():
    # using symplectic intergration
    # elastic force + gravitation force, divding mass to get the acceleration
    circlex = circle[0].x
    mg = ti.Vector([0.0, circle[0].m * grav_acc])
    circle[0].f = mg
    for i in range(N_vertex):
        if vertex[i].zz == 1:
            continue
        xi = vertex[i].x
        xcir = xi - circlex
        xcirn = xcir.norm()
        if xcirn < circle[0].r:  # hit the ball. ? is this correct??
            # just freeze other balls

            # normal vector
            xinorm = xcir / xcirn

            nx = xinorm * (circle[0].r + 0.0001) + circlex

            dr = xi - nx

            # ?
            k = YoungsModulus[None] * spring_length[i]

            vertex[i].f += mg
            circle[0].f -= k * dr

            vertex[i].x = nx

        # Gravitational acceleration
        acc = -vertex[i].f * imass

        vertex[i].v += time_dh * acc * 0.001  # mul damping
        vertex[i].x += time_dh * vertex.v[i]

    # # explicit damping
    # for i in vertex:
    #     vertex[i].v *= ti.exp(-time_dh*0.1)


@ti.kernel
def update_circle(ispress: int):
    if ispress == 1:
        circle[0].x = curser[0]
        circle[0].v = [0.0, 0.0]
    else:
        acc = -circle[0].f / circle[0].m
        circle[0].v += time_dh * acc * 0.1  # mul damping
        # boundary check
        if circle[0].x[0] < 0.1:
            circle[0].x[0] = 0.1
            circle[0].v[0] *= -0.2
        elif circle[0].x[0] > 0.9:
            circle[0].x[0] = 0.9
            circle[0].v[0] *= -0.2
        elif circle[0].x[1] < 0.1:
            circle[0].x[1] = 0.1
            circle[0].v[1] *= -0.2
        else:
            circle[0].x += time_dh * circle[0].v

        # circle[0].v *= ti.exp(-time_dh*5)


init_mesh()
init_circle()

gui = ti.GUI("Ball hit spring", (800, 800), 0x101010)
# gui.fps_limit = 3

YoungsModulus[None] = 6e7


is_capture = False
cnt = 0
while gui.running:

    gui.circle(circle[0].x,
               color=0x4884FA, radius=circle[0].r*800)
    for i in range(N_edges):
        e = edges[i]
        gui.line(vertex[e[0]].x, vertex[e[1]].x, radius=2, color=0x82AD2A)

    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.LMB:
            curser[0] = gui.get_cursor_pos()
            print("clicked", curser[0])
            update_circle(1)
        if e.key == "r":
            init_mesh()
            init_circle()

    if is_capture and cnt % 5 == 0:
        gui.show(f"imgs/ball-hit-spring-{cnt//5:03d}.jpg")
    else:
        gui.show()
    cnt += 1

    for _ in range(substepping):
        update_circle(0)
        compute_gradient()
        update()
