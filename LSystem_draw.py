# reference 
# 1. http://paulbourke.net/fractals/lsys/
# 2. http://www.motionesque.com/beautyoffractals/

import taichi as ti
import numpy as np
import random
from LSystem import LSys, angle_360_2pi

PI_HALF = np.pi / 2
TWO_PI = 2 * np.pi

ti.init()

LsysCmds = [
    LSys("B+B--B+B;B=C+C--C+C;C=D+D--D+D;D=F+F--F+F", 5, np.pi / 3, 0.01),
    LSys("Y;X = X[-FFF][+FFF]FX;Y = YFX[+Y][-Y];", 5, angle_360_2pi(25.7), 0.01),
    LSys("F;F=FF+[+F-F-F]-[-F+F+F]", 5, angle_360_2pi(22.5), 0.008),
    LSys("F;F=F[+FF][-FF]F[-F][+F]F", 4, angle_360_2pi(35), 0.008),
    LSys("VZFFF; V = [+++W][---W]YV; W = +X[-W]Z; X = -W[+X]Z; Y = YZ; Z = [-FFF][+FFF]F", 10, angle_360_2pi(20), 0.01),
    LSys("FFX; X = >[-FFX]+FFX", 6, angle_360_2pi(40), 0.1),
    LSys("X; F = FF; X = F[+X]F[-X]+X;", 5, angle_360_2pi(20), 0.015),
    LSys("F; F = FF-[XY]+[XY]; X = +FY; Y = -FX", 6, angle_360_2pi(22.5), 0.01),
    LSys("F+XF+F+XF; X = XF-F+F-XF+F+XF-F+F-X;", 4, PI_HALF, 0.01, loc_start=(0.0, 0.5)),
    LSys("F+F+F+F; F = FF+F++F+F", 5, PI_HALF, 0.003, loc_start=(0.1,0.1)),
    LSys("F++F++F; F = F-F++F-F", 5, np.pi / 3, 0.003, loc_start=(0.1, 0.25)),
    LSys("-D--D; A = F++FFFF--F--FFFF++F++FFFF--F; B = F--FFFF++F++FFFF--F--FFFF++F; C = BFA--BFA; D = CFC--CFC", 6, np.pi/4, 0.02, loc_start=(0.5,0.5)),
    LSys("Y---Y; X = {F-F}{F-F}--[--X]{F-F}{F-F}--{F-F}{F-F}--; Y = f-F+X+F-fY", 6, np.pi/3, 0.02, loc_start=(0.5, 0.5)),
    LSys("F; F = -F++F-", 12, np.pi/4, 0.005, loc_start=(0.5, 0.3)),
    LSys("F+F+F+F+F+F;F=F++F-F-F-F-F++F;", 5, np.pi/3, 0.002, loc_start=(0.0, 0.5)),
]

lsys = random.choice(LsysCmds)
# lsys = LsysCmds[-1]

lsys.expand_cmd()


gui = ti.GUI("LSystem", (512, 512))

gui.clear(0x010101)
# TODO: try to make it move
LStart, Lend, Lwidth, Lcirc, Lradius = lsys.step()
while gui.running:
    gui.lines(LStart, Lend, Lwidth, 0x0aca5a)
    gui.show()
