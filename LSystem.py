"""
The L System 



Refernece:
1. http://paulbourke.net/fractals/lsys/

Character        Meaning
   F	         Move forward by line length drawing a line
   f	         Move forward by line length without drawing a line
   +	         Turn left by turning angle
   -	         Turn right by turning angle
   |	         Reverse direction (ie: turn by 180 degrees)
   [	         Push current drawing state onto stack
   ]	         Pop current drawing state from the stack
   #	         Increment the line width by line width increment
   !	         Decrement the line width by line width increment
   @	         Draw a dot with line width radius
   {	         Open a polygon
   }	         Close a polygon and fill it with fill colour
   >	         Multiply the line length by the line length scale factor
   <	         Divide the line length by the line length scale factor
   &	         Swap the meaning of + and -
   (	         Decrement turning angle by turning angle increment
   )	         Increment turning angle by turning angle increment
"""
import numpy as np

TWO_PI = 2 * np.pi
WORD_CHAR = list(r"Ff+-|[]#!@{}><&()")


def _force_0_to_2pi(x: float) -> float:
    # make sure angle_min in (0, 2 pi)
    k = np.floor((TWO_PI - x) / TWO_PI)
    return x + k * TWO_PI


def _iter_expand(axiom, idict, depth=1, maxdepth=4):
    olist = []
    for c in axiom:
        if c in idict:
            if depth >= maxdepth:
                v = idict[c]
            else:
                v = _iter_expand(idict[c], idict, depth+1, maxdepth)
            olist.append(v)
        else:
            olist.append(c)
    return "".join(olist)

def angle_360_2pi(x: float) -> float:
    return x / 360.0 * TWO_PI

def expand_Lcmd(cmd: str, maxdepth=4) -> str:
    """
    The first line is axiom
    separated by ';'
    ```
    B;
    B=A+A--A+A;
    A=F+F--F+F;
    ```
    """
    lines = cmd.split(';')
    axiom = lines[0].strip()
    idict = {}

    for line in lines[1::]:
        line = line.strip()
        if len(line) == 0:
            continue
        item = line.split('=')
        c = item[0].strip()
        assert len(c) == 1 and c.isalpha(
        ), "only one [a-zA-Z] char is supported: %s" % c
        idict[c] = item[1].strip()

    return _iter_expand(axiom, idict, 1, maxdepth)


class LSys:
    def __init__(
            self,
            cmd: str,  # L cmd
            max_expand: int = 4,  # max iteration for expand cmd
            angle: float = np.pi / 3.0,  # (pi / 3) 60 degree
            linelen: float = 0.01,  # basic line length
            loc_start: tuple = (0.5, 0.0),  # start location
            angle_delta: float = np.pi/6.0,  # pi / 6
            linescale: float = 0.6,  # line length scale factor for '<' and '>'
            lineinc: float = 0.01,  # line size increment factor
            linewidth: float = 1.0,  # basic line width
    ) -> None:
        """
        Lsystem:

        """
        self.cmd = cmd
        self.expand_max_iter = max_expand
        self.expanded_cmd = ""

        self.angle = _force_0_to_2pi(angle)
        self.angle_delta = _force_0_to_2pi(angle_delta)
        self.loc_start = loc_start
        self.linescale = linescale
        self.lineinc = lineinc
        self.linelen = linelen
        self.linewidth = linewidth
    
    def expand_cmd(self):
        self.expanded_cmd = expand_Lcmd(self.cmd, self.expand_max_iter)

    def step(self,
             loc: tuple = None,
             angle: float = None,
             linelen: float = None,
             linewidth: float = None):
        """
        process the L-system
        - loc: start point
        """
        if loc is None:
            x, y = self.loc_start
        else:
            x, y = loc

        if angle is None:
            angle = self.angle

        if linelen is None:
            linelen = self.linelen

        if linewidth is None:
            linewidth = self.linewidth
        
        if self.expanded_cmd == "":
            self.expand_cmd()

        Lstart = []
        Lend = []
        Lwidth = []
        Lcirc = []
        Lradius = []

        cacheStack = []

        turning_angle = self.angle

        angle_sign = 1.0

        pstart = (0, 0)

        for c in self.expanded_cmd:
            if c == 'F':
                # move forward forward
                Lstart.append((x, y))
                x = x + linelen * np.cos(angle)
                y = y + linelen * np.sin(angle)
                Lend.append((x, y))
                Lwidth.append(linewidth)
            elif c == 'f':
                # move forward by line length without drawing a line
                x = x + linelen * np.cos(angle)
                y = y + linelen * np.sin(angle)
            elif c == '+':
                # turn left
                angle -= angle_sign * turning_angle
            elif c == '-':
                # turn right
                angle += angle_sign * turning_angle
            elif c == '|':
                # reverse direction
                angle += np.pi
            elif c == '[':
                # push current drawing state onto stack
                cacheStack.append((x, y, angle, linelen, linewidth))
            elif c == ']':
                # pop current drawing state from stack
                x, y, angle, linelen, linewidth = cacheStack.pop(-1)
            elif c == '#':
                # increment the line with by line width increment
                linewidth += self.lineinc
            elif c == '!':
                # decrease the line width by line width increment
                linewidth -= self.lineinc
            elif c == '@':
                # draw a dot with line width radius
                Lcirc.append((x, y))
                Lradius.append(linelen)
            elif c == '{':
                # open a polygon
                pstart = (x, y)
            elif c == '}':
                # close a polygon and fill it with fill color
                Lstart.append(pstart)
                Lend.append((x, y))
                Lwidth.append(linelen)
            elif c == '>':
                # multiply the line length by scalar
                linelen *= self.linescale
            elif c == '<':
                # divide the line length by scalar
                linelen /= self.linescale
            elif c == '&':
                # swap the meaning of '+' and '-'
                angle_sign *= -1.0
            elif c == '(':
                # decrease turring angle
                turning_angle -= self.angle_delta
            elif c == ')':
                # increase turning angle
                turning_angle += self.angle_delta
            else:
                continue
                # print(f"character {c} is not supported")

        return np.asarray(Lstart), np.asarray(Lend), np.asarray(Lwidth), np.asarray(Lcirc), np.asarray(Lradius)
