import taichi as ti
from PIL import Image
import numpy as np

from plyread2 import NPLYReader

ply = NPLYReader("./mesh/main-all.ply")

img = Image.open("./mesh/main-2-big.png").rotate(-90).resize((512,512))
imgnp = np.asarray(img)

gui = ti.GUI("wiew UV", res=(512,512))


while gui.running:

    gui.set_image(imgnp)

    for f in ply.faces:
        gui.line(ply.vertex[f[0], 6:8], ply.vertex[f[1], 6:8], color=0xaaaaaa)
        gui.line(ply.vertex[f[1], 6:8], ply.vertex[f[2], 6:8], color=0xaaaaaa)
        gui.line(ply.vertex[f[2], 6:8], ply.vertex[f[0], 6:8], color=0xaaaaaa)
    
    gui.show()

