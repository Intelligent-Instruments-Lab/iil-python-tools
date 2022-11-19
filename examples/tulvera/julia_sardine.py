from sardine import *
import random
from julia import JuliaSet
import taichi as ti

size = 512
juliaSet = JuliaSet(size)
gui = ti.GUI("Julia Set", res=(size * 2, size), fast_gui=True)

@swim
def main(d=0.05, idx=0.0, zoom=1.0):
    juliaSet.paint(idx, zoom)
    gui.set_image(juliaSet.pixels)
    gui.show()
    again(main,
        idx=idx+random.random(), 
        zoom=zoom*(2**(random.random()-0.5))
        )
