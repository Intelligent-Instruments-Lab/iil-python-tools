import taichi as ti
import numpy as np
import tolvera as tol
from pysensel import TiSensel

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 4096
    boids = tol.vera.Boids(x, y, n)
    ti_sensel = TiSensel(x, y)
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()
    while window.running:
        # update(boids) # jurigged: costs 10fps
        ti_sensel.process()
        canvas.set_image(boids.process())
        # canvas.set_image(ti_sensel.process())
        window.show()

if __name__ == '__main__':
    main()
