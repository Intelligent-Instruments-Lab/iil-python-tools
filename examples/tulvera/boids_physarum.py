import taichi as ti
import numpy as np
ti.init(arch=ti.vulkan)
import sys
sys.path.append('/Users/jack/Documents/git/pro/iil/py/iil-python-tools/examples/alife/systems')

from boids import Boids
from physarum import Physarum

def main():
    ti.init(arch=ti.vulkan)
    res = 1080
    n = 1024
    boids = Boids(res, res, n)
    physarum = Physarum(res, n)
    window = ti.ui.Window("Boids", (res, res))
    canvas = window.get_canvas()
    while window.running:
        boids.update()
        physarum.update()
        boids.world # ti.Vector.field(3, dtype=ti.i32, shape=(self.resx, self.resy)) .astype(np.uint8)
        physarum.world # ti.field(dtype=ti.f32, shape=[2, self.GRID_SIZE, self.GRID_SIZE]) [0]
        canvas.set_image(boids.world.to_numpy().astype(np.uint8))
        canvas.set_image(physarum.world.to_numpy()[0])
        window.show()

if __name__ == '__main__':
    main()
