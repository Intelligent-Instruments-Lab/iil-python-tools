import taichi as ti
import tulvera as tul
import numpy as np

def main():
    ti.init(arch=ti.vulkan)
    resx = 1920
    resy = 1080
    n = 8192
    boids = tul.vera.Boids(resx, resy, n)
    window = ti.ui.Window("Boids", (resx, resy))
    canvas = window.get_canvas()
    while window.running:
        boids.update()
        canvas.set_image(boids.world.to_numpy().astype(np.uint8))
        window.show()

if __name__ == '__main__':
    main()
