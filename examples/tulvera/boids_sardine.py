from sardine import *
import taichi as ti
import numpy as np
import math
import tulvera as tul

ti.init(arch=ti.vulkan)
c.bpm = 250
c.link()
resx = 1920
resy = 1080
n = 8192
boids = tul.vera.Boids(resx, resy, n)
window = ti.ui.Window("Boids", (resx, resy))
canvas = window.get_canvas()

@swim
def gui_loop(d=0.5, i=0):
    boids.update()
    canvas.set_image(boids.world.to_numpy().astype(np.uint8))
    window.show()
    a(gui_loop, d=1/16, i=i+1)

@swim
def param_loop(d=16, i=0):
    boids.vis_radius[None] = P('40.0,80.0,150.0',i)
    # boids.max_speed[None] = P('1.0,2.0,3.0',i)
    boids.max_speed[None] = P('2*sin($/2)')
    a(param_loop, d=8, i=i+1)

hush()

boids.dt[None]           = 0.5
boids.vis_radius[None]   = 100.0
boids.max_speed[None]    = 10.0
boids.boids_radius[None] = 2
