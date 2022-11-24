from sardine import *
import taichi as ti
import numpy as np
import math
import tulvera as tul

ti.init(arch=ti.vulkan)
c.bpm = 250
c.link()
x = 1920
y = 1080
n = 8192
boids = tul.vera.BoidsP(x, y, n)
boids.pause = False
window = ti.ui.Window("BoidsP", (x, y))
canvas = window.get_canvas()

@swim
def gui_loop(d=0.5, i=0):
    boids.process()
    canvas.set_image(boids.world.to_numpy()[0])
    window.show()
    a(gui_loop, d=1/16, i=i+1)

@swim
def param_loop(d=16, i=0):
    boids.radius[None] = P('50*cos($/3)')
    boids.dt[None]     = P('2*cos($/9)')
    boids.speed[None]  = P('2*sin($/7)')
    boids.separate[None] = P('0.5+0.5*sin($/5)')
    boids.align[None]    = P('0.5+0.5*cos($/3)')
    boids.cohere[None]   = P('0.5+0.5*sin($/4)')
    a(param_loop, d=1, i=i+1)

hush(param_loop)

boids.randomise()

boids.separate[None] = 0
boids.align[None]    = 1
boids.cohere[None]   = 1
boids.dt[None]     = 2
boids.radius[None] = 40.0
boids.speed[None]  = 3.0

