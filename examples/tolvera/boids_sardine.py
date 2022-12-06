from sardine import *
import taichi as ti
import numpy as np
import math
import tolvera as tol

ti.init(arch=ti.vulkan)
c.bpm = 250
c.link()
x = 1920
y = 1080
# boids = tol.vera.Boids(x, y, 1024, colormode='g', species=1)
# boids = tol.vera.Boids(x, y, 4096, colormode='g', species=1)
# boids = tol.vera.Boids(x, y, 4096, colormode='g', species=3)
# boids = tol.vera.Boids(x, y, 4096, colormode='rgb', species=3)
boids = tol.vera.Boids(x, y, 16384, colormode='rgb', species=20)
# boids = tol.vera.Boids(x, y, 8192, colormode='rgb', species=5)
window = ti.ui.Window("Boids", (x, y))
canvas = window.get_canvas()

@swim
def gui_loop(d=0.5, i=0):
    canvas.set_image(boids.process())
    window.show()
    a(gui_loop, d=1/16, i=i+1)

boids.separate[None] =0.1
boids.align[None]    =1.0
boids.cohere[None]   =1.0
boids.fear[None]     =10.0
boids.dt[None]       =2
boids.radius[None]   =20.0
boids.speed[None]    =3.0
boids.size[None] = 1

@swim
def param_loop(d=16, i=0):
    boids.radius[None] = P('50*cos($/3)')
    boids.dt[None]     = P('2*cos($/9)')
    boids.speed[None]  = P('2*sin($/7)')
    boids.separate[None] = P('0.5+0.5*sin($/5)')
    boids.align[None]    = P('0.5+0.5*cos($/3)')
    boids.cohere[None]   = P('0.5+0.5*sin($/4)')
    boids.fear[None]     = P('4+4*sin($/4)')
    a(param_loop, d=8, i=i+1)

hush(param_loop)

boids.randomise()

