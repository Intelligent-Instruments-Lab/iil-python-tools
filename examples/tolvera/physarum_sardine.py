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
n = 2048
physarum = tol.vera.Physarum(x, y, n)
window = ti.ui.Window("Physarum", (x, y))
canvas = window.get_canvas()

@swim
def gui_loop(d=0.5, i=0):
    canvas.set_image(1-physarum.process())
    window.show()
    a(gui_loop, d=1/16, i=i+1)

@swim
def param_loop(d=16, i=0):
    physarum.sense_angle[None]    = P('1+0.5*sin($/3)*3.14') # 0.20 * np.pi
    # physarum.sense_dist[None]     = P('4+2*sin($/4)') # 4.0
    # physarum.evaporation[None]    = P('1+0.49*sin($/2)') # 0.95
    # physarum.move_angle[None]     = P('1+0.5*sin($/2)*3.14') # 0.1 * np.pi
    # physarum.move_step[None]      = P('1,2,4,3,5,3',i) # 2.0
    physarum.step_per_frame[None] = P('1,10,9,4,3,7,5,8',i) # 10
    a(param_loop, d=8, i=i+1)

hush(param_loop)

physarum.sense_angle[None]    = 0.40 * np.pi
physarum.sense_dist[None]     = 10.0
physarum.evaporation[None]    = 0.99 # 0-1
physarum.move_angle[None]     = 0.1 * np.pi
physarum.move_step[None]      = 0.5
physarum.step_per_frame[None] = 10
