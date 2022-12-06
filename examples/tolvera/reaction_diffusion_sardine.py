from sardine import *
import taichi as ti
import numpy as np
import math
import tolvera as tol
from pysensel import SenselOSC

def handle_contact(*args):
    sensel.handle_contact(args)

sensel = SenselOSC(7006, handle_contact)
ti.init(arch=ti.vulkan)
c.bpm = 250
c.link()
x = 1920
y = 1080
rea_diff = tol.vera.ReactionDiffusion(x, y)
window = ti.ui.Window("Reaction Diffusion", (x, y))
canvas = window.get_canvas()

@swim
def gui_loop(d=0.5, i=0):
    canvas.set_image(rea_diff.process())
    window.show()
    a(gui_loop, d=1/16, i=i+1)

@swim
def param_loop(d=16, i=0):
    _x0 = sensel.contacts[0]['x']
    _y0 = sensel.contacts[0]['y']
    _x1 = sensel.contacts[1]['x']
    _y1 = sensel.contacts[1]['y']
    dv = _x0 * 0.160+0.08
    du = _y0 * 0.080+0.04
    feed = _x0 * 0.060+0.03
    kill = _y0 * 0.062+0.03
    # rea_diff.Du[None] = _x0 # P('0.5+0.5*sin($/3)*0.160+0.01')# 0.160 # || 210
    # rea_diff.Dv[None] = _y0 # P('0.5+0.5*sin($/5)*0.080+0.01')# 0.080 # || 105
    rea_diff.feed[None] = feed #sensel.contacts[0]['f'] # P('0.5+0.5*sin($/7)*0.060+0.01')# 0.060 # || 018
    rea_diff.kill[None] = kill #sensel.contacts[0]['a'] # P('0.5+0.5*sin($/9)*0.062+0.01')# 0.062 # || 051
    print(round(feed,3), round(kill,3))
    rea_diff.substep[None] = 16 # || 64
    a(param_loop, d=1, i=i+1)

rea_diff.randomise()

hush(param_loop)

rea_diff.Du[None]      = 0.160 # min: 0.14 || 210
rea_diff.Dv[None]      = 0.080 # max: 0.8 || 105
rea_diff.feed[None]    = 0.06 # || 018
rea_diff.kill[None]    = 0.062 # || 051
rea_diff.substep[None] = 32 # || 64