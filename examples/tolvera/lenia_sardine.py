from sardine import *
import taichi as ti
import numpy as np
import math
import tolvera as tol

ti.init(arch=ti.vulkan)
c.bpm = 250
c.link()
x = 1024
y = 1024
lenia = tol.vera.Lenia(res=x,
            scatter=4,
            conv_r=20,
            time=10,
            miu=0.15,
            sig=0.016,
            kr=1,
            kb=ti.Vector([1]))
lenia.init()
lenia.set_pause(False)
window = ti.ui.Window("Lenia", (x, y))
canvas = window.get_canvas()

@swim
def gui_loop(d=0.5, i=0):
    canvas.set_image(lenia.pixels)
    # lenia.set_brush(P("draw, erase", i), 0.1, P('0.1,0.5,0.9',i), 0.5)
    window.show()
    match lenia.brush_mode:
        case "draw":
            lenia.draw()
            lenia.render()
        case "erase":
            lenia.erase()
            lenia.render()
    if not lenia.paused:
        lenia.update()
    a(gui_loop, d=1/16, i=i+1)

@swim
def param_loop(d=0.5, i=0):
    lenia.set_brush(P("draw, erase", i), 0.1, P('0.5+0.5*sin($/3)', i), P('0.5+0.5*sin($/4)', i))
    # lenia.grow_miu[None] = P("0.01, 1.0, 0.15, 0.3, 0.5, 0.4",i) # 0.15
    # lenia.grow_sig[None] = P("0.016, 0.1, 0.01",i) # 0.016
    a(param_loop, d=1/16, i=i+1)

hush(param_loop)

lenia.set_pause(False)

lenia.set_brush('draw', 0.1, 0.1, 0.5)
lenia.set_time(20)
lenia.set_conv_r(30)
lenia.grow_miu[None] = 0.03 # 0.15
lenia.grow_sig[None] = 0.015 # 0.016

hush()
