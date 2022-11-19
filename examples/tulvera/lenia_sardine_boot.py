# 1. boot + import
from sardine import *
import taichi as ti
import sys
sys.path.append('/Users/jack/Documents/git/pro/iil/py/iil-python-tools/examples/alife/lenia')
from lenia import Lenia

# 2. setup
res = 256
scatter = 4
draw = False
erase = False
window = ti.ui.Window("Taichi-Lenia", (res * scatter, res * scatter))
canvas = window.get_canvas()
lenia = Lenia(res=res,
             scatter=scatter,
             conv_r=20,
             time=10,
             miu=0.15,
             sig=0.016,
             kr=1,
             kb=ti.Vector([1]))
lenia.init()

def lenia_world_save(*args):
    lenia.save_world()

def lenia_world_load(*args):
    lenia.load_world()

def lenia_pause(*args):
    lenia.set_pause(args[0])

def lenia_reset(*args):
    lenia.init()

def lenia_time(*args):
    lenia.set_time(args[0])

def lenia_convr(*args):
    lenia.set_convr(args[0])

def lenia_grow_miu(*args):
    lenia.set_grow_miu(args[0])

def lenia_grow_sig(*args):
    lenia.set_grow_sig(args[0])

def lenia_set_brush(*args):
	lenia.set_brush(*args)

# def lenia_world_slice(*args):
#     lenia.slice_world(int(0)) # args[0]
#     world_slice = lenia.world_slice.to_numpy().astype(float)

def lenia_sample(*args):
    lenia.sample(*args)

receive = Receiver(7563)
receive.attach('/lenia/world/save', lenia_world_save)
receive.attach('/lenia/world/load', lenia_world_load)
receive.attach('/lenia/pause', lenia_pause)
receive.attach('/lenia/reset', lenia_reset)
receive.attach('/lenia/time', lenia_time)
receive.attach('/lenia/convr', lenia_convr)
receive.attach('/lenia/grow/miu', lenia_grow_miu)
receive.attach('/lenia/grow/sig', lenia_grow_sig)
receive.attach('/lenia/brush', lenia_set_brush)

c.bpm = 250
c.link()

@swim
def gui_loop(d=0.5):
    canvas.set_image(lenia.pixels)
    # canvas.lines(vertices, width, indices, color, per_vertex_color)
    # window.GUI.begin("Taichi Lenia", 0.01, 0.01, 0.6, 0.15)
    # window.GUI.end()
    match lenia.brush_mode:
        case "draw":
            lenia.draw()
            lenia.render()
        case "erase":
            lenia.erase()
            lenia.render()
    if not lenia.paused:
        lenia.update()
    window.show()
    a(gui_loop, d=1/16)
