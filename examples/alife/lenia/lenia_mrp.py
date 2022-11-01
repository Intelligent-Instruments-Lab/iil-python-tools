"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

import numpy as np
from iipyper import OSC, run, repeat, cleanup
import taichi as ti
from lenia import Lenia
from mrp import MRP

def main(host="127.0.0.1", port=7563):

    osc = OSC(host, port, verbose=False)
    osc.create_client("mrp", host="127.0.0.1", port=7770)
    mrp = MRP(osc)
    notes = [
        24, 29, 31, 
        36, 41, 43, 
        48, 53, 55, 
        60, 65, 67, 
        72, 77, 79, 
        84
    ]
    mrp.notes_on(notes)
    # [mrp.note_on(n) for n in notes]
    [mrp.quality_update(n, 'intensity', 1) for n in notes]

    # @osc.args("/*")
    # def _(address, *args):
    #     print(f"{address} {args}")

    # @repeat(1)
    # def _():
    #     osc("max", "/max", *arr)

    res = 256
    scatter = 4

    for i,n in enumerate(notes):
        print('world slice indexes:',res,i,n,int(res*(i/len(notes))))

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

    # @repeat(1)
    # def _():
    #     lenia.cursor[1] = 50
    #     lenia.cursor[0] = 50
    #     lenia.draw()
    #     lenia.render()
    #     canvas.set_image(lenia.pixels)

    # @repeat(0.016)
    # def _():
    #     lenia.cursor[1] = 50
    #     lenia.cursor[0] = 50
    #     lenia.draw()
    #     lenia.cursor[1] = 10
    #     lenia.cursor[0] = 10
    #     lenia.draw()
    #     canvas.set_image(lenia.pixels)
    #     lenia.update()
    #     window.show()

    @repeat(0.005)
    def _():
        for e in window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            elif e.key == 'p':
                lenia.paused = not lenia.paused
                print('Pause state:{}'.format(lenia.paused))
            elif e.key == 'r':
                lenia.init()
                print("Reset world")
            elif e.key == 's':
                lenia.save_world()
                print("Saved current world")
                lenia.render()
            elif e.key == 'l':
                lenia.load_world()
                print("Loaded saved world")
                lenia.render()

        if window.is_pressed(ti.ui.LMB):
            lenia.cursor[1] = window.get_cursor_pos()[1]
            lenia.cursor[0] = window.get_cursor_pos()[0]
            lenia.draw()
            lenia.render()
        elif window.is_pressed(ti.ui.RMB):
            lenia.cursor[1] = window.get_cursor_pos()[1]
            lenia.cursor[0] = window.get_cursor_pos()[0]
            lenia.erase()
            lenia.render()

        canvas.set_image(lenia.pixels)

        window.GUI.begin("Taichi Lenia", 0.01, 0.01, 0.6, 0.15)
        window.GUI.text("LB press: draw, RB press clear")
        window.GUI.text("r : Reset, SPACE : pause")
        window.GUI.text("S : save, L : load")
        # lenia.conv_r = window.GUI.slider(
        #     "Convolution kernel radius",
        #     lenia.conv_r, 5, 40
        # )
        # lenia.time = window.GUI.slider(
        #     "time step",
        #     lenia.time, 1, 20
        # )
        lenia.grow_miu[None] = window.GUI.slider_float("Growth function miu",
                                                       lenia.grow_miu[None],
                                                       0.01, 0.30)
        lenia.grow_sig[None] = window.GUI.slider_float("Growth function sigma",
                                                       lenia.grow_sig[None],
                                                       0.001, 0.0300)
        lenia.brush[None] = window.GUI.slider_float("Brush radius",
                                                    lenia.brush[None], 0.01,
                                                    0.06)
        window.GUI.end()

        if not lenia.paused:
            lenia.update()
            for i,n in enumerate(notes):
                lenia.slice_world(int(res*(i/len(notes))))
                world_slice = lenia.world_slice.to_numpy().astype(float)
                mrp.quality_update(n, 'harmonics_raw', world_slice)

        window.show()

    @cleanup
    def _():
        mrp.cleanup()

if __name__=='__main__':
    run(main)
