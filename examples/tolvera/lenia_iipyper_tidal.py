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

def main(host="127.0.0.1", port=7563):
    # osc
    osc = OSC(host, port, verbose=False)
    osc.create_client("tidal", host="127.0.0.1", port=7564)
    gui_update_rate = 0.005

    # lenia
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

    @osc.args("/lenia/*")
    def _(address, *args):
        nonlocal draw, erase
        # print(f"{address} {args}")
        address = address.split("/")
        match address[2]:
            case "world":
                match address[3]:
                    case "save":
                        lenia.save_world()
                    case "load":
                        lenia.load_world()
                    case _:
                        print("bad msg:", f"{address} {args}")
            case "gui": # 0.005-?
                gui_update_rate = args[0]
            case "pause":
                lenia.set_pause(args[0])
            case "reset":
                lenia.init()
            case "time": # 1-20
                lenia.set_time(args[0])
            case "convr": # 5-40
                lenia.set_conv_r(args[0])
            case "grow":
                match address[3]:
                    case "miu": # 0.01-0.30
                        lenia.set_grow_miu(args[0])
                    case "sig": # 0.001-0.030
                        lenia.set_grow_sig(args[0])
                    case _:
                        print("bad msg:", f"{address} {args}")
            case "brush":
                match args[0]:
                    case "draw":
                        draw = True
                        erase = False
                    case "erase":
                        erase = True
                        draw = False
                    case _:
                        draw = False
                        erase = False
                lenia.cursor[0] = args[2] # 0-1
                lenia.cursor[1] = args[3] # 0-1
                # lenia.set_brush(args[0], args[1], args[2], args[3], args[4])
            # case "sample":
            #     match address[3]:
            #         case "row":    # x
            #         case "column": # y
            #         case "point":  # x1, y1
            #         case "line":   # x1, y1, x2, y2
            #         case "rect":   # x1, y1, x2, y2
            #         case "circle": # x1, y1, r
            #         case "show":   # bool
            #         case _:
            #             print("bad msg:", f"{address} {args}")
            case _:
                print("bad msg:", f"{address} {args}")

    @repeat(gui_update_rate)
    def _():
        canvas.set_image(lenia.pixels)
        window.GUI.begin("Taichi Lenia", 0.01, 0.01, 0.6, 0.15)
        window.GUI.end()
        if draw == True:
            lenia.draw()
            lenia.render()
        if erase == True:
            lenia.erase()
            lenia.render()
        if not lenia.paused:
            lenia.update()
        window.show()

    # @cleanup
    # def _():
    #     lenia.cleanup()

if __name__=='__main__':
    run(main)
