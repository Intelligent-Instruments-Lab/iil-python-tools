# RAVE explorer idea:
#  []: generate random constellation
#  []: recenter at current position
#  []: store z -> []: store control

"""
Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

import iml as iml_module
from iml import IML
from iipyper import TUI, Audio, OSC, run, _lock
import numpy as np
from textual import events
import torch

import sounddevice as sd
import sys
# import contextlib

from rich.panel import Panel
from textual.widgets import Header, Footer, Static, Button, TextLog, Sparkline
from textual.containers import Container
from textual.reactive import reactive

# class IMLState(Static):
#     value = reactive(([0],[0]))
#     def __init__(self, **kw):
#         super().__init__(**kw)
#         self.src = Sparkline()
#         self.tgt = Sparkline()

#     def compose(self):
#         yield self.src
#         yield self.tgt

#     def watch_value(self, time: float) -> None:
#         src, tgt = self.value
#         self.src.data = src
#         self.tgt.data = tgt

class CtrlPad(Container):
    # value = reactive((16,16))
    # def watch_value(self, time: float) -> None:
    #     x, y = self.value
    #     s = ''
    #     for r in range(32):
    #         for c in range(32):
    #             if r==x and c==y:
    #                 s += '*'
    #             else:
    #                 s += ' '
    #         s += '\n'
    #     self.update(Panel(s))
    def mouse_move(self, event):
        pass

    def on_mount(self) -> None:
        self.capture_mouse()

    def on_mouse_move(self, event):
        self.mouse_move(event)
        self.screen.query_one('#std_log').write(event)
        self.query_one(Pointer).offset = event.offset - (1, 1)

class Pointer(Static):
    pass

class IMLState(Static):
    value = reactive((None,None))
    def watch_value(self, time: float) -> None:
        src, tgt = self.value
        self.update(Panel(f'src: {src}\ntgt: {tgt}', title='state'))

class IMLTUI(TUI):
    CSS_PATH = 'tui.css'

    BINDINGS = [
        ("r", "randomize", "randomize around current mapping"),
        ("z", "zero", "reset to zero vector"),
        ("s", "store", "store a source / target"),
    ]

    def __init__(self):
        super().__init__()
        self.ctrl_pad = CtrlPad(Pointer("O"))

    def compose(self):
        """Create child widgets for the app."""
        yield Header()
        yield self.std_log
        yield self.ctrl_pad
        yield IMLState(id='state')
        yield Footer()

def main(
        # osc_host="127.0.0.1", osc_port=9999,
        device=None,
        rave_path=None, checkpoint=None):
    
    # osc = OSC(osc_host, osc_port)
    
    rave = torch.jit.load(rave_path)

    d_src = 2
    d_tgt = rave.encode_params[2]

    try:
        sr = rave.sr
    except Exception:
        sr = rave.sampling_rate

    tui = IMLTUI()
    # sys.stdout = tui
    # sys.stderr = tui
    print = iml_module.print = tui.print

    ctrl = torch.zeros(d_src)
    z = torch.zeros(d_tgt)
    def rave_callback(
            indata: np.ndarray, outdata: np.ndarray, #[frames x channels]
            frames: int, time, status):
        with torch.inference_mode():
            # indata = torch.from_numpy(indata).float()
            # indata = indata.mean(-1)[None,None]
            # outdata[:,:] = rave(indata)[:,0].T
            # ctrl.mul_(0.99)
            # ctrl.add_(torch.randn(d_src)*0.05)
            # z[:] = torch.from_numpy(iml.map(ctrl, k=5))#.float()
            tui(state=(
                ' '.join(f'{x.item():+0.2f}' for x in ctrl),
                ' '.join(f'{x.item():+0.2f}' for x in z)))
            # z = torch.from_numpy(z).float()[None,:,None]
            # outdata[:,:] = rave.decode(z)[:,0].T
            outdata[:,:] = rave.decode(z[None,:,None])[:,0].T
            # print(outdata)
        
    audio = Audio(
        device=device, dtype=np.float32,
        samplerate=sr, blocksize=rave.encode_params[-1], 
        callback=rave_callback)
    
    iml = IML(d_src)

    # @tui.set_mouse_move
    # def on_mouse_move(self, event):
    #     print(event)
    #     tui.query_one(Pointer).offset = event.offset - (1, 1)

    def mm(event):
        ctrl[0] = (event.x - 64) / 30
        ctrl[1] = (event.y - 15) / 5
        z[:] = torch.from_numpy(iml.map(ctrl, k=5))#.float()
    tui.ctrl_pad.mouse_move = mm

    # TODO: button to fix current neighbors
    # or record a gesture and fix those neighbors
    # draw locations of current sources

    @tui.set_action
    def randomize():
        # keep the current nearest neighbors and rerandomize the rest
        print('randomize:')
        k = 5
        if len(iml.pairs):
            srcs, tgts, scores = iml.search(ctrl, k=k)
            max_score = max(scores)
            iml.reset()
            for s,t in zip(srcs,tgts):
                iml.add(s,t)
        else:
            max_score = 0

        while(len(iml.pairs) < 32):
            src = torch.randn(d_src)#/(ctrl.abs()/2+1)
            if iml.neighbors.distance(ctrl, src) < max_score:
                continue
            tgt = z + torch.randn(d_tgt)*2#/(z.abs()/2+1)
            iml.add(src, tgt)

    ###
    randomize()
    # print('add')
    # for _ in range(32):
    #     iml.add(torch.randn(d_src), torch.randn(rave.cropped_latent_size)*2)

    audio.stream.start()

    tui.run()


if __name__=='__main__':
    run(main)