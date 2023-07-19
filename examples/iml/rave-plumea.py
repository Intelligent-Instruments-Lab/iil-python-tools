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

    def compose(self):
        """Create child widgets for the app."""
        yield Header()
        yield self.std_log
        yield IMLState(id='state')
        yield Footer()

def main(
        osc_host='', osc_port=7562, osc_return=8888,
        device=None, gain=1,
        block_size=None, # disabled
        rave_path=None, checkpoint=None):

    osc = OSC(osc_host, osc_port)

    rave = torch.jit.load(rave_path)

    d_src = 6
    d_tgt = rave.encode_params[2] - 1

    try:
        sr = rave.sr
    except Exception:
        sr = rave.sampling_rate

    tui = IMLTUI()
    print = iml_module.print = tui.print

    ctrl = torch.zeros(d_src)
    z = torch.zeros(d_tgt+1)
    def rave_callback(
            indata: np.ndarray, outdata: np.ndarray, #[frames x channels]
            frames: int, time, status):
        with torch.inference_mode():
            tui(state=(
                ' '.join(f'{x.item():+0.2f}' for x in ctrl),
                ' '.join(f'{x.item():+0.2f}' for x in z)))
            outdata[:,:] = gain*rave.decode(z[None,:,None])[:,0].T
            # print(outdata)

    audio = Audio(
        device=device, dtype=np.float32,
        samplerate=sr, blocksize=block_size or rave.encode_params[-1],
        callback=rave_callback)

    iml = IML(d_src, interp='ripple')

    # @tui.set_mouse_move
    # def on_mouse_move(self, event):
    #     print(event)
    #     tui.query_one(Pointer).offset = event.offset - (1, 1)

    # def mm(event):
        # ctrl[0] = (event.x - 64) / 30
        # ctrl[1] = (event.y - 15) / 5
        # z[:] = torch.from_numpy(iml.map(ctrl, k=5, ripple=7))#.float()
    # tui.ctrl_pad.mouse_move = mm


    @tui.set_action
    def randomize():
        # keep the current nearest neighbors and rerandomize the rest
        print('randomize:')
        k = 5
        if len(iml.pairs):
            srcs, tgts, scores = iml.search(ctrl, k=k)
            max_score = max(scores)
            iml.reset(keep_near=ctrl, k=k)
            # for s,t in zip(srcs,tgts):
                # iml.add(s,t)
        else:
            max_score = 0

        while(len(iml.pairs) < 128):
            src = torch.rand(d_src)/2 #/(ctrl.abs()/2+1)
            if iml.neighbors.distance(ctrl, src) < max_score:
                continue
            tgt = z[1:] + torch.randn(d_tgt)*2#/(z.abs()/2+1)
            iml.add(src, tgt)

    ###
    randomize()

    controls = dict(
        value01=0, value02=1,
        value03=2, value04=3,
        value05=4, value06=5,
        #value07=6, value08=7,
        #value09=8, value10=9,
        #value11=10, value12=11,
        #value13=12, value14=13,
        #value15=14

    )
    @osc.args('/*')
    def _(k, v):
        k = k.split('/')[1]
        if k=='value00':
            z[0] = max(-2, 2 - 8*v)
        if k in controls:
            ctrl[controls[k]] = v**0.5
        z[1:] = torch.from_numpy(iml.map(ctrl, k=5, ripple=7))
        # print(k, v)
        # print(controls)

    audio.stream.start()

    tui.run()


if __name__=='__main__':
    run(main)
