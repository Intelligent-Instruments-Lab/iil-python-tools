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

class IMLState(Static):
    value = reactive((None,None))
    def watch_value(self, time: float) -> None:
        src, tgt = self.value
        self.update(Panel(f'src: {src}\ntgt: {tgt}', title='state'))

class IMLTUI(TUI):
    CSS_PATH = 'tui.css'

    BINDINGS = [
        ("r", "randomize_all", "randomize full mapping"),
        ("n", "randomize_nonlocal", "randomize nonlocal mapping"),
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
        rave_path=None, checkpoint=None,
        n_neighbors = 5, # number of neighbors in mapping
        n_points = 256, # total points in mapping
        d_src = 6, # number of last control (e.g., value06) being sent by OSC
        ):

    osc = OSC(osc_host, osc_port)

    rave = torch.jit.load(rave_path)

    block_size = rave.encode_params[-1]

    d_out = rave.encode_params[2]
    d_tgt = (d_out - 1) * 2

    try:
        sr = rave.sr
    except Exception:
        sr = rave.sampling_rate

    tui = IMLTUI()
    print = iml_module.print = tui.print

    src = torch.zeros(d_src)
    z = torch.zeros(d_out)
    z_freq = torch.zeros(d_tgt//2)
    z_phase = torch.zeros(d_tgt//2)
    z_mean = torch.zeros(d_tgt//2)
    def rave_callback(
            indata: np.ndarray, outdata: np.ndarray, #[frames x channels]
            frames: int, time, status):
        with torch.inference_mode():
            z_phase[:] = z_phase + z_freq * (block_size / sr * 2 * np.pi)
            z[1:] = z_mean #+ z_phase.sin()
            outdata[:,:] = gain*rave.decode(z[None,:,None])[:,0].T

            tui(state=(
                ' '.join(f'{x.item():+0.2f}' for x in src),
                ' '.join(f'{x.item():+0.2f}' for x in z)))
            # print(outdata)

    audio = Audio(
        device=device, dtype=np.float32,
        samplerate=sr, blocksize=block_size,
        callback=rave_callback)

    iml = IML(d_src, interp='softmax')

    def rand_src():
        return torch.rand(d_src)
    
    def rand_tgt():
        return torch.cat((
            torch.randn_like(z_mean)*2,
            (torch.randn_like(z_mean)+1).exp()
            ))

    @tui.set_action
    def randomize_nonlocal():
        # keep the current nearest neighbors and rerandomize the rest
        print('randomize nonlocal points:')
        if len(iml.pairs):
            _, _, scores = iml.search(src, k=n_neighbors)
            max_score = max(scores) # furthest neighbor of current src
            iml.reset(keep_near=src, k=n_neighbors)
        else:
            max_score = 0

        while(len(iml.pairs) < n_points):
            new_src = rand_src()
            if iml.neighbors.distance(src, new_src) < max_score:
                continue
            tgt = torch.cat((z_mean, z_freq)) + rand_tgt()
            iml.add(new_src, tgt)

    @tui.set_action
    def randomize_all():
        # randomize all points in mapping
        print('randomize all points:')
        iml.reset()
       
        while(len(iml.pairs) < n_points):
            iml.add(rand_src(), rand_tgt())

    ###
    randomize_all()

    # maps OSC addresses to vector indices
    control_names = dict(
        value01=0, value02=1,
        value03=2, value04=3,
        value05=4, value06=5,
        value07=6, value08=7,
        value09=8, value10=9,
        value11=10, value12=11,
        value13=12, value14=13,
        value15=14, value16=15
    )
    @osc.args('/*')
    def _(k, v):
        # shave '/' off of OSC address
        k = k.split('/')[1]

        # special handling for addresses not in control_names:
        if k=='value00':
            # directly control first latent dimension
            z[0] = max(-2, 2 - 8*v)

        # pack the rest into a vector which will go into the IML mapping
        if k in control_names:
            idx = control_names[k]
            if idx > d_src:
                raise ValueError("""
                pluma: ERROR: unexpected OSC control.
                increase --d-src argument when running script?
                """)
            src[idx] = v**0.5

        # run through the mapping
        tgt = torch.from_numpy(iml.map(src, k=n_neighbors))
        z_mean[:], z_freq[:] = tgt.chunk(2)
        # print(k, v)

    audio.stream.start()

    tui.run()


if __name__=='__main__':
    run(main)
