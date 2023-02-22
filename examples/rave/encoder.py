"""
RAVE encoder iipyper app.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

import math

import torch
import numpy as np

from iipyper import Audio, MIDI, run, Stopwatch, repeat, cleanup, TUI, profile
from typing import Dict

import rich
from rich.align import Align
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Button, TextLog

bars = ' ▁▂▃▄▅▆▇█'
space = '   '
def num2text(x, n=10, c='[red]'):
    sgn = c if x>0 else '[blue]'
    x = abs(x)/5*9*n
    parts = [0]
    for _ in range(n):
        x = x - parts[-1]
        parts.append(max(0, min(8, int(x))))

    return [sgn+bars[p]*2 for p in parts[1:][::-1]]

# class RaveDisplay(Static):
#     value = reactive(None)
#     def watch_value(self, time: float) -> None:
#         if self.value is None: return
#         z = self.value
#         self.update(Panel.fit(
#             '\n'.join(
#                 space.join(cs) 
#                 for cs in zip(
#                     *(num2text(v.item()) for v in z)
#                 )
#             )
#         ))

# class RaveTUI(TUI):
#     CSS_PATH = 'encoder.css'

#     def compose(self):
#         yield Header()
#         yield self.std_log
#         yield RaveDisplay(id='display')
#         yield Footer()

def main(
        channels=1,
        device=0,
        block_size=2048,
        sample_rate=48000,
        midi_in=None, # MIDI port for player input
        midi_out=None, # MIDI port for Notochord output
        model="artifacts/rave_organ.ts" # RAVE model
        ):
    """
    Args:
        midi_in: MIDI port for player input
        midi_out: MIDI port for Notochord output
        checkpoint: path to RAVE exported model
    """
    midi = MIDI(midi_in, midi_out)

    ### Textual UI
    # tui = RaveTUI()
    # print = tui.print
    print = rich.print
    ###

    model = torch.jit.load(model)
    # print(model)
       
    z_mu = torch.zeros(model.cropped_latent_size)
    # z_var = torch.zeros(16)
    def callback(indata, outdata, frames, time, status):
        if status:
            print(f'sounddevice error {status=}')
        x = torch.tensor(indata, dtype=torch.float32)[None].permute(0,2,1)

        with torch.no_grad():
            z = model.encode(x)

        z = z[0,:,0]
        z_mu[:] += (z-z_mu)*0.2
        # z_var[:] += ((z-z_mu)**2 - z_var)*0.2
        # z_std = z_var**0.5

        # tui(display=z)

        print(Panel.fit(
            '\n'.join(
                space.join(cs) 
                for cs in zip(
                    *(num2text(v.item()) for v in z_mu)
                )
            )
        ))
            # +'\n'+

        # print(Panel.fit(
            # '\n'.join(
            #     space.join(cs) 
            #     for cs in zip(
            #         *(num2text(v.item(), n=5, c='[yellow]') for v in z_std)
            #     )
            # )))
  
    Audio(callback=callback,
        channels=channels, device=device, 
        blocksize=block_size, samplerate=sample_rate)
    
    # tui.run()

if __name__=='__main__':
    run(main)
