"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

import numpy as np

from iipyper import OSC, run, repeat

def main(host="192.168.7.1", port=7563, verbose=False):

    osc = OSC(host, port)
    osc.create_client("bela", host="192.168.7.2", port=7562)

    pitch = { "val": 60.0, "min": 48, "max": 84, "step": 4 }
    amplitude = { "val": 0.1, "min": 0.0, "max": 0.5, "step": 0.01 }

    @osc.args("/pitch")
    def _(address, *args):
        print(f"Received pitch", args)
        pitch['val'] = args[0]

    @osc.args("/amplitude")
    def _(address, *args):
        print(f"Received amplitude", args)
        amplitude['val'] = args[0]

    @repeat(0.125)
    def _():
        nonlocal pitch, amplitude
        coin_flip = np.random.choice(a=[True,False], size=2)
        
        if coin_flip[0]==True:
            step = np.random.randint(-pitch['step'], pitch['step']+1)
            pitch['val'] = constrain(pitch['val'] + step, pitch['min'], pitch['max'])
            osc("bela", "/pitch", pitch['val'])

        elif coin_flip[1]==True:
            step = np.random.random() * (amplitude['step']*2) - amplitude['step']
            amplitude['val'] = constrain(amplitude['val'] + step, amplitude['min'], amplitude['max'])
            osc("bela", "/amplitude", amplitude['val'])

    def constrain(val, min_val, max_val):
        return min(max_val, max(min_val, val))

if __name__=='__main__':
    run(main)
