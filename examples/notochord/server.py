"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

import json

from notochord import Notochord

from iipyper import OSC, run

def main(host="127.0.0.1", port=9999, checkpoint=None):
    osc = OSC(host, port)

    if checkpoint is not None:
        predictor = Notochord.from_checkpoint(checkpoint)
        predictor.eval()
    else:
        predictor = None
 
    @osc.kwargs('/predictor/*', json_keys=[
        'include_pitch', 'exclude_pitch', 'include_instrument', 'exclude_instrument'])
    def _(address, **kw):
        """
        Handle OSC messages to Predictor
        """
        print(f"{address} {kw}")

        address = address.split("/")
        cmd = address[2]

        if cmd=="load":
            # `nonlocal` is needed to assign to closed-over name
            nonlocal predictor
            predictor = Notochord.from_checkpoint(**kw)
            predictor.eval()

        elif cmd=="predict":
            if predictor is None:
                print('no model loaded')
            else:
                r = predictor.predict(**kw)
                return '/prediction', r['instrument'], r['pitch'], r['time'], r['velocity'], r['end'], r['step']

        elif cmd=="reset":
            if predictor is None:
                print('no model loaded')
            else:
                predictor.reset(**kw)
            
        else:
            print(f"PitchPredictor: Unrecognised OSC {address} with {kw}")

if __name__=='__main__':
    run(main)
