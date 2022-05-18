"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from notochord import Notochord
from iipyper import OSC, run
import numpy as np

def main(host="127.0.0.1", receive_port=9999, send_port=None, checkpoint=None):
    osc = OSC(host, receive_port)

    if checkpoint is not None:
        predictor = Notochord.from_checkpoint(checkpoint)
        predictor.eval()
    else:
        predictor = None
 
    @osc.kwargs('/notochord/*', return_port=send_port)
    def _(address, **kw):
        """
        Handle OSC messages to Notochord
        """
        print(f"{address} {kw}")

        address = address.split("/")
        cmd = address[2]

        if cmd=="load":
            # `nonlocal` is needed to assign to closed-over name
            nonlocal predictor
            predictor = Notochord.from_checkpoint(**kw)
            predictor.eval()

        elif cmd=="feed":
            # print(kw)
            if predictor is None:
                print('no model loaded')
            else:
                r = predictor.feed(**kw) 

        elif cmd=="query_feed":
            # print(kw)
            if predictor is None:
                print('no model loaded')
            else:
                r = predictor.query_feed(**kw) 
                return ('/notochord/query_return', 
                    *[x for pair in r.items() for x in pair])

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
