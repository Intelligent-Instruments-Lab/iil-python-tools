"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from notochord import Notochord
from iipyper import OSC, run
import numpy as np
from time import time

def main(host="127.0.0.1", receive_port=9999, send_port=None, 
        checkpoint="artifacts/notochord-latest.ckpt"):
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

        elif predictor is None:
            print('no model loaded')
        else:

            if cmd=="feed":
                r = predictor.feed(**kw) 

            elif cmd=='query':
                r = predictor.query(**kw) 
                return ('/notochord/query_return', 
                    *[x for pair in r.items() for x in pair])

            elif cmd=='query_feed':
                r = predictor.query_feed(**kw) 
                return ('/notochord/query_return', 
                    *[x for pair in r.items() for x in pair])

            elif cmd=="feed_query":
                r = predictor.feed_query(**kw)
                return ('/notochord/query_return', 
                    *[x for pair in r.items() for x in pair])

            elif cmd=="feed_query_feed":
                r = predictor.feed_query_feed(**kw)
                return ('/notochord/query_return', 
                    *[x for pair in r.items() for x in pair])

            elif cmd=="predict":
                # deprecated
                t = time()
                r = predictor.predict(**kw)
                print(time() - t)
                return '/prediction', r['inst'], r['pitch'], r['time'], r['vel'], r['end'], r['step']

            elif cmd=="reset":
                predictor.reset(**kw)

            else:
                print(f"Notochord: Unrecognised OSC {address} with {kw}")

if __name__=='__main__':
    run(main)
