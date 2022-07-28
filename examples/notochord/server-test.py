"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from notochord import Notochord
from iipyper import OSC, run
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict

# torch.set_num_threads(1)

def main(host="127.0.0.1", receive_port=9999, send_port=None, checkpoint=None):
    osc = OSC(host, receive_port)

    h = torch.randn(1,3000)
    W = torch.randn(1,3000,3000)

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
                # r = defaultdict(float)

                with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                    with record_function("predict"):
                        r = predictor.predict(**kw)
                        # nonlocal h
                        # h = h@W
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
                # prof.export_chrome_trace("trace2.json")

                # return '/prediction', r['instrument'], r['pitch'], r['time'], r['velocity'], r['end'], r['step']
                return '/prediction', 0,0,0,0,0,0

        elif cmd=="reset":
            if predictor is None:
                print('no model loaded')
            else:
                predictor.reset(**kw)
            
        else:
            print(f"PitchPredictor: Unrecognised OSC {address} with {kw}")

if __name__=='__main__':
    run(main)
