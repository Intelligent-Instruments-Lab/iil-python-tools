"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

import numpy as np
from iipyper import OSC, run, repeat

OSC_PACKET_LEN = 300
OUT_CHANNELS = 2

def main(host="192.168.7.1", port=7563):

    osc = OSC(host, port, verbose=False)
    osc.create_client("bela", host="192.168.7.2", port=7562)

    @osc.args("/*")
    def _(address, *args):
        print(f"{address} {args}")

    @repeat(1)
    def _():
        arr = np.random.rand(OSC_PACKET_LEN * OUT_CHANNELS)
        osc("bela", "/bela", *arr)

if __name__=='__main__':
    run(main)
