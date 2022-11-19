"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
Run this example alongside the Bela Pure Data osc-receive example at:
https://github.com/BelaPlatform/Bela/tree/master/examples/PureData/osc-receive
"""

import random

from iipyper import OSC, run, repeat

def main():
    
    osc = OSC()
    osc.create_client("bela", host="192.168.7.2", port=7562)

    count = 0
    countMax = 6

    @repeat(0.5)
    def _():
        nonlocal count
        nonlocal countMax
        osc("bela", "/slider", random.random())
        osc("bela", "/counter", count)
        count = count + 1
        if count==countMax:
            count = 0
    
if __name__=='__main__':
    run(main)
