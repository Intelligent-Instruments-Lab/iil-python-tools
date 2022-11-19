"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""
import random

from iipyper import OSC, run, repeat

def main(host="192.168.7.1", port=7563):
    
    osc = OSC(host, port)
    osc.create_client("bela", host="192.168.7.2", port=7562)

    connected = False

    count = 0
    freq = 440.0
    gain = 0.1
    decay = 0.5

    @repeat(1)
    def _():
        nonlocal freq, gain, decay, count
        freq = random.random() * 10000 + 50
        gain = random.random()
        decay = random.random()
        if count==0:
            osc("bela", "/resonator", freq, gain, decay)
        elif count==1:
            osc("bela", "/resonator/freq", freq)
        elif count==2:
            osc("bela", "/resonator/gain", gain)
        elif count==3:
            osc("bela", "/resonator/decay", decay)
        count = (count + 1) % 4

if __name__=='__main__':
    run(main)
