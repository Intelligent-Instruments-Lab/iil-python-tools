"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from iipyper import OSC, run, repeat
from mrp import MRP
import random

def main(host="127.0.0.1", receive_port=8888, send_port=7770):

    osc = OSC(host, receive_port, send_port)
    osc.create_client("mrp", port=send_port)

    mrp = None
    note_on = False
    note = 48
    count = 100 # frames
    counter = 0

    @osc.args(return_port=7777)
    def reset(address, kind=None):
        """
        reset the mrp
        """
        print("Resetting MRP...")
        nonlocal mrp, note
        mrp = MRP(osc)
        mrp.all_notes_off()
        mrp.note_on(note)
    
    @repeat(0.1)
    def _():
        nonlocal note, count, counter
        mrp.quality_update(note, 'intensity', 1.0)#counter/count)
        mrp.quality_update(note, 'brightness', counter/count)
        # mrp.quality_update(note, 'harmonic', counter/count)
        mrp.quality_update(note, 'pitch', (counter/count)*2-1)
        counter+=1
        if counter == count:
            mrp.note_off(note)
            note+=random.randint(-24, 24)
            mrp.note_on(note)
            counter=0
        print(counter, count, counter/count)

    reset(None)

if __name__=='__main__':
    run(main)
