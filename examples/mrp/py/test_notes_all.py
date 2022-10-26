"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from iipyper import OSC, run, repeat
from mrp import MRP

def main(host="127.0.0.1", receive_port=8888, send_port=7770):

    osc = OSC(host, receive_port, send_port)
    osc.create_client("mrp", port=send_port)

    mrp = None
    note_on = False
    note = 0

    @osc.args(return_port=7777)
    def reset(address, kind=None):
        """
        reset the mrp
        """
        print("Resetting MRP...")
        nonlocal mrp
        mrp = MRP(osc)
    
    @repeat(0.125)
    def _():
        nonlocal note_on, note
        current = (note % 88) + 21
        if note_on == False:
            mrp.note_on(current, 127)
            mrp.quality_update(current, 'brightness', 1)
            mrp.quality_update(current, 'intensity', 1)
            note_on = True
        else:
            mrp.note_off(current)
            note_on = False
            note+=1
        print(current, mrp.voices)

    @cleanup
    def _():
        mrp.cleanup()

    reset(None)

if __name__=='__main__':
    run(main)
