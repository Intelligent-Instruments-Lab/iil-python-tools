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
    note = 48
    note_on = False

    @osc.args(return_port=7777)
    def reset(address, kind=None):
        """
        reset the mrp
        """
        print("Resetting MRP...")
        nonlocal mrp
        mrp = MRP(osc)
    
    @repeat(1)
    def _():
        nonlocal note_on, note
        if note_on == False:
            mrp.note_on(note, 127)
            mrp.quality_brightness(note, 1)
            mrp.quality_intensity(note, 1)
            note_on = True
        else:
            mrp.note_off(note, 0)
            note_on = False

    reset(None)

if __name__=='__main__':
    run(main)
