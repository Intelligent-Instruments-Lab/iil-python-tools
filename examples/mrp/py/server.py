"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from iipyper import OSC, run, repeat
from mrp import MRP

def loop():
    print('loop')

def main(host="127.0.0.1", receive_port=8888, send_port=7770):

    osc = OSC(host, receive_port, send_port)
    osc.create_client("mrp", port=send_port)

    mrp = None
    note = 48
    note_on = False
    count=0
    test='code'#'voices', 'qualities'
    qualities=['brightness', 'intensity', 'pitch', 'pitch_vibrato', 'harmonic', 'harmonics_raw']

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
        nonlocal note_on, note, count, test
        if note_on == False:
            count+=1
            if test == 'code':
                mrp.note_on(note)
                mrp.quality_update(note, 'brightness', 0.5+count/10)
                mrp.quality_update(note, 'intensity', 1.9)
                mrp.quality_update(note, 'brightness', 1.5, relative=True)
                mrp.quality_update(note, 'intensity', 1.9, relative=True)
                mrp.quality_update(note, 'harmonics_raw', [1.1, 0.2, 0.3])
                mrp.quality_update(note, 'harmonics_raw', [i/10 for i in range(0, count, 1)])
                mrp.qualities_update(note, {
                    'brightness': 1.5,
                    'intensity': 1.0,
                    'harmonics_raw': [1.2, 0.3, 0.4]
                })
            else if test == 'voices':
                mrp.note_on(note+count)
                print(len(mrp.voices), 'voices:', mrp.voices)
            else if test == 'qualities':
                mrp.note_on(note)
                mrp.quality_update(qualities[0], count/10)
            note_on = True
        else:
            if test == 'code':
                mrp.note_off(note)
            else if test == 'voices':
                if count % 2:
                    mrp.note_off(note+int(count/2))
            note_on = False

    reset(None)

if __name__=='__main__':
    run(main)
