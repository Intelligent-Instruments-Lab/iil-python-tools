"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

import numpy as np

from iipyper import OSC, run, repeat

def main(port=5511, verbose=False):
    
    osc = OSC(port=port)
    osc.create_client("faust", port=5510)

    connected = False

    # Kisana parameters
    master = { "val": -12.0, "min": -60.0, "max": 0.0, "step": 2.0 }
    note = { "val": 5, "min": 0, "max": 11, "step": 1 }
    timbre = { "val": 0.5, "min": 0.0, "max": 1.0, "step": 0.25 }

    @osc.args("/*")
    def _(address, *args):
        """
        Handle OSC messages from Faust
        """

        if address=="/Kisana":
            # `osc("faust", "/*", "get")` response
            print(f"Kisana interface: {args}")
        
        else:
            address = address.split("/")
            cmd = address[2]
            if cmd=="level":
                print(f"Level updated to: {args}")

    @repeat(0.5)
    def _():
        nonlocal connected
        if connected==False:
            connect()
            connected=True
        update()

    def connect():
        nonlocal master
        osc("faust", "/*", "get") # discover OSC interface
        osc("faust", "/xmit", 1) # turn transmission on
        osc("faust", "/Kisana/master", master['val']) # set master gain
        print("Kisana connected!")

    def update():
        nonlocal note, timbre
        coin_flip = np.random.choice(a=[True,False], size=2)
        
        if coin_flip[0]==True:
            note_step = np.random.randint(-note['step'], note['step']+1)
            note['val'] = constrain(note['val'] + note_step, note['min'], note['max'])
            osc("faust", "/Kisana/loop38/note", note['val'])

        elif coin_flip[1]==True:
            timbre_step = np.random.random() * (timbre['step'] * 2 - timbre['step'])
            timbre['val'] = constrain(timbre['val'] + timbre_step, timbre['min'], timbre['max'])
            osc("faust", "/Kisana/timbre", timbre['val'])

    def constrain(val, min_val, max_val):
        return min(max_val, max(min_val, val))

if __name__=='__main__':
    run(main)
