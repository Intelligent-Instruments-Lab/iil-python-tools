"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

import asyncio
import fire
from notepredictor import NotePredictor
from osc import OSC

def predictor_handler(address, *args):
    """
    Handle OSC messages to Predictor
    """
    address = address.split("/")

    if(address[2] == "load"):
        print(f"/load {args}")
        global predictor
        predictor = NotePredictor.from_checkpoint(*args)
        predictor.eval()

    elif(address[2] == "predict"):
        print(f"/predict {args}")
        r = predictor.predict(*args)
        msg = (r['pitch'], r['time'])
        # print(msg)
        osc.send_message('/prediction', msg)

    elif(address[2] == "reset"):
        print(f"/reset {args}")
        predictor.reset(*args)
        
    else:
        print(f"PitchPredictor: Unrecognised OSC {address} with {args}")

async def loop():
    """
    Separate async loop.
    """
    i = 0
    while True:
        i += 1
        # osc.send_message("/hello", i)
        await asyncio.sleep(1)

async def init_main():
    await osc.create_server(asyncio.get_event_loop())
    osc.add_handler("/predictor/*", predictor_handler)
    osc.create_client()
    
    await loop()
    
    osc.close_server()

def main(ip="127.0.0.1", send=57120, receive=9999, checkpoint=None):
    global osc, predictor

    osc = OSC(ip, send, receive)

    if checkpoint is not None:
        predictor = NotePredictor.from_checkpoint(checkpoint)
        predictor.eval()

    asyncio.run(init_main())

if __name__=='__main__':
    fire.Fire(main)
