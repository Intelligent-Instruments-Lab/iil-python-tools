"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

import asyncio
import fire
from predictor import PitchPredictor
from osc import OSC

predictor = PitchPredictor()
osc = OSC()

def predictor_handler(address, *args):
    """
    Handle OSC messages to Predictor
    """
    address = address.split("/")

    if(address[2] == "forward"):
        print(f"/forward {args}")
        # predictor.forward(args) # sanity check args
    elif(address[2] == "predict"):
        print(f"/predict {args}")
        # predictor.predict(args) # sanity check args
    elif(address[2] == "reset"):
        print(f"/reset {args}")
        # predictor.reset()
    else:
        print(f"PitchPredictor: Unrecognised OSC {address} with {args}")

async def loop():
    """
    Separate async loop.
    """
    for i in range(10):
        osc.send_message("/hello", i)
        await asyncio.sleep(1)

async def init_main():
    fire.Fire(OSC) # Expose to CLI

    await osc.create_server(asyncio.get_event_loop())
    osc.add_handler("/predictor/*", predictor_handler)
    osc.create_client()
    
    await loop()
    
    osc.close_server()

asyncio.run(init_main())
