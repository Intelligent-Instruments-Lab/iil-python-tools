import asyncio

import fire

from .midi import *
from .osc import *

_loop_fns = []
# decorator to make a function loop
def repeat(time):
    # close the decorator over time argument
    def decorator(f):
        # define the coroutine
        async def g():
            # call `f` every `time` seconds
            while True:
                f()
                await asyncio.sleep(time)
        # track the coroutine in a global list
        _loop_fns.append(g)

    return decorator

async def _run_async():
    # start OSC server
    for osc in OSC.instances:
        await osc.create_server(asyncio.get_event_loop())

    for midi in MIDI.instances:
        asyncio.create_task(midi_coroutine(midi))

    # start loop tasks
    if len(_loop_fns):
        for f in _loop_fns:
            asyncio.create_task(f())

    while True:
        await asyncio.sleep(1)

    # clean up
    # for osc in OSC.instances:
    #     osc.close_server()

def run(main=None):
    if main is not None:
        fire.Fire(main)
    asyncio.get_event_loop().run_until_complete(_run_async())
    #asyncio.run(_run_async())
