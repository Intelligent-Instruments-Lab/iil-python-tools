import asyncio
import functools
import threading

import fire

from .midi import *
from .osc import *

_loop_fns = []

def repeat(_time):
    """
    decorator to repeat function every 'time' seconds
    """
    # close the decorator over time argument
    def decorator(f):
        # define the coroutine
        @functools.wraps(f)
        async def g(*args, **kwargs):
            # call `f` every `time` seconds
            while True:
                t1 = time.time()
                f(*args, **kwargs)
                t2 = time.time()
                delta_t = t2 - t1
                sleep_time = _time - delta_t

                if (sleep_time < 0):
                    print("Warning: repeat@ sleep time < 0")
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(sleep_time)

        # track the coroutine in a global list
        _loop_fns.append(g)
        return f
    return decorator


_exit_fns = []
# decorator to make a function run on KeyBoardInterrupt (before exit)
def cleanup(f=None):

    def decorator(f):
        _exit_fns.append(f)
        return f

    if f is None: # return a decorator
        return decorator
    else: #bare decorator case; return decorated function
        return decorator(f)


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

    # try:
    while True:
        await asyncio.sleep(1)
    # except KeyboardInterrupt:
        # for f in _exit_fns:
            # f()
        # raise

    # clean up
    # for osc in OSC.instances:
    #     osc.close_server()

def run(main=None):
    try:
        if main is not None:
            fire.Fire(main)
        asyncio.run(_run_async())
    except KeyboardInterrupt:
        for f in _exit_fns:
            f()
        raise