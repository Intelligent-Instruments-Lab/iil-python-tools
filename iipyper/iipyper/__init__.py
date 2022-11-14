import fire

from .midi import *
from .osc import *
from .lock import _lock


from threading import Timer


# Audio WIP
import sounddevice as sd
class Audio:
    instances = [] # 
    def __init__(self, *a, **kw):
        self.stream = sd.InputStream(*a, **kw) # TODO
        Audio.instances.append(self)

_repeat_timers = []
# decorator to make a function loop
def repeat(time):
    """@repeat decorator"""
    # close the decorator over time argument
    def decorator(f):
        # define the task

        def g():
            with _lock:
                f()
            Timer(time, g).start()

        # track the coroutine in a global list
        _repeat_timers.append(Timer(time, g))

    return decorator


_cleanup_fns = []
# decorator to make a function run on KeyBoardInterrupt (before exit)
def cleanup(f=None):
    """@cleanup decorator"""
    def decorator(f):
        _cleanup_fns.append(f)
        return f

    if f is None: # return a decorator
        return decorator
    else: #bare decorator case; return decorated function
        return decorator(f)


# async def _run_async():
#     # start OSC server
#     for osc in OSC.instances:
#         await osc.create_server(asyncio.get_event_loop())

#     for midi in MIDI.instances:
#         asyncio.create_task(midi_coroutine(midi))

#     # start loop tasks
#     if len(_loop_fns):
#         for f in _loop_fns:
#             asyncio.create_task(f())

#     while True:
#         await asyncio.sleep(1)

    # clean up
    # for osc in OSC.instances:
    #     osc.close_server()

def run(main=None):
    try:
        if main is not None:
            fire.Fire(main)

        for a in Audio.instances:
            a.stream.start()

        for osc in OSC.instances:
            osc.create_server()

        for midi in MIDI.instances:
            midi.start()
        
        for t in _repeat_timers:
            t.start()

    except KeyboardInterrupt:
        for a in Audio.instances:
            a.stream.stop()
            a.stream.close()
        for f in _cleanup_fns:
            f()
        raise