from threading import Timer
import time

import fire

from .midi import *
from .osc import *
from .state import _lock



# Audio WIP
import sounddevice as sd
class Audio:
    instances = [] # 
    def __init__(self, *a, **kw):
        self.stream = sd.InputStream(*a, **kw) # TODO
        Audio.instances.append(self)

_repeat_threads = []
# decorator to make a function loop
# def repeat(interval):
#     """@repeat decorator"""
#     # close the decorator over time argument
#     def decorator(f):
#         def g():
#             t = time.monotonic_ns()
#             # with _lock:
#             f()
#             dt = (time.monotonic_ns() - t)/1e9
#             wait_time = interval - dt
#             if wait_time < 0:
#                 print(f'late {-wait_time*1e3} ms')
#                 wait_time = 0
#             print(f'{wait_time=}')
#             print(f'{dt=}')
#             tt = time.monotonic()
#             Timer(wait_time, g).start()
#             dtt = (time.monotonic_ns() - tt)/1e6
#             print(dtt)

#         # track the coroutine in a global list
#         _repeat_timers.append(Timer(interval, g))

class Clock:
    def __init__(self):
        self.begin = time.perf_counter()

    def tick(self):
        return int((time.perf_counter() - self.begin)/self.interval)

    def __call__(self, interval):
        self.interval = interval
        r = self.tick() + 1
        while self.tick() < r:
            time.sleep(5e-4)

#     return decorator
def repeat(interval):
    """@repeat decorator"""
    # close the decorator over interval argument
    def decorator(f):
        def g():
            clock = Clock()
            while True:
                # with _lock:
                f()
                clock(interval)

        # track the Thread in a global list
        _repeat_threads.append(Thread(target=g))

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
        
        for t in _repeat_threads:
            t.start()

    except KeyboardInterrupt:
        for a in Audio.instances:
            a.stream.stop()
            a.stream.close()
        for f in _cleanup_fns:
            f()
        raise