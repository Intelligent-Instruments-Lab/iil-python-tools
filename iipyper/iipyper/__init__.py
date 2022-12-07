from threading import Thread
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


class Lag:
    def __init__(self, coef, val=None):
        self.coef = coef
        self.val = val

    def __call__(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.val*self.coef + val*(1-self.coef)
        return self.val
        

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

        Thread(target=g).start()

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


def run(main=None):
    try:
        if main is not None:
            fire.Fire(main)

        for a in Audio.instances:
            a.stream.start()

    except KeyboardInterrupt:
        for a in Audio.instances:
            a.stream.stop()
            a.stream.close()
        for f in _cleanup_fns:
            f()
        raise