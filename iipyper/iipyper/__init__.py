from threading import Thread
import time
from contextlib import contextmanager

import fire

from .midi import *
from .osc import *
from .tui import *
from .state import _lock

# Audio WIP
import sounddevice as sd
class Audio:
    instances = [] # 
    def __init__(self, *a, **kw):
        self.stream = sd.InputStream(*a, **kw) # TODO
        Audio.instances.append(self)

@contextmanager
def profile(label, print=print):
    t = time.perf_counter_ns()
    yield None
    dt = (time.perf_counter_ns() - t)*1e-9
    print(f'{label}:\t {int(1000*dt)} ms')


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
        """sleep for requested interval"""
        self.interval = interval
        r = self.tick() + 1
        while self.tick() < r:
            time.sleep(5e-4)

class Timer:
    def __init__(self, punch=False):
        self.t = None
        if punch:
            self.punch()

    def punch(self):
        """return elapsed time since last punch, then punch"""
        t = time.perf_counter_ns()
        if self.t is None:
            dt_ns = 0.0
        else:
            dt_ns = t - self.t
        self.t = t
        return dt_ns * 1e-9

    def read(self):
        """return elapsed time since last punch"""
        if self.t is None:
            return 0.0
        return (time.perf_counter_ns() - self.t) * 1e-9

_threads = []
def repeat(interval, lock=False):
    """@repeat decorator"""
    # close the decorator over interval argument
    def decorator(f):
        def g():
            clock = Clock()
            while True:
                if lock:
                    with _lock:
                        f()
                else:
                    f()
                clock(interval)

        th = Thread(target=g, daemon=True)
        th.start()
        _threads.append(th)

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

        # enter a loop if there is not one in main
        while True:
            time.sleep(3e-2)

    except KeyboardInterrupt:
        # for th in _threads:
            # pass
        for a in Audio.instances:
            a.stream.stop()
            a.stream.close()
        for f in _cleanup_fns:
            f()
        exit(0)