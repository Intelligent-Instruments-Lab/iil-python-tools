from threading import Thread
from threading import Timer as _Timer
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
        print(sd.query_devices())
        # self.stream = sd.InputStream(*a, **kw) # TODO
        self.stream = sd.Stream(*a, **kw) # TODO
        Audio.instances.append(self)

@contextmanager
def profile(label, print=print):
    t = time.perf_counter_ns()
    yield None
    dt = (time.perf_counter_ns() - t)*1e-9
    print(f'{label}:\t {int(1000*dt)} ms')


class Lag:
    def __init__(self, coef_up, coef_down=None, val=None):
        self.coef_up = coef_up
        self.coef_down = coef_down or coef_up
        self.val = val

    def __call__(self, val):
        if self.val is None:
            self.val = val
        else:
            coef = self.coef_up if val > self.val else self.coef_down
            self.val = self.val*coef + val*(1-coef)
        return self.val
    
    def hpf(self, val):
        return val - self(val)

class Clock:
    def __init__(self, tick=5e-4):
        self.begin = time.perf_counter()
        self.tick_len = tick

    def tick(self):
        # return the number of intervals since clock was started
        # print('tick')
        return int((time.perf_counter() - self.begin)/self.interval)

    def __call__(self, interval):
        """sleep for requested interval"""
        if interval<=0:
            time.sleep()
            return
        
        self.interval = interval
        # sleep until it has been 1 more interval than before
        r = self.tick() + 1
        while self.tick() < r:
            time.sleep(self.tick_len)

    # def __enter__(self):
    #     self.mark = time.perf_counter()

    # def __exit__(self, type, value, tb):
    #     t = time.perf_counter()
    #     self()

class Stopwatch:
    def __init__(self, punch=True):
        self.t = None
        if punch:
            self.punch()

    def punch(self, latency=0):
        """return elapsed time since last punch, then punch
        
        Args:
            latency: punch `latency` seconds in the past, 
                unless it would be before the previous punch
        """
        t = time.perf_counter_ns() - latency
        if self.t is None:
            dt_ns = 0
        else:
            t = max(self.t, t)
            dt_ns = t - self.t
        self.t = t
        return dt_ns * 1e-9

    def read(self):
        """return elapsed time since last punch"""
        if self.t is None:
            return self.punch()
        return (time.perf_counter_ns() - self.t) * 1e-9

def maybe_lock(f, lock):
    if lock:
        with _lock:
            f()
    else:
        f()

class Timer:
    """a threading.Timer using the global iipyper lock around the timed function
    also starts automatically by default.
    """
    def __init__(self, interval, f, lock=True, start=True, **kw):
        self.timer = _Timer(max(0,interval), maybe_lock(f, lock), **kw)
        if start:
            self.start()
    def cancel(self):
        self.timer.cancel()
    def start(self):
        self.timer.start()

_threads = []
def repeat(interval, between_calls=False, lock=True):
    """@repeat decorator
    
    Args:
        interval: time in seconds to repeat at
        between_calls: if True, interval is between call and next call,
            if False, between return and next call
        lock: if True, use the global iipyper lock to make calls thread-safe
    """
    # close the decorator over interval and lock arguments
    def decorator(f):
        def g():
            clock = Clock(5e-3)
            while True:
                t = time.perf_counter()
                maybe_lock(f, lock)
                if between_calls:
                    # interval is between calls to the functions
                    elapsed = time.perf_counter() - t
                    wait = interval - elapsed
                else:
                    wait = interval
                # else interval is between return of one and call and next call
                if wait > 0:
                    clock(wait)
                else:
                    print(
                        f'@repeat function "{f.__name__}" is late by {-interval}')

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

# locking decorator
def lock(f):
    """wrap the decorated function with the global iipyper lock"""
    def decorated(*a, **kw):
        with _lock:
            f(*a, **kw)
    return decorated

def start_audio():
    for a in Audio.instances:
        if not a.stream.active:
            a.stream.start()

def run(main=None):
    try:
        if main is not None:
            fire.Fire(main)

        # non-blocking main case:
        start_audio()

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