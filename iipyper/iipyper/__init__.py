from threading import Thread
from threading import Timer as _Timer
import time
from contextlib import contextmanager
from numbers import Number

import fire

from .midi import *
from .osc import *
from .maxmsp import *
from .oscmap import *
from .pd import *
from .audio import *
from .tui import *
from .state import _lock

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

# class Clock:
#     def __init__(self, tick=5e-4):
#         self.begin = time.perf_counter()
#         self.tick_len = tick

#     def tick(self):
#         # return the number of intervals since clock was started
#         # print('tick')
#         return int((time.perf_counter() - self.begin)/self.interval)

#     def __call__(self, interval):
#         """sleep for requested interval"""
#         if interval<=0:
#             time.sleep()
#             return
        
#         self.interval = interval
#         # sleep until it has been 1 more interval than before
#         r = self.tick() + 1
#         while self.tick() < r:
#             time.sleep(self.tick_len)

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
            return f()
    else:
        return f()

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
def repeat(interval=None, between_calls=False, lock=True, tick=5e-3):
    """@repeat decorator
    
    Args:
        interval: time in seconds to repeat at.
            If the decorated function returns a number, use that as the interval
            to the next call
        between_calls: if True, interval is between call and next call,
            if False, between return and next call
        lock: if True, use the global iipyper lock to make calls thread-safe
    """
    # close the decorator over interval and lock arguments
    def decorator(f):
        def g():
            # clock = Clock(tick)
            while True:
                t = time.perf_counter()
                returned_interval = maybe_lock(f, lock)

                if isinstance(returned_interval, Number):
                    wait_interval = returned_interval
                else:
                    wait_interval = interval

                # replace False or None with 0
                wait_interval = wait_interval or 0

                if between_calls:
                    # interval is between calls to the decorated function
                    elapsed = time.perf_counter() - t
                    wait = wait_interval - elapsed
                else:
                    t = time.perf_counter()
                    wait = wait_interval
                # else interval is between return of one and call and next call
                # print(f'{wait=}')
                # tt = time.perf_counter()
                if wait > 0:
                    sleep = wait - tick
                    if sleep > 0:
                        time.sleep(sleep)
                    spin_end = t + wait_interval
                    while time.perf_counter() < spin_end: pass
                    # print(f'waited = {time.perf_counter() - tt}')
                else:
                    print(
                        f'@repeat function "{f.__name__}" is late by {-wait}')

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