"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from notochord import Notochord
from torch.profiler import profile, record_function, ProfilerActivity
import fire
from time import time, sleep

def main(checkpoint=None, n=30, warm=10, wait=0):
    nc = Notochord.from_checkpoint(checkpoint)
    nc.eval()
    feed_t = query_t = predict_t = 0
    r = dict(
        instrument=0, pitch=60, time=0, velocity=0
    )
    verbose = False
    nc.reset()
    def once():
        nonlocal r
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #     with record_function("predict"):
        #         r = nc.predict(
        #             inst=r['instrument'], pitch=r['pitch'], 
        #             time=r['time'], vel=r['velocity'],
        #             pitch_temp=0.5, rhythm_temp=0.5, timing_temp=0.1)
        # if verbose:
        #     print(r)
        #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        #     prof.export_chrome_trace("trace.json")
        
        nonlocal feed_t, query_t, predict_t, r
        time
        # sleep(0.1)
        t = time()
        r = nc.predict(
            inst=r['instrument'], pitch=r['pitch'], 
            time=r['time'], vel=r['velocity'],
            pitch_temp=0.5, rhythm_temp=0.5, timing_temp=0.1)
        print(r)
        predict_t += time() - t

        t = time()
        nc.feed(
            inst=r['instrument'], pitch=r['pitch'], 
            time=r['time'], vel=r['velocity'])
        feed_t += time() - t

        t = time()
        r = nc.query(
            pitch_temp=0.5, rhythm_temp=0.5, timing_temp=0.1)
        print(r)
        query_t += time() - t
        # print(r.keys())

    for _ in range(warm):
        once()
    feed_t = query_t = predict_t = 0
    verbose = True
    for _ in range(n):
        once()

    print(dict(
        feed=feed_t/n, query=query_t/n, predict=predict_t/n))

fire.Fire(main)