"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from iipyper import OSC, run
import numpy as np
from time import time
from collections import deque
# import heapq

def feat(l):
    a,b,c,d,*_ = l
    return np.array([a, a-b, a-2*b+c, a-3*b+3*c-d])

def dist(x, y):
    return np.linalg.norm(x-y)

# KNN on recent history, weighted average of deltas
class Predictor(object):
    def __init__(self):
        self.max_n = 1000
        self.history = deque([0.]*3)
        self.features = deque()
        self.targets = deque()
        for _ in range(2):
            self.feed(0.)

    def feed(self, raw):
        self.history.appendleft(raw)
        x = feat(self.history)
        self.features.appendleft(x)
        # self.targets.appendleft(raw)
        self.targets.appendleft(x[:2])

        if len(self.features) > self.max_n:
            self.history.pop()
            self.features.pop()
            self.targets.pop()

        print(f'{x=}')

    def query(self):
        x = self.features[0]
        n = len(self.features) - 1
        dists = np.empty(n)
        for i in range(n):
            dists[i] = dist(x, self.features[i+1])

        i = np.argmin(dists)

        # print(f'{dists=}')
        print(f'feature {x} nearest neighbor {self.features[i+1]} at {i=}')

        # return self.targets[i]

        # average of nearest neighbor and nearest delta
        y, dy = self.targets[i]
        # return (y + dy + self.history[0]) / 2
        return (y)


def main(host="127.0.0.1", receive_port=8888, send_port=7777):
    osc = OSC(host, receive_port, send_port)

    predictor = None
 
    @osc.args(return_port=7777)
    def feed(address, value):
        """
        feed data to the predictor
        """
        print(f"{address} {value}")
        predictor.feed(value)

    @osc.args(return_port=7777)
    def query(address):
        """
        query data from the predictor
        """
        print(f"{address}")

        value = predictor.query()
        predictor.feed(value)
        return '/query_return', value

    @osc.args(return_port=7777)
    def reset(address, kind=None):
        """
        reset the predictor
        """
        print(f"{address} {kind}")

        nonlocal predictor
        predictor = Predictor()

    reset(None)


if __name__=='__main__':
    run(main)