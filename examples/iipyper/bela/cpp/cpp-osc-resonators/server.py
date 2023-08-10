"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""
import random
import numpy as np
from math import log
import pandas as pd
import json

from iipyper import OSC, run, repeat

def main(host="192.168.7.1", port=7563, verbose=False):
    
    osc = OSC(host, port)
    osc.create_client("bela", host="192.168.7.2", port=7562)

    model_size = 20

    def modelGen(size):
        res = pd.DataFrame ([], columns=['freq', 'gain', 'decay'])
        for i in range (0, size):
            res.loc[i] = [
                lognorm_within_range (40, 20000),
                np.random.uniform (0, 0.5),
                np.random.uniform (0, 1.0)
            ]
        res = res.sort_values(by=['freq']).reset_index(drop=True)
        return res

    def lognorm_within_range(vmin, vmax):
        vrand = random.random()
        logratio = log(vmax) / log(vmin)
        power = ((logratio-1.) * vrand)+1.
        return vmin**power

    def model2osc(model):
        osc = {}
        return osc

    def osc2model(msg):
        model = {}
        return model

    @repeat(1)
    def _():
        nonlocal model_size
        model = modelGen(model_size)
        osc("bela", "/resonators", *model.values.flatten())

if __name__=='__main__':
    run(main)
