"""
Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

from iml import IML
from iipyper import OSC, run
import numpy as np
from time import time
from collections import defaultdict
from typing import Optional

def vector_args(a):
    a = list(a)
    args = defaultdict(list)
    k = None
    while len(a):
        item = a.pop(0)
        # print(type(item), item)
        if isinstance(item, str):
            k = item
        else:
            if k is None:
                print('ERROR: iml: bad OSC syntax')
            args[k].append(item)
    return args

def main(
    osc_port:int=8732,
    osc_return_port:Optional[int]=None,
    osc_host:str='',
    ):
    """
    Args:
        osc_port: listen for OSC controls on this port
        osc_return_port: if supplied, reply on a different port than osc_port
        osc_host: leave this as empty string to get all traffic on the port
    """
    osc = OSC(osc_host, osc_port)

    iml = None

    config = {}

    @osc.kwargs('/iml/config')
    def _(address, **kw):
        nonlocal iml
        # TODO: validate input
        config.update(kw)
        print(config)
        iml = IML(**config)
 
 
    @osc.args('/iml/add')
    def _(address, *a):
        nonlocal iml
        kw = vector_args(a)

        if 'source' not in kw:
            print('ERROR: iml: no source vector supplied')
            return
        if 'target' not in kw:
            print('ERROR: iml: no target vector supplied')
            return

        d = len(kw['source'])
        if iml is None:
            print(f'new IML object with source dimension {d}')
            iml = IML(d)

        iml.add(**kw)


    @osc.args('/iml/map', return_port=osc_return_port)
    def _(address, *a):
        kw = vector_args(a)

        if 'source' not in kw:
            print('ERROR: iml: no source vector supplied')
            return
        
        if iml is None:
            print('ERROR: iml: call /iml/add at least once before /map')
            return
        
        result = iml.map(**kw)

        return '/iml/return', *result
    
    @osc.kwargs('/iml/reset')
    def _(address, **kw):
        if iml is not None:
            iml.reset()

if __name__=='__main__':
    run(main)
