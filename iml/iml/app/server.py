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

    OSC Routes:

        /iml/config/emb "Identity"
            set embedding to Identity (the default)
        /iml/config/emb "ProjectAndSort"
            set embedding to ProjectAndSort

        /iml/config/interp "Smooth"
            set interpolator to Smooth (the default)
        /iml/config/interp "Softmax"
            set interpolator to Softmax
        /iml/config/interp "Ripple"
            set interpolator to Ripple

        -- or --
        /iml/config "emb" ... "interp" ...

        /iml/add "input" ... "output"... 
            add a point to the mapping

        /iml/remove id 
            remove a point from the mapping by ID

        /iml/remove_near "input" ... ["k" k]
            remove k points from the mapping by proximity

        /iml/map "input" ... ["k" k] ["ripple" r] ["temp" t]
            map an input to an output using k neighbors
            "temp" 1 > t > 0 when using Softmax interpolator
            "ripple" r > 0 when using Ripple interpolator

        /iml/reset
            remove all points
        /iml/reset "keep_near" ... ["k" k]
            remove all points except the k neighbors of "keep_near"

        /iml/load path
            load IML from file at `path`
        /iml/save path
            save IML to file at `path`
    """
    osc = OSC(osc_host, osc_port)

    iml = None

    config = {}

    @osc.kwargs('/iml/config')
    def _(address, **kw):
        # TODO: validate input
        config.update(kw)
        print(config) 

    @osc.args('/iml/config/interp')
    def _(address, name):
        if iml is None:
            config['interp'] = name
        else:
            iml.set_interp(name)

    @osc.args('/iml/config/emb')
    def _(address, name):
        if iml is None:
            config['emb'] = name
        else:
            iml.set_emb(name)

    @osc.args('/iml/add')
    def _(address, *a):
        nonlocal iml
        kw = vector_args(a)

        if 'input' not in kw:
            print('ERROR: iml: no input vector supplied')
            return
        if 'output' not in kw:
            print('ERROR: iml: no output vector supplied')
            return

        # d = len(kw['input'])
        # config['feature_size'] = d
        if iml is None:
            # print(f'new IML object with Input dimension {d}')
            print(f'new IML object with {config}')
            iml = IML(**config)

        return '/iml/return/add', iml.add(**kw)
    
    @osc.args('/iml/remove')
    def _(address, id):
        iml.remove(id)

    @osc.args('/iml/remove_near')
    def _(address, *a):
        kw = vector_args(a)
        for k in ['k']:
            if k in kw:
                kw[k] = kw[k][0]

        if 'input' not in kw:
            print('ERROR: iml: no input vector supplied')
            return
        
        iml.remove_near(**kw)

    @osc.args('/iml/map', return_port=osc_return_port)
    def _(address, *a):
        kw = vector_args(a)
        for k in ['k', 'temp', 'ripple']:
            if k in kw:
                kw[k] = kw[k][0]

        if 'input' not in kw:
            print('ERROR: iml: no input vector supplied')
            return
        
        if iml is None:
            print('ERROR: iml: call /iml/add at least once before /map')
            return
        
        result = iml.map(**kw).tolist()

        return '/iml/return/map', *result
    
    @osc.args('/iml/reset')
    def _(address, *a):
        if iml is not None:
            kw = vector_args(a)
            for k in ['k']:
                if k in kw:
                    kw[k] = kw[k][0]

            iml.reset(**kw)

    @osc.args('/iml/load')
    def _(address, path):
        nonlocal iml
        assert isinstance(path, str)
        assert path.endswith('.json'), "ERROR: iml: path should end with .json"
        print(f'new IML object from {path}')
        iml = IML.load(path)

    @osc.args('/iml/save')
    def _(address, path):
        assert isinstance(path, str)
        assert path.endswith('.json'), "ERROR: iml: path should end with .json"
        print(f'saving IML object to {path}')
        iml.save(path)

if __name__=='__main__':
    run(main)
