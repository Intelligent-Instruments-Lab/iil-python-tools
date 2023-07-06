'''
TODO: send args? maybe this is only useful for sending repeated things with fixed args.
'''

from .osc import OSCSendUpdater, OSCSend, OSCReceiveUpdater
from .max import MaxPatcher
from .pd import PdPatcher

from typing import Any

class OSCMap:
    '''
    OSCMap maps OSC messages to functions
    It creates a Max/MSP patcher that can be used to control the OSCMap
    It uses OSCSendUpdater and OSCReceiveUpdater to decouple incoming messages
    TODO: make max_patch optional
    TODO: OSC sends should have broadcast vs event option?
    '''
    def __init__(self, osc, client_name="client", 
                 patch_type="Max", # | "Pd"
                 patch_filepath="osc_controls",
                 create_patch=True) -> None:
        self.osc = osc
        self.client_name = client_name
        self.client_address, self.client_port = self.osc.client_names[self.client_name]
        self.dict = {'send': {}, 'receive': {}}
        self.create_patch = create_patch
        if self.create_patch is True:
            self.patch_filepath = patch_filepath
            if patch_type == "Max":
                self.patcher = MaxPatcher(osc, client_name, self.patch_filepath)
            elif patch_type == "Pd":
                self.patcher = PdPatcher(osc, client_name, self.patch_filepath)
            else:
                assert False, "`patch_type` must be 'Max' or 'Pd'"

    def add_func_to_osc_map(self, func, kwargs):
        n = func.__name__
        address = '/'+n.replace('_', '/')
        params = {k: v for k, v in kwargs.items() \
                    if k != 'io' and k != 'count' and k != 'send_mode'}
        f = {'f': func, 'address': address, 'params': params}
        if 'io' not in kwargs:
            raise ValueError('io must be specified')
        if 'count' not in kwargs:
            raise ValueError('count must be specified')
        if kwargs['io'] == 'send':
            if kwargs['send_mode'] == 'broadcast':
                f['updater'] = OSCSendUpdater(self.osc, address, f=func, count=kwargs['count'], client=self.client_name)
            else:
                f['sender'] = OSCSend(self.osc, address, f=func, count=kwargs['count'], client=self.client_name)
            self.dict['send'][n] = f
        elif kwargs['io'] == 'receive':
            f['updater'] = OSCReceiveUpdater(self.osc, address, f=func, count=kwargs['count'])
            self.dict['receive'][n] = f
        else:
            raise ValueError('io must be send or receive')
        return f

    def add_func_to_patcher(self, func, io):
        f = self.dict[io][func.__name__]
        if io == 'send':
            self.patcher.add_send_func(f)
        elif io == 'receive':
            self.patcher.add_receive_func(f)
    
    def add(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_func_to_osc_map(func, kwargs)
                if self.create_patch is True:
                    self.add_func_to_patcher(func, kwargs['io'])
                # TODO: type checking/scaling/clamping of params
                # modified_kwargs = {key: decorator_kwargs.get(key, value) for key, value in kwargs.items()}
                if kwargs['io'] == "receive":
                    return func(*args)
                else:
                    return func()
            # call wrapped function on declaration to add to map
            default_args = [kwargs[a][0] for a in kwargs \
                            if a != 'io' and a != 'count']
            wrapper(*default_args)
            return wrapper
        return decorator
    
    def send(self, f, *args):
        self.dict['send'][f]['sender'](*args)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for k, v in self.dict['send'].items():
            if 'updater' in v:
                v['updater']()
            # v['updater']()
        for k, v in self.dict['receive'].items():
            v['updater']()
