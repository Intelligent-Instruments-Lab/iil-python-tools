import json
from typing import Any
from copy import deepcopy

class JSONSerializable:
    """JSON serialization for Python classes.
    Saves keyword arguments at construction,
    and also any state returned by the `save_state` method.

    to make class a serializable, subclass JSONSerializable, 
    and in the constructor use e.g. `super().__init__(a=0, b=1 ...)`
    with any keyword args which should be serialized.

    override `save_state` and `load_state` to handle any mutable state.

    Constructor args and return values of `save_state` can be other JSONSerializable objects.
    """
    def __init__(self, **kw):
        self._kw = deepcopy(kw)
        self._kw['__inst__'] = '.'.join((
            self.__class__.__module__,
            self.__class__.__name__))

    def _store(self):
        return {'__state__': self.save_state(), **self._kw}

    def save_state(self):
        """return object state in JSON serializable form"""
        return None

    def load_state(self, state):
        """restore from de-serialized state"""
        pass

class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, JSONSerializable):
            # instance of JSONSerializable
            return o._store()
        elif isinstance(o, type):
            # type 
            return {'__type__':'.'.join((o.__module__, o.__name__))}
        else:
            return super().default(o)
        
def get_cls(s):
    # sanitize inputs
    assert all(item.isidentifier() for item in s.split('.')), s
    parts = s.split('.')
    pkg = parts[0]
    mod = '.'.join(parts[:-1])
    # import top level package the type belongs to
    exec(f'import {pkg}') 
    # import submodule the type is in
    exec(f'import {mod}')
    # convert string to type
    return eval(s)

class JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, d):
        if '__inst__' in d:
            cls = get_cls(d.pop('__inst__'))
            state = d.pop('__state__')
            inst = cls(**d)
            inst.load_state(state)
            return inst
        elif '__type__' in d:
            assert len(d)==1, d
            cls = get_cls(d['__type__'])
            return cls
        return d
    
def load(path):
    with open(path, 'r') as f:
        return json.load(f, cls=JSONDecoder)        
    
def save(path, obj):
    with open(path, 'w') as f:
        return json.dump(obj, f, cls=JSONEncoder)
