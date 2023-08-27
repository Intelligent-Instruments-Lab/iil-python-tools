'''
TODO: Add send_kwargs and receive_kwargs
TODO: Refactor self.dict[io][name][params] to a labelled dictionary & add type hints
TODO: Load OSCMap from XML or JSON (probably involves refactoring, for the better)
TODO: Better handling of directories when saving/exporting (separate dir for xml/json?)
TODO: Update Max and Pd patch generation to use the new OSCMap send/receive list functions
TODO: Add generator for ESP32
TODO: send funcs in send_mode=broadcast mode cannot have args, but in event mode you might want args
'''

from .osc import OSCSendUpdater, OSCSend, OSCReceiveUpdater
from .maxmsp import MaxPatcher
from .pd import PdPatcher

from typing import Any, get_type_hints
import os
import xml.etree.ElementTree as ET
import json
import numpy as np

class OSCMap:
    '''
    OSCMap maps OSC messages to functions
    It creates a Max/MSP patcher that can be used to control the OSCMap
    It uses OSCSendUpdater and OSCReceiveUpdater to decouple incoming messages
    '''
    def __init__(self, osc, client_name="client", 
                 patch_type="Max", # | "Pd"
                 patch_filepath="osc_controls",
                 create_patch=True,
                 pd_net_or_udp="udp",
                 pd_bela=False,
                 export=None # 'JSON' | 'XML' | True
                 ) -> None:
        self.osc = osc
        self.client_name = client_name
        self.client_address, self.client_port = self.osc.client_names[self.client_name]
        self.dict = {'send': {}, 'receive': {}}
        self.create_patch = create_patch
        self.patch_filepath = patch_filepath
        self.patch_type = patch_type
        if create_patch is True:
            self.init_patcher(patch_type, patch_filepath, pd_net_or_udp, pd_bela)
        if export is not None:
            assert export == 'JSON' or export == 'XML' or export == True, "export must be 'JSON', 'XML' or True"
        self.export = export

    def init_patcher(self, patch_type, patch_filepath, pd_net_or_udp, pd_bela):
        # create self.patch_dir if it doesn't exist
        self.patch_dir = "pd" if patch_type=="Pd" else "max"
        if not os.path.exists(self.patch_dir):
            print(f"Creating {self.patch_dir} directory...")
            os.makedirs(self.patch_dir)
        self.patch_appendix = "_local" if self.osc.host=="127.0.0.1" else "_remote"
        self.patch_filepath = self.patch_dir+'/'+patch_filepath+self.patch_appendix
        if patch_type == "Max":
            self.patcher = MaxPatcher(self.osc, self.client_name, self.patch_filepath)
        elif patch_type == "Pd":
            if pd_bela is True:
                self.patcher = PdPatcher(self.osc, self.client_name, self.patch_filepath, net_or_udp=pd_net_or_udp, bela=True)
            else:
                self.patcher = PdPatcher(self.osc, self.client_name, self.patch_filepath, net_or_udp=pd_net_or_udp)
        else:
            assert False, "`patch_type` must be 'Max' or 'Pd'"

    def add(self, **kwargs):
        print("DeprecationError: OSCMap.add() has been split into separate functions: use `send_args`, `send_list`, `receive_args` or `receive_list` instead!")
        exit()
    
    def map_func_to_dict(self, func, kwargs):
        n = func.__name__
        address = '/'+n.replace('_', '/')
        # TODO: Move this into specific send/receive functions
        params = {k: v for k, v in kwargs.items() if \
                    k != 'count' and \
                    k != 'send_mode' and \
                    k != 'length'}
        # TODO: turn params into dict with type hints (see export_dict)
        hints = get_type_hints(func)
        f = {'f': func, 'name': n, 'address': address, 'params': params, 'hints': hints}
        return f

    '''
    send args
    '''
    
    def send_args(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_send_args_to_osc_map(func, kwargs)
                if self.create_patch is True:
                    self.add_send_args_to_patcher(func)
                return func()
            default_args = [kwargs[a][0] for a in kwargs \
                            if a != 'count' and a != 'send_mode']
            wrapper(*default_args)
            return wrapper
        return decorator

    def add_send_args_to_osc_map(self, func, kwargs):
        f = self.map_func_to_dict(func, kwargs)
        if kwargs['send_mode'] == 'broadcast':
            f['updater'] = OSCSendUpdater(self.osc, f['address'], f=func, count=kwargs['count'], client=self.client_name)
        else:
            f['sender'] = OSCSend(self.osc, f['address'], f=func, count=kwargs['count'], client=self.client_name)
        f['type'] = 'args'
        self.dict['send'][f['name']] = f
        if self.export is not None:
            self.export_dict()
    
    def add_send_args_to_patcher(self, func):
        f = self.dict['send'][func.__name__]
        self.patcher.add_send_args_func(f)

    '''
    send list
    '''

    def send_list(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_send_list_to_osc_map(func, kwargs)
                if self.create_patch is True:
                    self.add_send_list_to_patcher(func)
                return func()
            default_arg = [kwargs[a][0] for a in kwargs \
                            if a != 'count' and a != 'send_mode' and a != 'length']
            default_arg = default_arg#*kwargs['length']
            wrapper(default_arg)
            return wrapper
        return decorator

    def add_send_list_to_osc_map(self, func, kwargs):
        f = self.map_func_to_dict(func, kwargs)
        hint = f['hints']['return']
        assert hint == list[float], "send_list can only send list[float], found "+str(hint)
        if kwargs['send_mode'] == 'broadcast':
            f['updater'] = OSCSendUpdater(self.osc, f['address'], f=func, count=kwargs['count'], client=self.client_name)
        else:
            f['sender'] = OSCSend(self.osc, f['address'], f=func, count=kwargs['count'], client=self.client_name)
        f['type'] = 'list'
        f['length'] = kwargs['length']
        self.dict['send'][f['name']] = f
        if self.export is not None:
            self.export_dict()
    
    def add_send_list_to_patcher(self, func):
        f = self.dict['send'][func.__name__]
        self.patcher.add_send_list_func(f)

    '''
    send kwargs
    '''

    def send_kwargs(self, **kwargs):
        raise NotImplementedError("send_kwargs not implemented yet")

    '''
    receive args
    '''

    def receive_args(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_receive_args_to_osc_map(func, kwargs)
                if self.create_patch is True:
                    self.add_receive_args_to_patcher(func)
                return func(*args)
            default_args = [kwargs[a][0] for a in kwargs if a != 'count']
            wrapper(*default_args)
            return wrapper
        return decorator
    
    def add_receive_args_to_osc_map(self, func, kwargs):
        f = self.map_func_to_dict(func, kwargs)
        f['updater'] = OSCReceiveUpdater(self.osc, f['address'], f=func, count=kwargs['count'])
        f['type'] = 'args'
        self.dict['receive'][f['name']] = f

    def add_receive_args_to_patcher(self, func):
        f = self.dict['receive'][func.__name__]
        self.patcher.add_receive_args_func(f)

    '''
    receive list
    '''

    def receive_list(self, **kwargs):
        def decorator(func):
            def wrapper(*args):
                self.add_receive_list_to_osc_map(func, kwargs)
                if self.create_patch is True:
                    self.add_receive_list_to_patcher(func)
                return func(*args)
            # TODO: This probably shouldn't be here...
            v_len, v_min, v_max = kwargs['length'], kwargs['vector'][1], kwargs['vector'][2]
            randomise_list = v_min + (np.random.rand(v_len).astype(np.float32) * (v_max - v_min))
            wrapper(randomise_list)
            return wrapper
        return decorator
    
    def add_receive_list_to_osc_map(self, func, kwargs):
        '''
        TODO: Should this support list[float] only, or list[int] list[str] etc?
        '''
        f = self.map_func_to_dict(func, kwargs)
        assert len(f['params']) == 1, "receive_list can only receive one param (list[float])"
        hint = f['hints'][list(f['params'].keys())[0]]
        assert hint == list[float], "receive_list can only receive list[float], found "+str(hint)
        f['updater'] = OSCReceiveListUpdater(self.osc, f['address'], f=func, count=kwargs['count'])
        f['type'] = 'list'
        f['length'] = kwargs['length']
        self.dict['receive'][f['name']] = f
        if self.export is not None:
            self.export_dict()
    
    def add_receive_list_to_patcher(self, func):
        f = self.dict['receive'][func.__name__]
        self.patcher.add_receive_list_func(f)

    '''
    receive kwargs
    '''

    def receive_kwargs(self, **kwargs):
        '''
        Same as receive_args but with named params
        '''
        raise NotImplementedError("receive_kwargs not implemented yet")

    '''
    xml / json export
    '''

    def export_dict(self):
        '''
        Save the OSCMap dict as XML
        '''
        client_ip, client_port = self.osc.client_names[self.client_name]
        # TODO: This should be defined in the OSCMap dict / on init
        metadata = {
            'HostIP': self.osc.host,
            'HostPort': str(self.osc.port),
            'ClientName': self.client_name,
            'ClientIP': client_ip,
            'ClientPort': str(client_port)
        }
        root = ET.Element("OpenSoundControlSchema")
        metadata_element = ET.SubElement(root, "Metadata", **metadata)
        sends = self.dict['send']
        receives = self.dict['receive']
        for io in ['Send', 'Receive']:
            ET.SubElement(root, io)
        for io in ['send', 'receive']:
            for name in self.dict[io]:
                f = self.dict[io][name]
                if f['type'] == 'args':
                    self.xml_add_args_params(root, name, io, f)
                elif f['type'] == 'list':
                    self.xml_add_list_param(root, name, io, f)
                elif f['type'] == 'kwargs':
                    raise NotImplementedError("kwargs not implemented yet")
        self.export_update(root)
    
    def xml_add_args_params(self, root, name, io, f):
        params = f['params']
        hints = f['hints']
        kw = {
            "Address": '/'+name.replace('_', '/'),
            "Type": f['type'],
            "Params": str(len(params)),
        }
        route = ET.SubElement(root.find(io.capitalize()), "Route", **kw)
        for i, p in enumerate(params):
            # TODO: This should already be defined by this point
            if io == 'receive':
                p_type = hints[p].__name__
            elif io == 'send':
                p_type = hints['return'].__args__[i].__name__
            kw = {
                "Name": p,
                "Type": p_type,
                "Default": str(params[p][0]),
                "Min": str(params[p][1]),
                "Max": str(params[p][2])
            }
            ET.SubElement(route, "Param", **kw)
    
    def xml_add_list_param(self, root, name, io, f):
        params = f['params']
        hints = f['hints']
        length = f['length']
        kw = {
            "Address": '/'+name.replace('_', '/'),
            "Type": f['type'],
            "Length": str(length),
        }
        route = ET.SubElement(root.find(io.capitalize()), "Route", **kw)
        p = list(params.keys())[0]
        if io == 'receive':
            p_type = hints[p].__name__
        elif io == 'send':
            p_type = hints['return'].__args__[0].__name__
        kw = {
            "Name": p,
            "Type": p_type,
            "Default": str(params[p][0]),
            "Min": str(params[p][1]),
            "Max": str(params[p][2])
        }
        ET.SubElement(route, "ParamList", **kw)
    
    def export_update(self, root):
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        if self.export == 'XML':
            self.save_xml(tree, root)
        elif self.export == 'JSON':
            self.save_json(root)
        elif self.export == True:
            self.save_xml(tree, root)
            self.save_json(root)

    def save_xml(self, tree, root):
        tree.write(self.patch_filepath+".xml")
        print(f"Exported OSCMap to {self.patch_filepath}.xml")

    def save_json(self, xml_root):
        # TODO: params should be `params: []` and not `param: {}, param: {}, ...`
        json_dict = self.xml_to_json(ET.tostring(xml_root, encoding='utf8', method='xml'))
        with open(self.patch_filepath+".json", 'w') as f:
            f.write(json_dict)
        print(f"Exported OSCMap to {self.patch_filepath}.json")

    def pascal_to_camel(self, pascal_str):
        return pascal_str[0].lower() + pascal_str[1:]

    def etree_to_dict(self, t):
        tag = self.pascal_to_camel(t.tag)
        d = {tag: {} if t.attrib else None}
        children = list(t)
        if children:
            dd = {}
            for dc in map(self.etree_to_dict, children):
                for k, v in dc.items():
                    try:
                        dd[k].append(v)
                    except KeyError:
                        dd[k] = [v]
            d = {tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
        if t.attrib:
            d[tag].update((self.pascal_to_camel(k), v) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[tag]['#text'] = text
            else:
                d[tag] = text
        return d

    def xml_to_json(self, xml_str):
        e = ET.ElementTree(ET.fromstring(xml_str))
        return json.dumps(self.etree_to_dict(e.getroot()), indent=4)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for k, v in self.dict['send'].items():
            if 'updater' in v:
                v['updater']()
            # v['updater']()
        for k, v in self.dict['receive'].items():
            v['updater']()
