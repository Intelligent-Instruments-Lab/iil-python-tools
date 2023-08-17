'''
TODO: send args? maybe this is only useful for sending repeated things with fixed args.
TODO: Make it easier to send non-kwarg lists
TODO: Refactor self.dict[io][name][params] to a labelled dictionary & add type hints
TODO: Load OSCMap from XML or JSON (probably involves refactoring, for the better)
TODO: Better handling of directories when saving/exporting (separate dir for xml/json?)
'''

from .osc import OSCSendUpdater, OSCSend, OSCReceiveUpdater
from .max import MaxPatcher
from .pd import PdPatcher

from typing import Any, get_type_hints
import os
import sys
import xml.etree.ElementTree as ET
import json

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

    def export_dict(self):
        '''
        Save the OSCMap dict as XML
        '''
        client_ip, client_port = self.osc.client_names[self.client_name]
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
                kw = {
                    "Address": '/'+name.replace('_', '/'),
                    "Params": str(len(f['params']))
                }
                route = ET.SubElement(root.find(io.capitalize()), "Route", **kw)
                for i, param in enumerate(f['params']):
                    kw = {
                        "Name": param,
                        "Type": f['hints'][param].__name__,
                        "Default": str(f['params'][param][0]),
                        "Min": str(f['params'][param][1]),
                        "Max": str(f['params'][param][2])
                    }
                    ET.SubElement(route, "Param", **kw)
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

    def add_func_to_osc_map(self, func, kwargs):
        n = func.__name__
        address = '/'+n.replace('_', '/')
        params = {k: v for k, v in kwargs.items() \
                    if k != 'io' and k != 'count' and k != 'send_mode'}
        hints = get_type_hints(func)
        f = {'f': func, 'address': address, 'params': params, 'hints': hints}
        if 'io' not in kwargs:
            raise ValueError(f'io must be specified for {n}')
        if 'count' not in kwargs:
            raise ValueError(f'count must be specified for {n}')
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
        if self.export is not None:
            self.export_dict()
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

