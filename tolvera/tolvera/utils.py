'''
TODO: send args? maybe this is only useful for sending repeated things with fixed args.
'''
import json
import typing
from typing import Any

class Updater():
    '''
    Rate-limited function call
    '''

    def __init__(self, f, count=30):
        self.f = f
        self.count = count
        self.counter = 0

    def __call__(self):
        self.counter += 1
        if self.counter >= self.count:
            self.f()
            self.counter = 0

class ReceiveUpdater:
    '''
    Decouples event handling from updating
    Updating is rate-limited by a counter
    '''

    def __init__(self, f, state=None, count=5, update=False):
        self.f = f
        self.count = count
        self.counter = 0
        self.update = update
        self.state = state

    def set(self, state):
        '''
        Set the Updater's state
        '''
        self.state = state
        self.update = True

    def __call__(self):
        '''
        Update the target function with internal state
        '''
        self.counter += 1
        if not (self.update and
                self.counter > self.count and
                self.state is not None):
            return
        self.f(*self.state)
        self.counter = 0
        self.update = False


class OSCReceiveUpdater(ReceiveUpdater):
    '''
    ReceiveUpdater with an OSC handler
    '''

    def __init__(self, osc, address: str, f, state=None, count=10, update=False):
        super().__init__(f, state, count, update)
        self.osc = osc
        self.address = address
        osc.add_handler(self.address, self.receive)

    def receive(self, address, *args):
        # FIXME: ip:port/args
        '''
        v: first argument to the handler is the IP:port of the sender
        v: or you can use dispatcher.map directly
           and not set needs_reply_address=True
        j: can I get ip:port from osc itself?
        v: if you know the sender ahead of time yeah,
           but that lets you respond to different senders dynamically
        '''
        self.set(args[1:])


class OSCSendUpdater():
    '''
    Rate-limited OSC send
    '''

    def __init__(self, osc, address: str, f, count=30, client=None):
        self.osc = osc
        self.address = address
        self.f = f
        self.count = count
        self.counter = 0
        self.client = client

    def __call__(self):
        self.counter += 1
        if self.counter >= self.count:
            self.osc.send(self.address, *self.f(), client=self.client)
            self.counter = 0


class OSCReceiveUpdaters:
    '''
    o = OSCReceiveUpdaters(osc,
        {"/tolvera/particles/pos": s.osc_set_pos,
         "/tolvera/particles/vel": s.osc_set_vel})
    '''

    def __init__(self, osc, receives=None, count=10):
        self.osc = osc
        self.receives = []
        self.count = count
        if receives is not None:
            self.add_dict(receives, count=self.count)

    def add_dict(self, receives, count=None):
        if count is None:
            count = self.count
        {a: self.add(a, f, count=count) for a, f in receives.items()}

    def add(self, address, function, state=None, count=None, update=False):
        if count is None:
            count = self.count
        self.receives.append(
            OSCReceiveUpdater(self.osc, address, function,
                              state, count, update))

    def __call__(self):
        [r() for r in self.receives]


class OSCSendUpdaters:
    '''
    o = OSCSendUpdaters(osc, client="particles", count=10,
        sends={
            "/tolvera/particles/get/pos/all": s.osc_get_pos_all
        })
    '''

    def __init__(self, osc, sends=None, count=10, client=None):
        self.osc = osc
        self.sends = []
        self.count = count
        self.client = client
        if sends is not None:
            self.add_dict(sends, self.count, self.client)

    def add_dict(self, sends, count=None, client=None):
        if count is None:
            count = self.count
        if client is None:
            client = self.client
        {a: self.add(a, f, count=count, client=client)
                     for a, f in sends.items()}

    def add(self, address, function, state=None, count=None, update=False, client=None):
        if count is None:
            count = self.count
        if client is None:
            client = self.client
        self.sends.append(
            OSCSendUpdater(self.osc, address, function, count, client))

    def __call__(self):
        [s() for s in self.sends]


class OSCUpdaters:
    '''
    o = OSCUpdaters(osc, client="boids", count=10,
        receives={
            "/tolvera/boids/pos": b.osc_set_pos,
            "/tolvera/boids/vel": b.osc_set_vel
        },
        sends={
            "/tolvera/boids/pos/all": b.osc_get_all_pos
        }
    )
    '''

    def __init__(self, osc,
                 sends=None, receives=None,
                 send_count=60, receive_count=10,
                 client=None):
        self.osc = osc
        self.client = client
        self.send_count = send_count
        self.receive_count = receive_count
        self.sends = OSCSendUpdaters(
            self.osc, count=self.send_count, client=self.client)
        self.receives = OSCReceiveUpdaters(self.osc, count=self.receive_count)
        if sends is not None:
            self.add_sends(sends)
        if receives is not None:
            self.add_receives(receives)

    def add_sends(self, sends, count=None, client=None):
        if count is None:
            count = self.send_count
        if client is None:
            client = self.client
        self.sends.add_dict(sends, count, client)

    def add_send(self, send, count=None, client=None):
        if count is None:
            count = self.send_count
        if client is None:
            client = self.client
        self.sends.add(send, client=client, count=count)

    def add_receives(self, receives, count=None):
        if count is None:
            count = self.receive_count
        self.receives.add_dict(receives, count=count)

    def add_receive(self, receive, count=None):
        if count is None:
            count = self.receive_count
        self.receives.add(receive, count=count)

    def __call__(self):
        self.sends()
        self.receives()

class MaxPatcher:
    '''
    TODO: copy-paste using stdout
    TODO: add scale objects before send and after receive
    TODO: add default values via loadbangs
    TODO: move udpsend/udpreceive to the top left
    TODO: dict of object ids
    TODO: add abstraction i/o messages e.g. param names, state save/load/dumps
    '''
    def __init__(self, osc, client_name="client", filepath="osc_controls", x=0.0, y=0.0, w=1600.0, h=900.0, v='8.5.4') -> None:
        self.patch = {
            "patcher": 	{
                "fileversion": 1,
                "appversion": 		{
                    "major": v[0],
                    "minor": v[2],
                    "revision": v[4],
                    "architecture": "x64",
                    "modernui": 1
                },
                "classnamespace": "box",
                "rect": [x, y, w, h],
                "bglocked": 0,
                "openinpresentation": 0,
                "default_fontsize": 12.0,
                "default_fontface": 0,
                "default_fontname": "Arial",
                "gridonopen": 1,
                "gridsize": [15.0, 15.0],
                "gridsnaponopen": 1,
                "objectsnaponopen": 1,
                "statusbarvisible": 2,
                "toolbarvisible": 1,
                "lefttoolbarpinned": 0,
                "toptoolbarpinned": 0,
                "righttoolbarpinned": 0,
                "bottomtoolbarpinned": 0,
                "toolbars_unpinned_last_save": 0,
                "tallnewobj": 0,
                "boxanimatetime": 200,
                "enablehscroll": 1,
                "enablevscroll": 1,
                "devicewidth": 0.0,
                "description": "",
                "digest": "",
                "tags": "",
                "style": "",
                "subpatcher_template": "",
                "assistshowspatchername": 0,
                "boxes": [],
                "lines": [],
                "dependency_cache": [],
                "autosave": 0
            }
        }
        self.types = {
            "print": "print",
            "message": "message",
            "object": "newobj",
            "comment": "comment",
            "slider": "slider",
            "float": "flonum",
            "int": "number",
            "bang": "button",
        }
        self.osc = osc
        self.client_name = client_name
        self.client_address, self.client_port = self.osc.client_names[self.client_name]
        self.filepath = filepath
        self.init()

    def init(self):
        self.w = 5.5 # default width (scaling factor)
        self.h = 22.0 # default height (pixels)
        self.s_x, self.s_y = 30, 125 # insertion point
        self.r_x, self.r_y = 30, 575 # insertion point
        self.patcher_ids = {}
        self.patcher_ids['send_id'] = self.add_osc_send(self.osc.host, self.osc.port, self.s_x, 30, print_label="sent")
        self.patcher_ids['receive_id'] = self.add_osc_receive(self.client_port, self.s_x+150, 30, print_label="received")
        self.add_comment("Max → Python", self.s_x, self.s_y, 24)
        self.add_comment("Python → Max", self.r_x, self.r_y, 24)
        self.s_y+=50
        self.r_y+=50
        self.save(self.filepath)

    def add_box(self, box_type, inlets, outlets, x, y, w, h=None):
        if h is None: h = self.h
        box_id, box = self.create_box(box_type, inlets, outlets, x, y, w, h)
        return self._add_box(box)

    def _add_box(self, box):
        self.patch["patcher"]["boxes"].append(box)
        return self.id_from_str(box["box"]["id"])

    def create_box(self, box_type, inlets, outlets, x, y, w, h=None):
        if h is None: h = self.h
        box_id = len(self.patch["patcher"]["boxes"])+1
        box = {"box": {
            "id": 'obj-'+str(box_id),
            "maxclass": self.types[box_type],
            "numinlets": inlets, "numoutlets": outlets,
            "patching_rect": [x, y, w, h]
        }}
        if outlets > 0:
            if outlets == 1:
                box["box"]["outlettype"] = [""]
            match box_type:
                case "int" | "float" | "bang":
                    box["box"]["outlettype"] = ["", "bang"]
        return box_id, box

    def add_object(self, text, inlets, outlets, x, y):
        box_id, box = self.create_box(
            "object", inlets, outlets, x, y, len(text)*self.w)
        box["box"]["text"] = text
        self._add_box(box)
        return box_id

    def add_message(self, text, x, y):
        box_id, box = self.create_box("message", 2, 1, x, y, len(text)*self.w)
        box["box"]["text"] = text
        self._add_box(box)
        return box_id
    
    def add_comment(self, text, x, y, fontsize=12):
        box_id, box = self.create_box("comment", 0, 0, x, y, len(text)*self.w)
        box["box"]["text"] = text
        box["box"]["fontsize"] = fontsize
        self._add_box(box)
        return box_id
    
    def add_bang(self, x, y):
        box_id, box = self.create_box("bang", 1, 1, x, y, 20.0)
        self._add_box(box)
        return box_id

    def add_slider(self, x, y, min_val, size, float=False):
        box_id, box = self.create_box("slider", 1, 1, x, y, 20.0, 140.0)
        if float:
            box["box"]["floatoutput"] = 1
        box["box"]["min"] = min_val
        box["box"]["size"] = size
        self._add_box(box)
        return box_id

    def connect(self, src, src_outlet, dst, dst_inlet):
        patchline = {"patchline": {
            "destination": ['obj-'+str(dst), dst_inlet],
            "source": ['obj-'+str(src), src_outlet]
        }}
        self.patch["patcher"]["lines"].append(patchline)
        return patchline

    def save(self, name):
        with open(name+".maxpat", "w") as f:
            f.write(json.dumps(self.patch, indent=2))

    def load(self, name):
        with open(name+".maxpat", "r") as f:
            self.patch = json.loads(f.read())

    def get_box_by_id(self, id):
        for box in self.patch["patcher"]["boxes"]:
            if self.id_from_str(box["box"]["id"]) == id:
                return box
        return None
    
    def str_from_id(self, id):
        return 'obj-'+str(id)

    def id_from_str(self, obj_str):
        return int(obj_str[4:])

    def add_osc_send(self, ip, port, x, y, print=True, print_label=None):
        box_id_0 = self.add_object("r send", 0, 1, x, y)
        box_id = self.add_object("udpsend "+ip+" "+str(port), 1, 0, x, y+25)
        if print:
            text = "print" if print_label is None else "print "+print_label
            print_id = self.add_object(text, 1, 0, x+50, y)
            self.connect(box_id_0, 0, box_id, 0)
            self.connect(box_id_0, 0, print_id, 0)
            return box_id_0
        return box_id
    
    def add_osc_receive(self, port, x, y, print=True, print_label=None):
        box_id_0 = self.add_object("s receive", 0, 1, x, y+25)
        box_id = self.add_object("udpreceive "+str(port), 1, 1, x, y)
        if print:
            text = "print" if print_label is None else "print "+print_label
            print_id = self.add_object(text, 1, 0, x+60, y+25)
            self.connect(box_id, 0, print_id, 0)
            self.connect(box_id, 0, box_id_0, 0)
            return box_id_0
        return box_id
    
    def add_osc_route(self, port, x, y, print=True, print_label=None):
        '''
        [route path]
        [s name] [print]
        [unpack] ?
        [r name]
        '''
        pass

    def add_sliders(self, x, y, sliders):
        '''
        sliders = [
          { 'label': 'x', data: 'float', min_val: 0.0, size: 0.0 },
        ]
        
        [slider] ...
        |
        [number] ...
        '''
        slider_ids = []
        float_ids = []
        y_off = 0
        for i, s in enumerate(sliders):
            y_off = 0
            x_i = x+(i*52.0)
            y_off+=self.h
            slider_id = self.add_slider(x_i, y+y_off,s["min_val"], s["size"], float=s["data"]=="float")
            y_off+=150
            float_id = self.add_box("float", 1, 2, x_i, y+y_off, 50)
            slider_ids.append(slider_id)
            float_ids.append(float_id)
        return slider_ids, float_ids, y_off

    def add_param_comments(self, x, y, params):
        comment_ids = []
        y_off = 0
        for i, p in enumerate(params):
            y_off = 0
            x_i = x+(i*52.0)
            p_max = p["min_val"]+p["size"] if p["data"] == "float" else p["min_val"]+p["size"]-1
            comment_id1 = self.add_comment(f'{p["label"]}', x_i, y)
            y_off+=15
            comment_id2 = self.add_comment(f'{p["data"][0]} {p["min_val"]}-{p_max}', x_i, y+y_off)
            comment_ids.append(comment_id1)
            comment_ids.append(comment_id2)
        return comment_ids, y_off
    
    def add_osc_send_msg(self, x, y, path):
        msg_id = self.add_message(path, x, y+225+self.h)
        send_id = self.add_object("s send", 1, 0, x, y+250+self.h)
        self.connect(msg_id, 0, send_id, 0)
        return msg_id

    def add_osc_receive_msg(self, x, y, path):
        receive_id = self.add_object("r receive", 0, 1, x, y+225+self.h)
        msg_id = self.add_message(path, x, y+250+self.h)
        self.connect(receive_id, 0, msg_id, 0)
        return msg_id

    def add_osc_send_with_controls(self, x, y, path, parameters):
        # TODO: add default param value and a loadbang
        '''
        [comment path]
        [comment args]
        [r path_arg_name]
        sliders
        |                   |
        [pack $1 $2 $3 ...] [bang]
        |
        [msg /path $1 $2 $3 ...]
        |
        [s send]
        '''
        y_off = 0
        # [comment path]
        path_comment_id = self.add_comment(path, x, y+y_off)
        y_off+=15
        param_comment_ids, _y_off = self.add_param_comments(x, y+y_off, parameters)

        # [r path_arg_name]
        y_off+=35
        receive_ids = [self.add_object("r "+path.replace('/', '_')[1:]+'_'+p['label'][0:3], 1, 0, x+i*52.0, y+y_off+(0 if i % 2 == 0 else 25)) for i, p in enumerate(parameters)]
        y_off+=30

        # sliders
        slider_ids, slider_float_ids, _y_off = self.add_sliders(x, y+y_off, parameters)
        y_off+=_y_off+50
        # [pack $1 $2 $3 ...] [bang]
        pack_id = self.add_object(
            "pack "+self._pack_args(parameters), len(parameters)+1, 1, x, y+y_off)
        pack_width = self.get_box_by_id(pack_id)["box"]["patching_rect"][2]
        bang_id = self.add_bang(x+pack_width+10, y+y_off)
        # [msg /path $1 $2 $3 ...]
        y_off+=25
        msg_id = self.add_message(
            path+" "+self._msg_args(parameters), x, y+y_off)
        # [s send]
        y_off+=25
        send_id = self.add_object("s send", 1, 0, x, y+y_off)
        # connections
        [self.connect(receive_ids[i], 0, slider_ids[i], 0) for i in range(len(parameters))]
        [self.connect(slider_ids[i], 0, slider_float_ids[i], 0) for i in range(len(parameters))]
        [self.connect(slider_ids[i+1], 0, bang_id, 0) for i in range(len(parameters)-1)]
        [self.connect(slider_float_ids[i], 0, pack_id, i) for i in range(len(parameters))]
        self.connect(bang_id, 0, pack_id, 0)
        self.connect(pack_id, 0, msg_id, 0)
        self.connect(msg_id, 0, send_id, 0)
        return slider_ids, pack_id, msg_id
    
    def add_osc_receive_with_controls(self, x, y, path, parameters):
        # TODO: add default param value and a loadbang
        '''
        [comment path]
        [r receive]
        |
        [route /path]
        |                  |
        [unpack f f f ...] [print /path]
        |
        [slider] ...
        |
        [number] ...
        |
        [s arg_name]
        [comment path_arg_name]
        [comment type min-max]
        '''
        # [comment path]
        y_off = 0
        path_comment_id = self.add_comment(path, x, y+y_off)
        
        # [r receive]
        y_off+=25
        receive_id = self.add_object("r receive", 0, 1, x, y+y_off)

        # [route /path]
        y_off+=25
        route_id = self.add_object("route "+path, 1, 1, x, y+y_off)

        # [unpack f f f ...] [print /path]
        y_off+=25
        unpack_id = self.add_object("unpack "+self._pack_args(parameters), len(parameters)+1, 1, x, y+y_off)
        unpack_width = self.get_box_by_id(unpack_id)["box"]["patching_rect"][2]
        print_id = self.add_object("print "+path, 1, 0, x+unpack_width+10, y+y_off)

        # sliders
        y_off+=10
        slider_ids, float_ids, _y_off = self.add_sliders(x, y+y_off, parameters)

        # [s arg_name]
        y_off+=_y_off+25
        send_ids = [self.add_object("s "+path.replace('/', '_')[1:]+'_'+p['label'][0:3], 1, 0, x+i*52.0, y+y_off+(0 if i % 2 == 0 else 25)) for i, p in enumerate(parameters)]

        # [comment params]
        y_off+=50
        param_comment_ids, _y_off = self.add_param_comments(x, y+y_off, parameters)
        
        # connections
        self.connect(receive_id, 0, route_id, 0)
        self.connect(route_id, 0, unpack_id, 0)
        self.connect(route_id, 0, print_id, 0)
        [self.connect(unpack_id, i, slider_ids[i], 0) for i in range(len(parameters))]
        [self.connect(slider_ids[i], 0, float_ids[i], 0) for i in range(len(parameters))]
        [self.connect(float_ids[i], 0, send_ids[i], 0) for i in range(len(parameters))]

        return slider_ids, unpack_id

    def add_send_func(self, f):
        hints = typing.get_type_hints(f['f'])['return'].__args__
        f_p = f['params']
        params = []
        if len(f_p) == 0:
            self.add_osc_receive_msg(self.r_x, self.r_y, f['address'])
        else:
            for i, p in enumerate(f_p):
                p_def, p_min, p_max = f_p[p][0], f_p[p][1], f_p[p][2]
                params.append({
                    "label":   p,     "data": hints[i].__name__, 
                    "min_val": p_min, "size": p_max-p_min
                })
            self.add_osc_receive_with_controls(self.r_x, self.r_y, f['address'], params)
        self.r_x += max(len(params) * 52.0 + 100.0, len(f['address'])*6.0 + 25.0)
        self.save(self.filepath)

    def add_receive_func(self, f):
        hints = typing.get_type_hints(f['f'])
        f_p = f['params']
        params = []
        if len(f_p) == 0:
            self.add_osc_send_msg(self.s_x, self.s_y, f['address'])
        else:
            for p in f_p:
                p_def, p_min, p_max = f_p[p][0], f_p[p][1], f_p[p][2]
                params.append({
                    "label":   p,     "data": hints[p].__name__, 
                    "min_val": p_min, "size": p_max-p_min
                })
            self.add_osc_send_with_controls(self.s_x, self.s_y, f['address'], params)
        self.s_x += max(len(params) * 52.0 + 100.0, len(f['address'])*6.0 + 25.0)
        self.save(self.filepath)

    def _msg_args(self, args):
        return " ".join(["$"+str(i+1) for i in range(len(args))])

    def _pack_args(self, args):
        arg_types = []
        for a in args:
            match a["data"]:
                case "int":
                    arg_types.append("i")
                case "float":
                    arg_types.append("f")
                case "string":
                    arg_types.append("s")
        return " ".join(arg_types)

class OSCMap:
    '''
    OSCMap maps OSC messages to functions
    It creates a Max/MSP patcher that can be used to control the OSCMap
    It uses OSCSendUpdater and OSCReceiveUpdater to decouple incoming messages
    TODO: make max_patch optional
    '''
    def __init__(self, osc, client_name="client", max_patch_filepath="osc_controls") -> None:
        self.osc = osc
        self.client_name = client_name
        self.client_address, self.client_port = self.osc.client_names[self.client_name]
        self.dict = {'send': {}, 'receive': {}}
        self.patcher_filepath = max_patch_filepath
        self.patcher = MaxPatcher(osc, client_name, max_patch_filepath)

    def add_func_to_osc_map(self, func, kwargs):
        n = func.__name__
        address = '/'+n.replace('_', '/')
        params = {k: v for k, v in kwargs.items() \
                    if k != 'io' and k != 'count'}
        f = {'f': func, 'address': address, 'params': params}
        if 'io' not in kwargs:
            raise ValueError('io must be specified')
        if 'count' not in kwargs:
            raise ValueError('count must be specified')
        if kwargs['io'] == 'send':
            f['updater'] = OSCSendUpdater(self.osc, address, f=func, count=kwargs['count'], client=self.client_name)
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

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        for k, v in self.dict['send'].items():
            v['updater']()
        for k, v in self.dict['receive'].items():
            v['updater']()
