import typing

'''
TODO: Bela project directory name
TODO: Consistent send/receive terminology
'''

class PdPatcher:
    def __init__(self, osc, client_name="client", filepath="osc_controls", 
                 x=0.0, y=0.0, w=1600.0, h=900.0,
                 net_or_udp="udp",
                 bela=False) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h
        self.patch_objects = [f"#N canvas {x} {y} {w} {h} 12;\n"]
        self.patch_connections = []
        self.types = {
            "object": "obj",
            "message": "msg",
            "number": "floatatom",
            "symbol": "symbolatom",
            "toggle": "toggle",
            "slider": "vslider",
            "bang": "bng",
            "comment": "text",
        }
        self.patch_ids = {}
        self.osc = osc
        self.client_name = client_name
        self.client_address, self.client_port = self.osc.client_names[self.client_name]
        self.filepath = filepath
        self.net_or_udp = net_or_udp
        self.bela = bela
        self.init()
    
    def init(self):
        self.w = 5.5 # default width (scaling factor)
        self.h = 27.0 # default height (pixels)
        self.param_width = 70
        self.s_x, self.s_y = 30, 30 # sends insertion point
        self.r_x, self.r_y = 30, 530 # receives insertion point
        self.add_comment("Pd → Python", self.s_x, self.s_y)
        self.add_comment("===========", self.s_x, self.s_y+self.h/2)
        self.patch_ids['send'] = self.add_osc_send(self.osc.host, self.osc.port, self.s_x, self.s_y+self.h*2)
        self.add_comment("Python → Pd", self.r_x, self.r_y)
        self.add_comment("===========", self.r_x, self.r_y+self.h/2)
        self.patch_ids['receive'] = self.add_osc_receive(self.client_port, self.r_x, self.r_y+self.h*2)
        self.s_x+=300
        self.r_x+=300
        if self.bela: self.create_bela_main()
        self.save(self.filepath)

    def create_bela_main(self):
        if self.filepath.startswith("pd/"): abstraction = self.filepath[3:]
        with open("pd/_main.pd", 'w') as f:
            f.write(f"#N canvas {self.x} {self.y} {self.w} {self.h} 12;\n")
            f.write(f"#X obj {30} {30} {abstraction};\n")

    def get_last_id(self):
        return len(self.patch_objects)-2

    def add_box(self, box_type, x, y, box_text):
        self.patch_objects.append(f"#X {box_type} {x} {y} {box_text};\n")
        return self.get_last_id()

    def add_object(self, obj, x, y):
        return self.add_box("obj", x, y, obj)
    
    def add_msg(self, msg, x, y):
        return self.add_box("msg", x, y, msg)

    def add_comment(self, text, x, y):
        return self.add_box("text", x, y, text)
    
    def add_number(self, x, y):
        return self.add_box("floatatom", x, y, f"5 0 0 0 - - - 0")

    def connect(self, a_id, a_outlet, b_id, b_inlet):
        self.patch_connections.append(f"#X connect {a_id} {a_outlet} {b_id} {b_inlet};\n")
    
    def add_osc_send(self, host, port, x, y, send_rate_limit=100):
        loadbang_id = self.add_object("loadbang", x, y)
        y += self.h
        connect_id = self.add_msg(f"connect {host} {port}", x, y)
        y += self.h
        disconnect_id = self.add_msg("disconnect", x+10, y)
        metro_id = self.add_object(f"metro {send_rate_limit}", x+100, y)
        y += self.h
        send_rate_id = self.add_object("s rate", x+100, y)
        y += self.h
        receive_id = self.add_object("r send.to.iipyper", x+10, y)
        y += self.h
        packOSC_id = self.add_object("packOSC", x+10, y)
        y += self.h
        send_type = "netsend -u" if self.net_or_udp == "net" else "udpsend"
        send_id = self.add_object(send_type, x, y)
        y += self.h
        status_id = self.add_number(x, y)
        print_id = self.add_object("print reply.from.netreceive", x+40, y)
        # loadbang->connect->send->print
        self.connect(loadbang_id, 0, connect_id, 0)
        self.connect(connect_id, 0, send_id, 0)
        self.connect(send_id, 0, status_id, 0)
        self.connect(send_id, 1, print_id, 0)
        # loadbang->metro->send_rate
        self.connect(loadbang_id, 0, metro_id, 0)
        self.connect(metro_id, 0, send_rate_id, 0)
        # disconnect->send
        self.connect(disconnect_id, 0, send_id, 0)
        # receive->packOSC->send
        self.connect(receive_id, 0, packOSC_id, 0)
        self.connect(packOSC_id, 0, send_id, 0)
        return send_id

    def add_osc_receive(self, port, x, y):
        receive_type = f"netreceive -u {port}" if self.net_or_udp == "net" else f"udpreceive {port}"
        receive_id = self.add_object(receive_type, x, y)
        y += self.h
        unpackOSC_id = self.add_object("unpackOSC", x, y)
        y += self.h
        print_id = self.add_object("print receive.from.iipyper", x+20, y)
        y += self.h
        s_receive_id = self.add_object("s receive.from.iipyper", x, y)
        self.connect(receive_id, 0, unpackOSC_id, 0)
        self.connect(unpackOSC_id, 0, s_receive_id, 0)
        self.connect(unpackOSC_id, 0, print_id, 0)
        return self.get_last_id()

    def add_send_args_func(self, f):
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
        self.r_x += max(len(params) * self.param_width + 100.0, len(f['address'])*15.0 + 25.0)
        self.save(self.filepath)

    def add_send_list_func(self, f):
        self.add_osc_receive_list(self.r_x, self.r_y, f['address'], f['params'])
        self.r_x += len(f['address'])*15.0 + 25.0
        self.save(self.filepath)
    
    def add_receive_args_func(self, f):
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
        self.s_x += max(len(params) * self.param_width + 100.0, len(f['address'])*15.0 + 25.0)
        self.save(self.filepath)
    
    def add_receive_list_func(self, f):
        self.add_osc_send_list(self.s_x, self.s_y, f['address'], f['params'])
        self.s_x += len(f['address'])*15.0 + 25.0
        self.save(self.filepath)

    def add_osc_receive_msg(self, x, y, path):
        '''
        does this even make sense?
        '''
        receive_id = self.add_msg("r receive.from.iipyper", x, y)
        msg_id = self.add_comment(path, x, y)
        self.connect(receive_id, 0, msg_id, 0)
        return msg_id

    def add_osc_send_msg(self, x, y, path):
        msg_id = self.add_msg(path, x, y+225+self.h)
        send_id = self.add_object("s send.to.iipyper", x, y+250+self.h)
        self.connect(msg_id, 0, send_id, 0)
        return msg_id

    def add_osc_receive_with_controls(self, x, y, path, parameters):
        '''
        TODO: Does [route] need to be broken down into individual subpaths?
        '''

        # [comment path]
        y_off = 0
        path_comment_id = self.add_comment(path, x, y+y_off)
        
        # [r receive]
        y_off+=self.h
        receive_id = self.add_object("r receive.from.iipyper", x, y+y_off)

        # [route /path]
        y_off+=self.h
        route_id = self.add_object("routeOSC "+path, x, y+y_off)

        # [unpack f f f ...] [print /path]
        y_off+=self.h
        unpack_id = self.add_object("unpack "+self._pack_args(parameters), x, y+y_off)
        unpack_width = len(parameters)*7+60
        print_id = self.add_object("print "+path, x+unpack_width+10, y+y_off)

        # sliders
        y_off+=10
        slider_ids, float_ids, int_ids, tbf_ids, _y_off = self.add_sliders(x, y+y_off, parameters, "receive")
        y_off+=160

        # [s arg_name]
        y_off+=_y_off+75
        send_ids = [
            self.add_object("s "+self.path_to_snakecase(path)+'_'+p['label'][0:3], 
                            x+i*self.param_width, 
                            y+y_off+(0 if i % 2 == 0 else 25)) 
            for i, p in enumerate(parameters)]

        # [comment params]
        y_off+=50
        param_comment_ids, _y_off = self.add_param_comments(x, y+y_off, parameters)
        
        # # connections
        self.connect(receive_id, 0, route_id, 0)
        self.connect(route_id, 0, unpack_id, 0)
        self.connect(route_id, 0, print_id, 0)
        [self.connect(unpack_id, i, slider_ids[i], 0) for i in range(len(parameters))]
        [self.connect(float_ids[i], 0, send_ids[i], 0) for i in range(len(parameters))]

        return slider_ids, unpack_id

    def add_osc_send_with_controls(self, x, y, path, parameters):
        y_off = 0
        # [comment path]
        path_comment_id = self.add_comment(path, x, y+y_off)
        y_off+=15
        param_comment_ids, _y_off = self.add_param_comments(x, y+y_off, parameters)

        # [r path_arg_name]
        y_off+=35
        receive_ids = [
            self.add_object("r "+self.path_to_snakecase(path)+'_'+p['label'][0:3], 
                            x+i*self.param_width, 
                            y+y_off+(0 if i % 2 == 0 else 25)) 
            for i, p in enumerate(parameters)]
        y_off+=30

        # sliders
        slider_ids, slider_float_ids, int_ids, tbf_ids, _y_off = self.add_sliders(x, y+y_off, parameters, "send")
        y_off+=_y_off+25
        y_off+=225

        pack_id = -1
        out_id = -1
        # [pack $1 $2 $3 ...]
        if len(parameters) > 1:
            pack_id = self.add_object("pack "+self._pack_args(parameters), x, y+y_off)
            out_id = pack_id

        # [msg /path $1 $2 $3 ...]
        y_off+=25
        msg_args = self._msg_args(parameters)
        msg_id = self.add_msg(
            path+" "+msg_args, x, y+y_off)
        out_id = msg_id if len(parameters) == 1 else out_id
        # [s send]
        y_off+=25
        send_id = self.add_object("s send.to.iipyper", x, y+y_off)

        # connections
        for i in range(len(parameters)):
            rcv = receive_ids[i]
            slider = slider_ids[i]
            slider_float = slider_float_ids[i]
            int_id = int_ids[i]
            tbf_id = tbf_ids[i]

            self.connect(rcv, 0, slider, 0)
            if   int_id == -1 and tbf_id == -1: # if no int or tbf
                self.connect(slider_float, 0, out_id,       0)
            elif int_id != -1 and tbf_id == -1: # if int but no tbf
                self.connect(slider_float, 0, out_id,       0)
            elif int_id == -1 and tbf_id != -1: # if tbf but no int
                self.connect(tbf_id,       0, out_id,       0)
                self.connect(tbf_id,       1, pack_id,      i) if pack_id != -1 else None
            elif int_id != -1 and tbf_id != -1: # if both int and tbf
                self.connect(tbf_id,       0, out_id,       0)
                self.connect(tbf_id,       1, pack_id,      i) if pack_id != -1 else None

        self.connect(pack_id, 0, msg_id, 0) if pack_id != -1 else None
        self.connect(msg_id, 0, send_id, 0)
        return slider_ids, pack_id, msg_id
    
    def add_sliders(self, x, y, sliders, io=None):
        assert io is not None, "io must be \"send\" or \"receive\""
        '''
        sliders = [
          { 'label': 'x', data: 'float', min_val: 0.0, size: 0.0 },
        ]
        '''
        slider_ids = []
        float_ids = []
        int_ids = []
        tbf_ids = []
        y_off = 0
        send_rate_id = self.add_object("r rate", x-50, y+155)
        for i, s in enumerate(sliders):
            y_off = 0
            x_i = x+(i*self.param_width)
            y_off+=self.h
            slider_id, int_id, float_id, tbf_id = self.add_slider(
                send_rate_id,
                x_i, 
                y+y_off,s["min_val"], 
                s["size"], 
                float=s["data"]=="float", 
                io=io if i > 0 else "skip")
            slider_ids.append(slider_id)
            float_ids.append(float_id)
            int_ids.append(int_id)
            tbf_ids.append(tbf_id)
        return slider_ids, float_ids, int_ids, tbf_ids, y_off

    def add_slider(self, send_rate_id, x, y, min_val, size, float=False, io=None):
        assert io is not None, "io must be \"send\" or \"receive\""
        slider_id = self.add_box("obj", x, y, f"vsl 20 120 {min_val} {min_val+size} 0 0 empty empty empty 0 -9 0 12 #fcfcfc #000000 #000000 0 1")
        y += 120+8
        int_id = -1
        tbf_id = -1
        float_id = -1
        if float == False and io == "send": # if int and send
            # int -> number -> t b f
            int_id = self.add_object("int", x, y)
            y+=self.h
            float_id = self.add_number(x, y)
            y+=self.h
            zl_id = self.add_object("zl reg", x, y)
            y+=self.h
            change_id = self.add_object("change", x, y)
            y+=self.h
            tbf_id = self.add_object("t b f", x, y)
            self.connect(slider_id, 0, int_id, 0)
            self.connect(int_id, 0, float_id, 0)
            self.connect(float_id, 0, zl_id, 1)
            self.connect(send_rate_id, 0, zl_id, 0)
            self.connect(zl_id, 0, change_id, 0)
            self.connect(change_id, 0, tbf_id, 0)
        elif float == False and io != "send": # if int and not send
            # int -> number
            int_id = self.add_object("int", x, y)
            y+=self.h
            float_id = self.add_number(x, y)
            y+=self.h
            zl_id = self.add_object("zl reg", x, y)
            y+=self.h
            change_id = self.add_object("change", x, y)
            self.connect(slider_id, 0, int_id, 0)
            self.connect(int_id, 0, float_id, 0)
            self.connect(float_id, 0, zl_id, 1)
            self.connect(send_rate_id, 0, zl_id, 0)
            self.connect(zl_id, 0, change_id, 0)
        elif float == True and io == "send": # if float and send
            # number -> t b f
            float_id = self.add_number(x, y)
            y+=self.h
            zl_id = self.add_object("zl reg", x, y)
            y+=self.h
            change_id = self.add_object("change", x, y)
            y+=self.h
            tbf_id = self.add_object("t b f", x, y)
            self.connect(slider_id, 0, float_id, 0)
            self.connect(float_id, 0, zl_id, 1)
            self.connect(send_rate_id, 0, zl_id, 0)
            self.connect(zl_id, 0, change_id, 0)
            self.connect(change_id, 0, tbf_id, 0)
        elif float == True and io != "send": # if float and not send
            # number
            float_id = self.add_number(x, y)
            y+=self.h
            zl_id = self.add_object("zl reg", x, y)
            y+=self.h
            change_id = self.add_object("change", x, y)
            self.connect(slider_id, 0, float_id, 0)
            self.connect(float_id, 0, zl_id, 1)
            self.connect(send_rate_id, 0, zl_id, 0)
            self.connect(zl_id, 0, change_id, 0)
        return slider_id, int_id, change_id, tbf_id

    def add_param_comments(self, x, y, params):
        comment_ids = []
        y_off = 0
        for i, p in enumerate(params):
            y_off = 0
            x_i = x+(i*self.param_width)
            p_max = p["min_val"]+p["size"] if p["data"] == "float" else p["min_val"]+p["size"]-1
            comment_id1 = self.add_comment(f'{p["label"]}', x_i, y)
            y_off+=15
            comment_id2 = self.add_comment(f'{p["data"][0]} {p["min_val"]} {p_max}', x_i, y+y_off)
            comment_ids.append(comment_id1)
            comment_ids.append(comment_id2)
        return comment_ids, y_off
    
    def add_osc_send_list(self, x, y, path, params):
        '''
        [comment] path, list name, params
        [r] path
        [list prepend path]
        [list trim]
        [s send.to.iipyper]
        '''
        y_off=0
        self.add_comment(path, x, y)
        y_off+=15
        l = list(params.items())[0]
        self.add_comment(f"{l[0]}", x, y+y_off)
        y_off+=15
        self.add_comment(f"l {l[1][1]} {l[1][2]}", x, y+y_off)
        y_off+=self.h
        receive_id = self.add_object(f"r {self.path_to_snakecase(path)}", x, y+y_off)
        y_off+=self.h
        prepend_id = self.add_object(f"list prepend {path}", x, y+y_off)
        y_off+=self.h
        trim_id = self.add_object(f"list trim", x, y+y_off)
        y_off+=self.h
        send_id = self.add_object(f"s send.to.iipyper", x, y+y_off)
        self.connect(receive_id, 0, prepend_id, 0)
        self.connect(prepend_id, 0, trim_id, 0)
        self.connect(trim_id, 0, send_id, 0)
    
    def add_osc_receive_list(self, x, y, path, params):
        '''
        [comment] path
        [r receive.from.iipyper]
        [routeOSC path]
        [s path]
        [comment] params
        '''
        y_off=0
        self.add_comment(path, x, y)
        y_off+=self.h
        receive_id = self.add_object(f"r receive.from.iipyper", x, y+y_off)
        y_off+=self.h
        route_id = self.add_object(f"routeOSC {path}", x, y+y_off)
        y_off+=self.h
        send_id = self.add_object(f"s {self.path_to_snakecase(path)}", x, y+y_off)
        y_off+=self.h
        l = list(params.items())[0]
        self.add_comment(f"{l[0]}", x, y+y_off)
        y_off+=15
        self.add_comment(f"l {l[1][1]} {l[1][2]}", x, y+y_off)
        self.connect(receive_id, 0, route_id, 0)
        self.connect(route_id, 0, send_id, 0)

    def _pack_args(self, args):
        arg_types = []
        for a in args:
            match a["data"]:
                case "int":
                    arg_types.append("f")
                case "float":
                    arg_types.append("f")
                case "string":
                    arg_types.append("s")
        return " ".join(arg_types)

    def _msg_args(self, args):
        return " ".join(["\$"+str(i+1) for i in range(len(args))])

    def path_to_snakecase(self, path):
        return path.replace('/', '_')[1:]#+'_'+label[0:3]

    def save(self, name):
        with open(name+".pd", 'w') as f:
            [f.write(o) for o in self.patch_objects]
            [f.write(c) for c in self.patch_connections]
    
    def load(self, name):
        with open(name+".pd", 'r') as f:
            for line in f:
                if f.startswith("#X connect"):
                    self.patch_connections.append(f)
                else:
                    self.patch_objects.append(f)

