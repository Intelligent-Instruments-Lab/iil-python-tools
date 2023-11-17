'''
TODO: subclass OSC? or wrap common funcs?
TODO: allow passing in external iipyper.osc instance?
'''

from iipyper import OSC as iiOSC, OSCMap

class OSC:
    def __init__(self, context, **kwargs) -> None:
        self.ctx = context
        self.kwargs = kwargs
        self.osc = kwargs.get('osc', False)
        self.init_osc(**kwargs)
        self.init_map(**kwargs)
        print(f"[Tölvera.OSC] OSC initialization complete.")
    def init_osc(self, **kwargs):
        print(f"[Tölvera.OSC] Initializing OSC...")
        self.host_ip      = kwargs.get('host', "127.0.0.1")
        self.client_ip    = kwargs.get('client', "127.0.0.1")
        self.client_name  = kwargs.get('client_name', self.ctx.name_clean)
        self.receive_port = kwargs.get('receive_port', 5001)
        self.send_port    = kwargs.get('send_port', 5000)
        self.trace        = kwargs.get('osc_trace', False)
        self.host = iiOSC(self.host_ip, self.receive_port, verbose=True, concurrent=True)
        self.host.create_client(self.client_name, self.client_ip, self.send_port)
        if self.trace:
            def trace(address, *args):
                print(f"[Tölvera.OSC.trace] '{address}' {args}")
            self.host.args('/*')(trace)
    def init_map(self, **kwargs):
        print(f"[Tölvera.OSC] Initializing OSCMap...")
        self.create_patch   = kwargs.get('create_patch', False)
        self.patch_type     = kwargs.get('patch_type', "Pd")
        self.patch_filepath = kwargs.get('patch_filepath', "tolvera_osc")
        self.export_patch   = kwargs.get('export_patch', None)
        self.map = OSCMap(self.host, self.client_name, self.patch_type, self.patch_filepath, self.create_patch, export=self.export_patch)
