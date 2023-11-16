'''
TODO: subclass OSC? or wrap common funcs?
TODO: allow passing in external iipyper.osc instance?
'''

from iipyper import OSC, OSCMap

class OSC:
    def __init__(self, tolvera, **kwargs) -> None:
        self.tv = tolvera
        self.kwargs = kwargs
        self.osc          = kwargs.get('osc', False)
        self.host         = kwargs.get('host', "127.0.0.1")
        self.client       = kwargs.get('client', "127.0.0.1")
        self.client_name  = kwargs.get('client_name', self.tv.name_clean)
        self.receive_port = kwargs.get('receive_port', 5001)
        self.send_port    = kwargs.get('send_port', 5000)
        self.debug    = kwargs.get('osc_debug', False)
        # OSCMap
        self.create_patch   = kwargs.get('create_patch', False)
        self.patch_type     = kwargs.get('patch_type', "Pd")
        self.patch_filepath = kwargs.get('patch_filepath', "tolvera_osc")
        self.export_patch   = kwargs.get('export_patch', None)
        if self.osc is True: self.init_osc()
    def init_osc(self):
        self.osc = OSC(self.host, self.receive_port, verbose=True, concurrent=True)
        self.osc.create_client(self.client_name, self.client, self.send_port)
        self.map = OSCMap(self.osc, self.client_name, self.patch_type, self.patch_filepath, self.create_patch, export=self.export_patch)
        if self.debug:
            def debug(address, *args):
                print(f"[Tolvera.Options.debug_osc] {address} {args}")
            self.osc.args('/*')(debug)
