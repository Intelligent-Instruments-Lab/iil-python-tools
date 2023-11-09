'''
TODO: global scalars for speed, i: global frame index???
TODO: async runner based on sardine @s[wim?
TODO: implement reset()
TODO: particles_per_species only due to way species are allocated to particles
TODO: turn Options into a Singleton?
    decouple tolvera globals from taichi globals first?
'''

from typing import Any
import taichi as ti
from taichi._lib.core.taichi_python import DataType
import time
from iipyper import OSC, OSCMap
from iipyper.state import _lock
from sys import exit
import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

class Options:
    def __init__(self, **kwargs):
        self.reset(**kwargs)

    def reset(self, **kwargs):
        # Tölvera
        self.x         = kwargs.get('x', 1920)
        self.y         = kwargs.get('y', 1080)
        self.particles = kwargs.get('particles', 1024)
        self.species   = kwargs.get('species', 4)
        self.substep   = kwargs.get('substep', 1)
        self.evaporate = kwargs.get('evaporate', 0.95)
        self.particles_per_species = self.particles // self.species
        self.s = self.species
        self.p = self.particles
        # Taichi
        self.gpu      = kwargs.get('gpu', 'vulkan')
        self.cpu      = kwargs.get('cpu', None)
        self.fps      = kwargs.get('fps', 120)
        self.seed     = kwargs.get('seed', int(time.time()))
        self.name     = kwargs.get('name', 'Tölvera')
        self.name_clean = remove_accents(self.name).strip().lower()
        self.headless = kwargs.get('headless', False)
        # OSC
        self.osc          = kwargs.get('osc', False)
        self.host         = kwargs.get('host', "127.0.0.1")
        self.client       = kwargs.get('client', "127.0.0.1")
        self.client_name  = kwargs.get('client_name', self.name_clean)
        self.receive_port = kwargs.get('receive_port', 5001)
        self.send_port    = kwargs.get('send_port', 5000)
        self.osc_debug    = kwargs.get('osc_debug', False)
        # OSCMap
        self.create_patch   = kwargs.get('create_patch', False)
        self.patch_type     = kwargs.get('patch_type', "Pd")
        self.patch_filepath = kwargs.get('patch_filepath', "tolvera_osc")
        self.export_patch   = kwargs.get('export_patch', None)

    def init(self):
        if self.osc: self.init_osc()
        self.init_ti()
        self.init_ui()
        print(f"[Tölvera] Initialised with: {vars(self)}")
        return True
    
    def init_osc(self):
        self.osc = OSC(self.host, self.receive_port, verbose=True, concurrent=True)
        self.osc.create_client(self.client_name, self.client, self.send_port)
        self.osc_map = OSCMap(self.osc, self.client_name, self.patch_type, self.patch_filepath, self.create_patch, export=self.export_patch)
        if self.osc_debug:
            def debug_osc(address, *args):
                print(f"[Tolvera.Options.debug_osc] {address} {args}")
            self.osc.args('/*')(debug_osc)

    def init_ti(self):
        if self.cpu:
            ti.init(arch=ti.cpu, random_seed=self.seed)
            self.gpu = None
            print("[Tölvera] Running on CPU")
        else:
            if self.gpu == "vulkan":
                ti.init(arch=ti.vulkan, random_seed=self.seed)
            elif self.gpu == "cuda":
                ti.init(arch=ti.cuda, random_seed=self.seed)
            else:
                print(f"[Tölvera] Invalid GPU: {self.gpu}")
                return False
            print(f"[Tölvera] Running on {self.gpu}")
    
    def init_ui(self):
        self.window = ti.ui.Window(self.name, (self.x, self.y), fps_limit=self.fps, show_window=not self.headless)
        self.canvas = self.window.get_canvas()

options = Options()

def _run(f, px, **kwargs):
    o = options
    while o.window.running:
        with _lock:
            if f is not None: f(**kwargs)
            if o.osc is not False: o.osc_map()
            _show(px)

def _show(px):
    o = options
    o.canvas.set_image(px.px.rgba)
    if not o.headless: o.window.show()

def _stop():
    o = options
    print(f"\n[Tölvera] Exiting {o.name}...")
    for f in _cleanup_fns:
        f()
    exit(0)

def init(**kwargs):
    o = options
    o.reset(**kwargs)
    if o.init():
        return o
    else:
        print("[Tölvera] Error: failed to initialise")
        exit(0)

def render(f=None, px=None, **kwargs):
    try: _run(f, px, **kwargs)
    except KeyboardInterrupt: _stop()

_cleanup_fns = []
# decorator to make a function run on KeyBoardInterrupt (before exit)
def cleanup(f=None):
    """
    @cleanup decorator based on iipyper
    cleanup functions must be defined before render is called!
    """
    def decorator(f):
        _cleanup_fns.append(f)
        return f

    if f is None: # return a decorator
        return decorator
    else: #bare decorator case; return decorated function
        return decorator(f)

# @ti.data_oriented
class CONSTS:
    '''
    Dict of CONSTS that can be used in Taichi scope
    '''
    def __init__(self, dict: dict[str, (DataType, Any)]):
        self.struct = ti.types.struct(**{k: v[0] for k, v in dict.items()})
        self.consts = self.struct(**{k: v[1] for k, v in dict.items()})
    def __getattr__(self, name):
        try:
            return self.consts[name]
        except:
            raise AttributeError(f"CONSTS has no attribute {name}")
    def __getitem__(self, name):
        try:
            return self.consts[name]
        except:
            raise AttributeError(f"CONSTS has no attribute {name}")
