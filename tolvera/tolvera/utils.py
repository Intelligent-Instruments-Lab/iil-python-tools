'''
TODO: Naming: State? Options? Config? Settings?
'''

import taichi as ti
import time
from iipyper.state import _lock

class Options:
    def __init__(self, **kwargs):
        self.reset(**kwargs)
    
    def reset(self, **kwargs):
        self.x             = kwargs.get('x', 1920)
        self.y             = kwargs.get('y', 1080)
        self.n             = kwargs.get('n', 1024)
        self.species       = kwargs.get('species', 4)
        self.gpu           = kwargs.get('gpu', 'vulkan')
        self.cpu           = kwargs.get('cpu', None)
        self.fps           = kwargs.get('fps', 120)
        self.evaporate     = kwargs.get('evaporate', 0.95)
        self.seed          = kwargs.get('seed', int(time.time()))
        self.name          = kwargs.get('name', 'Tolvera')
        self.headless      = kwargs.get('headless', False)
        self.headless_rate = kwargs.get('headless_rate', 1/60)
        self.host          = kwargs.get('host', "127.0.0.1")
        self.client        = kwargs.get('client', "127.0.0.1")
        self.client_name   = kwargs.get('client_name', self.name)
        self.receive_port  = kwargs.get('receive_port', 5001)
        self.send_port     = kwargs.get('send_port', 5000)
        self.window = None
        self.canvas = None

    def init(self):
        print(f"[Tölvera] Initialising with: {vars(self)}")
        if self.cpu:
            ti.init(arch=ti.cpu, random_seed=self.seed)
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

        self.window = ti.ui.Window(self.name, (self.x, self.y), fps_limit=self.fps, show_window=not self.headless)
        self.canvas = self.window.get_canvas()
        return True

options = Options()

def _run(f, px, **kwargs):
    o = options
    while o.window.running:
        with _lock:
            if f is not None: f(**kwargs)
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
        print(f"[Tölvera] Initialised with name '{o.name}'.")
        return o

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
