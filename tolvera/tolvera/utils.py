'''
TODO: globals better pattern needed
TODO: render() catch other exceptions?
'''
import time
import taichi as ti
from iipyper.state import _lock

# Globals
gpu = 'vulkan'
cpu = None
x = 1920
y = 1080
fps = 120
name = 'tolvera'
window = None
canvas = None
headless = False
headless_rate = 1/60
seed = int(time.time())
n = 1024
species = 4
evaporate = 0.95
host = "127.0.0.1"
client = "127.0.0.1"
client_name = name
receive_port = 5001
send_port = 5000

def init(**kwargs):
    global gpu, cpu, x, y, fps, name, headless, headless_rate, seed, n, species, evaporate, host, client, client_name, receive_port, send_port
    print(f"kwargs: {kwargs}")

    for key, value in kwargs.items():
        match key:
            case 'gpu':
                gpu = value
            case 'cpu':
                cpu = value
            case 'x':
                x = value
            case 'y':
                y = value
            case 'fps':
                fps = value
            case 'name':
                name = value
            case 'headless':
                headless = value
            case 'headless_rate':
                headless_rate = value
            case 'seed':
                seed = value
            case 'n':
                n = value
            case 'species':
                species = value
            case 'evaporate':
                evaporate = value
            case 'host':
                host = value
            case 'client':
                client = value
            case 'client_name':
                client_name = value
            case 'receive_port':
                receive_port = value
            case 'send_port':
                send_port = value
            case _:
                assert False, f"Invalid key: {key}"
    if cpu:
        ti.init(arch=ti.cpu, random_seed=seed)
        print("Tolvera running on cpu")
    else: 
        match gpu:
            case "vulkan":
                ti.init(arch=ti.vulkan, random_seed=seed)
            case "cuda":
                ti.init(arch=ti.cuda, random_seed=seed)
            case _:
                assert False, f"Invalid GPU: {gpu}"
        print(f"Tolvera running on {gpu}")
    global window, canvas
    window = ti.ui.Window(name, (x,y), fps_limit=fps, show_window=not headless)
    canvas = window.get_canvas()

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

def show(px):
    global window, canvas, headless
    canvas.set_image(px.px.rgba)
    if not headless: window.show()

def run(f, px):
    global window
    # TODO: Add **kwargs to f()
    while window.running:
        with _lock:
            if f is not None: f()
            show(px)

def stop():
    print(f"\nExiting {name}...")
    for f in _cleanup_fns:
        f()
    exit(0)

def render(f=None, px=None):
    try: run(f, px)
    except KeyboardInterrupt: stop()
