from iipyper import repeat
import time
import taichi as ti
from iipyper.state import _lock

'''
Globals
'''
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
    if headless is False:
        global window, canvas
        window = ti.ui.Window(name, (x,y), fps_limit=fps)
        canvas = window.get_canvas()

def show(px):
    global window, canvas
    canvas.set_image(px.px.rgba)
    window.show()

def render(f=None, px=None):
    # TODO: Add **kwargs to f()
    global canvas, window, headless, headless_rate
    if headless:
        @repeat(headless_rate)
        def _():
            if f is not None: f()
    else:
        while window.running:
            with _lock:
                if f is not None: f()
                show(px)

