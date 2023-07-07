from iipyper import repeat
import time
import taichi as ti

def init(gpu='vulkan', cpu=None):
    seed = int(time.time())
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

def headless(r, rate=1/60):
    '''
    Repeat the render function 
    '''
    @repeat(rate)
    def _():
        r()
