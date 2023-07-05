import iml as iml_module
from iml import IML
from iipyper import OSC, run, _lock, repeat
import tolvera as tol
import taichi as ti
import numpy as np
import torch
import time

def main(x=1920, y=1080, n=64, species=5, fps=120, 
        host="127.0.0.1", client="127.0.0.1", receive_port=5001, send_port=5000,
        headless=False, tgt_size=8, gpu=True):
    
    seed = int(time.time())
    if gpu:
        ti.init(arch=ti.vulkan, random_seed=seed)
    else:
        ti.init(random_seed=seed)
    osc = OSC(host, receive_port, verbose=True, concurrent=True)
    client_name = "ferroneural"
    osc.create_client(client_name, client, send_port)

    ti.init(arch=ti.vulkan)
    # ti.init(random_seed=seed)
    particles = tol.Particles(x, y, n, species, wall_margin=0)
    pixels = tol.Pixels(x, y, evaporate=0.95, fps=fps)
    boids = tol.vera.Boids(x, y, species)

    d_src = n*2
    d_tgt = torch.zeros(tgt_size)

    print(d_tgt, type(d_tgt))

    ctrl = torch.zeros(d_src)
    z = torch.zeros(d_tgt)
    
    iml = IML(d_src)

    def iml_map():
        while(len(iml.pairs) < 32):
            src = torch.rand(d_src)#/(ctrl.abs()/2+1)
            tgt = z + torch.randn(d_tgt)*2#/(z.abs()/2+1)
            iml.add(src, tgt)
    iml_map()
    
    def update_pos():
        ctrl = torch.FloatTensor(particles.osc_get_pos_all()) # d_src
        ctrl = ctrl.sort().values
        z[:] = torch.from_numpy(iml.map(ctrl, k=5))#.float()
        print('update_pos')
    update = tol.utils.Updater(update_pos, 24)

    def send_tgt():
        return d_tgt.toList()
    osc_send_tgt = tol.utils.OSCSendUpdater(osc, "/tolvera/tgt", send_tgt, fps)
 
    def render():
        update()
        osc_send_tgt()
        pixels.diffuse()
        pixels.decay()
        boids(particles)
        particles(pixels)

    @repeat(0.016)
    def loop():
        render()

    # pixels.show(render)

if __name__=='__main__':
    run(main)
