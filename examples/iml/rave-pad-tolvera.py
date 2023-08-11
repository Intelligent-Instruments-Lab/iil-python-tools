import iml as iml_module
from iml import IML
from iipyper import TUI, Audio, OSC, run, _lock
import tolvera as tol
import taichi as ti
import numpy as np
import torch

import sounddevice as sd
import sys

def main(x=1920, y=1080, n=64, species=5, fps=120, host="127.0.0.1", receive_port=4000, send_port=5000,
        device=None,
        rave_path=None, checkpoint=None):
    
    # osc = OSC(osc_host, osc_port)

    ti.init(arch=ti.vulkan)
    # ti.init(random_seed=seed)
    particles = tol.Particles(x, y, n, species, wall_margin=0)
    pixels = tol.Pixels(x, y, evaporate=0.95, fps=fps)
    boids = tol.vera.Boids(x, y, species)
    
    rave = torch.jit.load(rave_path)

    d_src = n*2
    d_tgt = rave.encode_params[2]

    try:
        sr = rave.sr
    except Exception:
        sr = rave.sampling_rate

    ctrl = torch.zeros(d_src)
    z = torch.zeros(d_tgt)
    def rave_callback(
            indata: np.ndarray, outdata: np.ndarray, #[frames x channels]
            frames: int, time, status):
        with torch.inference_mode():
            outdata[:,:] = rave.decode(z[None,:,None])[:,0].T
        
    audio = Audio(
        device=device, dtype=np.float32,
        samplerate=sr, blocksize=rave.encode_params[-1], 
        callback=rave_callback)
    
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

    audio.stream.start()
    
    def render():
        update()
        pixels.diffuse()
        pixels.decay()
        # particles.activity_decay()
        # pixels.clear()
        boids(particles)
        particles(pixels)

    pixels.show(render)

if __name__=='__main__':
    run(main)
