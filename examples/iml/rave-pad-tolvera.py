import iml as iml_module
from iml import IML
from iipyper import TUI, Audio, OSC, run, _lock, Updater
import tolvera as tol
import taichi as ti
import numpy as np
import torch

import sounddevice as sd
import sys

def main(x=1920, y=1080, n=64, species=5, attractors=3, fps=120, host="127.0.0.1", receive_port=4000, send_port=5000,
        device=None,
        rave_path=None, checkpoint=None):
    
    # osc = OSC(osc_host, osc_port)

    tol.init(x=x, y=y, n=n, species=species, fps=fps)
    particles = tol.Particles(x, y, n, species)#, wall_margin=0)
    pixels = tol.Pixels(x, y, evaporate=0.95, fps=fps)
    boids = tol.vera.Boids(x, y, species)
    attractors_n = attractors
    if (attractors_n > 0):
        attractors = tol.vera.Attractors(x, y, n=attractors)
        attractors.set(0, tol.vera.Attractor(p=tol.Particle(active=1,pos=[x/2, y/2], mass=2), radius=y))
    
    rave = torch.jit.load(rave_path)

    try:
        sr = rave.sr
    except Exception:
        sr = rave.sampling_rate

    def rave_callback(
            indata: np.ndarray, outdata: np.ndarray, #[frames x channels]
            frames: int, time, status):
        with torch.inference_mode():
            outdata[:,:] = rave.decode(z[None,:,None])[:,0].T
        
    audio = Audio(
        device=device, dtype=np.float32,
        samplerate=sr, blocksize=rave.encode_params[-1], 
        callback=rave_callback)
    
    # Setup IML
    d_src = (n,2)
    d_tgt = rave.encode_params[2]
    z = torch.zeros(d_tgt)
    ctrl = torch.zeros(d_src)
    z = torch.zeros(d_tgt)
    iml = IML(d_src, emb='ProjectAndSort')
    def iml_add():
        while(len(iml.pairs) < 32):
            src = torch.rand(d_src)#/(ctrl.abs()/2+1)
            tgt = z + torch.randn(d_tgt)*2#/(z.abs()/2+1)
            iml.add(src, tgt)
    iml_add()
    
    def iml_map():
        ctrl = particles.get_pos_all_2d()
        z[:] = torch.from_numpy(iml.map(ctrl, k=5))
    iml_update = Updater(iml_map, 24)

    audio.stream.start()
    
    def render():
        iml_update()
        pixels.diffuse()
        pixels.decay()
        if (attractors_n > 0):
            particles.seeks(attractors)
        boids(particles)
        particles(pixels)

    tol.utils.render(render, pixels)

if __name__=='__main__':
    run(main)
