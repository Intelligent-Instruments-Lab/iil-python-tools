from iml import IML
from iipyper import OSC, run, Updater, OSCSendUpdater
import tolvera as tol
import torch

def main(x=1920, y=1080, n=64, species=5, fps=120, 
        host="127.0.0.1", client="127.0.0.1", receive_port=5001, send_port=5000,
        headless=False, tgt_size=8, gpu='vulkan' # | cuda
        ):
    
    # Setup OSC
    osc = OSC(host, receive_port, verbose=True, concurrent=True)
    client_name = "tolvera-iml-example"
    osc.create_client(client_name, client, send_port)
    
    # Setup Tolvera
    tol.init(x=x, y=y, n=n, species=species, fps=fps, gpu=gpu, host=host, client=client, receive_port=receive_port, send_port=send_port, headless=headless)
    particles = tol.Particles(x, y, n, species, wall_margin=0)
    pixels = tol.Pixels(x, y, evaporate=0.95, fps=fps)
    boids = tol.vera.Boids(x, y, species)

    # Setup IML
    d_src = n*2
    d_tgt = torch.zeros(tgt_size)
    ctrl = torch.zeros(d_src)
    z = torch.zeros(tgt_size)
    iml = IML(d_src)
    def iml_map():
        while(len(iml.pairs) < 32):
            src = torch.rand(d_src)#/(ctrl.abs()/2+1)
            tgt = z + torch.randn(tgt_size)*2#/(z.abs()/2+1)
            iml.add(src, tgt)
    iml_map()
    
    def update_pos():
        ctrl = torch.FloatTensor(particles.osc_get_pos_all()) # d_src
        ctrl = ctrl.sort().values
        z[:] = torch.from_numpy(iml.map(ctrl, k=5))#.float()
    update = Updater(update_pos, 24)

    def send_tgt():
        return d_tgt.tolist()
    osc_send_tgt = OSCSendUpdater(osc, "/tolvera/tgt", send_tgt, fps)
 
    # Render loop
    def render():
        # update()
        # osc_send_tgt()
        pixels.diffuse()
        pixels.decay()
        boids(particles)
        particles(pixels)

    if headless:
        tol.headless(render)
    else:
        pixels.show(render)

if __name__=='__main__':
    run(main)
