import numpy as np
import taichi as ti
import time

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

from tolvera.particles import Particle, Particles
from tolvera.pixels import Pixels
from tolvera.utils import OSCUpdaters

@ti.dataclass
class SimpleParticleLife:
    particle: Particle

@ti.data_oriented
class SimpleParticlesLife(Particles):
    def __init__(self, x: ti.i32, y: ti.i32, max_n: ti.i32, species=3):
        super().__init__(SimpleParticleLife, x, y, max_n)
        self.species_n = species
        self.species_c = ti.Vector.field(4, ti.f32, shape=(species))
        self.init()
    def init(self):
        self.randomise(self.max_n)
        self.speciate(self.max_n)
    # @ti.kernel
    def randomise(self, n: ti.i32):
        for i in range(n):
            self.field[i] = SimpleParticleLife(
                particle=Particle(
                    pos=[self.x*np.random.rand(1)[0],
                         self.y*np.random.rand(1)[0]],
                    # pos=[(self.x/2)+200*np.random.rand(1)[0]-100,
                    #      (self.y/2)+200*np.random.rand(1)[0]-100],
                    vel=[0.0,0.0],
                    # vel=[2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)],
                    active=1, size=2, speed=1, max_speed=1.1))
    def speciate(self, n: ti.i32):
        self.species_colours()
        for i in range(n):
            s = i % self.species_n
            self.field[i].particle.species = s
            self.field[i].particle.rgba = self.species_c[s]
    # @ti.kernel
    def species_colours(self):
        for i in range(self.species_n):
            r = np.random.rand(3)
            self.species_c[i] = [r[0],r[1],r[2],1]

@ti.data_oriented
class ParticleLife():
    def __init__(self, x, y, species, attract=0.0010) -> None:
        self.x = x
        self.y = y
        self.species = species
        # TODO: how to update this?
        self.rules = ti.field(ti.f32, shape=(species, species))
        self.attract = attract
        self.init()
    # @ti.kernel
    def init(self):
        self.init_rules()
        self.simple_rules()
    # @ti.func
    def init_rules(self):
        a = self.attract
        for i in range(self.species):
            for j in range(self.species):
                r = np.random.rand(1)
                self.rules[i,j] = (r[0]*a - a/2.0)
                print('rule',i,j,self.rules[i,j])
    def simple_rules(self):
        # self.rules[0,0] =  0.5
        self.rules[0,1] =  1.0
        # self.rules[0,2] =  0.0
        # self.rules[1,0] =  0.0
        # self.rules[1,1] =  0.5
        self.rules[1,2] =  1.0
        self.rules[2,0] =  1.0
        # self.rules[2,1] =  1.0
        # self.rules[2,2] =  0.5
    @ti.kernel
    def step(self, field: ti.template()):
        for i in range(field.shape[0]):
            if field[i].particle.active > 0.0:
                self.stepinner(field, i)
    @ti.func
    def stepinner(self, field: ti.template(), i: ti.i32):
        p1 = field[i].particle
        for j in range(field.shape[0]):
            p2 = field[j].particle
            if p2.active > 0.0:
                r = self.rules[p1.species, p2.species]
                fx, fy = 0.0, 0.0
                dx = p1.pos[0] - p2.pos[0]
                dy = p1.pos[1] - p2.pos[1]
                d = ti.sqrt(dx * dx + dy * dy)
                if d > 0.0 and d < 300:
                    F = r/d
                    fx += F*dx
                    fy += F*dy
                field[i].particle.vel[0] += fx * 0.5
                field[i].particle.vel[1] += fy * 0.5
                field[i].particle.pos[0] += p1.vel[0]
                field[i].particle.pos[1] += p1.vel[1]
                # if(field[i].particle.pos[0] <= 10
                #    or field[i].particle.pos[0] >= self.x-10):
                #     field[i].particle.vel[0] *= -1
                # if(field[i].particle.pos[1] <= 10 
                #    or field[i].particle.pos[1] >= self.y-10):
                #     field[i].particle.vel[1] *= -1
        # print()
    def process(self):
        pass
    def __call__(self, field):
        self.step(field)

def main(host="127.0.0.1", port=4000):
    # ti.init()
    np.random.seed(int(time.time()))
    ti.init(arch=ti.vulkan)
    osc = OSC(host, port, verbose=False, concurrent=True)
    osc.create_client("particles", host="127.0.0.1", port=7564)
    x = 1920
    y = 1080
    n = 64
    species = 3
    s = SimpleParticlesLife(x, y, n, species)
    pl = ParticleLife(x, y, species, attract=0.00010)
    window = ti.ui.Window("Particle Life", (x, y))
    canvas = window.get_canvas()
    px = Pixels(x,y)

    # o = OSCUpdaters(osc, client="particles",
    #     receives={
    #         "/tolvera/particles/set/pos": s.osc_set_pos,
    #         "/tolvera/particles/set/vel": s.osc_set_vel
    #     }, receive_count=10,
    #     sends={
    #         "/tolvera/particles/get/pos/all": s.osc_get_pos_all
    #     }, send_count=60
    # )
    # pl(s.field)

    while window.running:
        with _lock:
            # px.diffuse()
            px.clear()
            # px.decay()
            # o()
            pl(s.field)
            s(px())
            canvas.set_image(px())
            window.show()

if __name__ == '__main__':
    main()
