import taichi as ti
import time

from iipyper import OSC, run

from tolvera.particles import Particle, Particles
from tolvera.pixels import Pixels
from iipyper.osc import OSCUpdaters

# FIXME: doesn't work

@ti.data_oriented
class ParticleLife():
    def __init__(self, x, y, species, attract=0.0010) -> None:
        self.x = x
        self.y = y
        self.species = species
        self.rules = ti.field(ti.f32, shape=(species, species))
        self.attract = attract
        self.init()
    @ti.kernel
    def init(self):
        self.init_rules()
        self.simple_rules()
    @ti.func
    def init_rules(self):
        a = self.attract
        for i in range(self.species):
            for j in range(self.species):
                self.rules[i,j] = (ti.random(ti.f32)*a - a/2.0)
    @ti.func
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
            if field[i].active > 0.0:
                self.stepinner(field, i)
    @ti.func
    def stepinner(self, field: ti.template(), i: ti.i32):
        p1 = field[i]
        for j in range(field.shape[0]):
            p2 = field[j]
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
                field[i].vel[0] += fx * 0.5
                field[i].vel[1] += fy * 0.5
                field[i].pos[0] += p1.vel[0]
                field[i].pos[1] += p1.vel[1]
                # if(field[i].pos[0] <= 10
                #    or field[i].pos[0] >= self.x-10):
                #     field[i].vel[0] *= -1
                # if(field[i].pos[1] <= 10 
                #    or field[i].pos[1] >= self.y-10):
                #     field[i].vel[1] *= -1
        # print()
    def process(self):
        pass
    def reset(self):
        self.init()
    def __call__(self, field):
        self.step(field)

def main(host="127.0.0.1", port=4000):
    seed = int(time.time())
    ti.init(arch=ti.vulkan, random_seed=seed)
    # ti.init()
    osc = OSC(host, port, verbose=False, concurrent=True)
    # osc.create_client("particle_life", host="127.0.0.1", port=7564)
    fps = 120
    x = 1920
    y = 1080
    n = 64
    species = 3
    particles = Particles(x, y, n, species)
    particle_life = ParticleLife(x, y, species, attract=0.00010)
    pixels = Pixels(x,y,fps=fps)

    # osc_update = OSCUpdaters(osc, client="particles",
    #     receives={
    #         "/tolvera/particles/set/pos": particles.osc_set_pos,
    #         "/tolvera/particles/set/vel": particles.osc_set_vel
    #     }, receive_count=10,
    #     sends={
    #         "/tolvera/particles/get/pos/all": particles.osc_get_pos_all
    #     }, send_count=60
    # )
    # particle_life(particles.field)

    def render():
        # pixels.diffuse()
        pixels.clear()
        # pixels.decay()
        # osc_update()
        # update()
        particle_life(particles.field)
        particles(pixels())
    
    pixels.show(render)

if __name__ == '__main__':
    run(main())
