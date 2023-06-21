import time
import taichi as ti

# from tolvera.particles import Particle, Particles
from tolvera.particles import Particle, Particles
from tolvera.pixels import Pixels
from tolvera.utils import OSCUpdaters, Updater

from iipyper import OSC, run

@ti.dataclass
class BoidsParams:
    separate: ti.f32
    align:    ti.f32
    cohere:   ti.f32
    radius:   ti.f32

@ti.data_oriented
class Boids():
    def __init__(self,
                 x: ti.i32,
                 y: ti.i32,
                 species=3) -> None:
        self.x = x
        self.y = y
        self.species_n = species
        self.rules = BoidsParams.field(shape=(species,species))
        self.init()
    @ti.kernel
    def init(self):
        self.init_rules()
    @ti.func
    def init_rules(self):
        for i in range(self.species_n):
            for j in range(self.species_n):
                self.rules[i,j] = BoidsParams(
                    separate= ti.random(ti.f32),
                    align   = ti.random(ti.f32), 
                    cohere  = ti.random(ti.f32),
                    radius  = ti.random(ti.f32)*300.0)
    @ti.kernel
    def step(self, field: ti.template()):
        for i in range(field.shape[0]):
            if field[i].active > 0.0:
                self.step_inner(field, i)
    @ti.func
    def step_inner(self, field: ti.template(), i: ti.i32):
        p1 = field[i]
        separate = ti.Vector([0.,0.])
        align    = ti.Vector([0.,0.])
        cohere   = ti.Vector([0.,0.])
        nearby = 0
        r = BoidsParams()
        for j in range(field.shape[0]):
            p2 = field[j]
            if i!=j and p2.active > 0.0:
                dis = field[i].dist(p2)
                dis_norm = dis.norm()
                r = self.rules[p1.species, p2.species]
                if dis_norm < r.radius:
                    field[i].vel += \
                        dis.normalized()/dis_norm * p1.max_speed
                    separate += dis
                    align    += p2.vel
                    cohere   += p2.pos
                    nearby   += 1
        if nearby != 0:
            separate = separate/nearby         * r.separate * p1.active
            align    = align/nearby            * r.align    * p1.active
            cohere   = (cohere/nearby-p1.pos)  * r.cohere   * p1.active
            field[i].nearby = nearby
            field[i].vel += (cohere+align+separate).normalized()
    def reset(self):
        self.init()
    def __call__(self, particles):
        self.step(particles.field)
    def osc_set_rule(self, i, j, separate, align, cohere, radius):
        self.rules[i,j] = BoidsParams(separate, align, cohere, radius)
    def osc_get_rule(self, i, j):
        return self.rules[i,j].to_numpy().tolist()

def main(x=1920, y=1080, n=8192, species=5, fps=120, host="127.0.0.1", receive_port=4000, send_port=5000):
    seed = int(time.time())
    ti.init(arch=ti.vulkan, random_seed=seed)
    # ti.init(random_seed=seed)
    osc = OSC(host, receive_port, verbose=False, concurrent=True)
    osc.create_client("boids", host, send_port)
    particles = Particles(x, y, n, species, wall_margin=0)
    pixels = Pixels(x, y, evaporate=0.95, fps=fps)
    boids = Boids(x, y, species)

    def reset():
        particles.reset()
        pixels.reset()
        boids.reset()
    update = Updater(reset, fps*8)

    def render():
        update()
        pixels.diffuse()
        pixels.decay()
        # particles.activity_decay()
        # pixels.clear()
        boids(particles)
        particles(pixels)

    pixels.show(render)

if __name__ == '__main__':
    run(main())
