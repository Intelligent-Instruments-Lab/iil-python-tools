import taichi as ti
import numpy as np

from tolvera.vera.particles import Particle, Particles
from tolvera.vera.pixels import Pixels

from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

@ti.dataclass
class PhysarumParticle:
    particle:    Particle
    heading:     ti.f32
    sense_angle: ti.f32
    sense_dist:  ti.f32
    move_angle:  ti.f32
    move_dist:   ti.f32

@ti.data_oriented
class Physarum(Particles):
    def __init__(self,
                 max_n: ti.i32,
                 trail_x: ti.i32,
                 trail_y: ti.i32,
                 diffuse_amount: ti.f32 = 0.95) -> None:
        super().__init__(PhysarumParticle, trail_x, trail_y, max_n)
        self.max_n = max_n
        self.field = PhysarumParticle.field(shape=(self.max_n))
        self.x = trail_x
        self.y = trail_y
        # self.trail = Pixels.field(shape=(trail_x, trail_y))
        self.trail = Pixels(self.x, self.y, evaporate=diffuse_amount)
    @ti.kernel
    def randomise(self, n: int):
        for i in range(n):
            p = Particle(
                    pos=[self.x*ti.random(ti.f32),self.y*ti.random(ti.f32)],
                    active=1,size=2,speed=1,max_speed=2,mass=1)
            # p.rgba=[1,1,1,1]
            p.rgba=[ti.random(ti.f32),
                    ti.random(ti.f32),
                    ti.random(ti.f32),1]
            phys = PhysarumParticle(
                heading=ti.random() * 2.0 * np.pi,
                sense_angle=0.20 * np.pi,
                sense_dist=4.0,
                move_angle=0.2 * np.pi,
                move_dist=4.0,
                particle=p)
            self.field[i] = phys
    @ti.kernel
    def move(self):
        """
        Sense centre vs left vs right
        Adjust heading (ang) based on trail strength
        Update position and heading
        """
        for i in range(self.max_n):
            if self.field[i].particle.active > 0.0:
                p = self.field[i]
                pos, ang = p.particle.pos, p.heading
                c = self.sense(pos, ang, p.sense_dist).norm()
                l = self.sense(pos, ang - p.sense_angle, p.sense_dist).norm()
                r = self.sense(pos, ang + p.sense_angle, p.sense_dist).norm()
                if l < c < r:
                    ang += p.move_angle
                elif l > c > r:
                    ang -= p.move_angle
                elif r > c and c < l:
                    ang += p.move_angle * (2 * (ti.random() < 0.5) - 1) # TODO: magic numbers
                pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) #* p.move_step
                self.field[i].particle.pos = pos
                self.field[i].heading = ang
    @ti.func
    def sense(self, pos, ang, dist):
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * dist
        px = ti.cast(p[0], ti.i32) % self.x
        py = ti.cast(p[1], ti.i32) % self.y
        return self.trail.px.rgba[px, py]
    @ti.kernel
    def deposit(self):
        for i in range(self.max_n):
            if self.field[i].particle.active > 0.0:
                p = self.field[i].particle
                x = ti.cast(p.pos[0], ti.i32) % self.x
                y = ti.cast(p.pos[1], ti.i32) % self.y
                self.trail.px.rgba[x, y] += p.rgba * p.mass
    def step(self):
        self.deposit()
        self.trail.diffuse()
    def process(self):
        for i in range(self.substep):
            self.step()
            self.move()
    def __call__(self):
        self.process()
        return self.trail.px.rgba

def main(host="127.0.0.1", port=4000):
    ti.init(arch=ti.vulkan)
    osc = OSC(host, port, verbose=False, concurrent=True)
    x = 1920
    y = 1080
    n = 8192
    p = Physarum(n, x, y)
    p.randomise(n)
    # px = Pixels(x,y)
    window = ti.ui.Window("Physarum", (x, y))
    canvas = window.get_canvas()

    while window.running:
        with _lock:
            # px.set(p.trail.px) # 
            canvas.set_image(p())
            window.show()

if __name__ == '__main__':
    main()
