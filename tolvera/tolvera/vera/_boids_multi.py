# original: https://forum.taichi-lang.cn/t/homework0-boids/563/6

import taichi as ti
import numpy as np
import math
import time

from tolvera.vera._particle import Particles
from tolvera.vera._obstacles import Obstacles

# TODO: Add docstrings
# TODO: abstract out 'fear'
# TODO: implement 'avoid'

# TODO: Replace 0D fields with dataclass?
# @ti.dataclass
# class BoidsParams:
#     separate: ti.f32
#     align: ti.f32
#     cohere: ti.f32
#     fear: ti.f32
#     dt: ti.f32
#     radius: ti.f32
#     speed: ti.f32
#     size: ti.f32

@ti.data_oriented
class BoidsMulti(Particles):
    def __init__(self,
                 x=1024,
                 y=1024,
                 n=1024,
                 species=3,
                 separate=0.1,
                 align=1.0,
                 cohere=1.0,
                 fear=10.0,
                 dt=1,
                 radius=40.0,
                 speed=3.0,
                 colormode='rgb',
                 size=2):
        super().__init__(x, y, n)
        self._vel = ti.Vector.field(2, dtype=ti.f32, shape=(self._n))
        self._species_n = species
        self._species = ti.field(dtype=ti.i32, shape=(1, self._n))
        self.px_g = ti.field(dtype=ti.f32, shape=(1, self._x, self._y))
        self.px_rgb = ti.Vector.field(3, dtype=ti.f32, shape=(self._x, self._y))
        self.px_rgba = ti.Vector.field(4, dtype=ti.f32, shape=(self._x, self._y))
        self._palette = ti.Vector.field(4, ti.f32, shape=(self._species_n, ))
        self.colormode = colormode
        self.separate = ti.field(dtype=ti.f32, shape=(self._species_n))
        self.align    = ti.field(dtype=ti.f32, shape=(self._species_n))
        self.cohere   = ti.field(dtype=ti.f32, shape=(self._species_n))
        self.fear     = ti.field(dtype=ti.f32, shape=(self._species_n))
        self.dt       = ti.field(dtype=ti.f32, shape=(self._species_n))
        self.radius   = ti.field(dtype=ti.f32, shape=(self._species_n))
        self.speed    = ti.field(dtype=ti.f32, shape=(self._species_n))
        self.size     = ti.field(dtype=ti.f32, shape=(self._species_n))
        self.separate.fill(separate)
        self.align.fill(align)
        self.cohere.fill(cohere)
        self.fear.fill(fear)
        self.dt.fill(dt)
        self.radius.fill(radius)
        self.speed.fill(speed)
        self.size.fill(size)
        self.init()

    def init(self):
        np.random.seed(int(time.time()))
        self.differentiate_species()
        self.randomise()

    @ti.kernel
    def randomise(self):
        for b in range(self._n):
            species = b % self._species_n
            self._species[0, b] = species
            self._pos[b] = ti.Vector([ti.random(ti.f32) * self._x, ti.random(ti.f32) * self._y])
            self._vel[b] = ti.Vector([self.speed[species] * (ti.random(ti.f32)-0.5), self.speed[species] * (ti.random(ti.f32)-0.5)])

    # @ti.kernel
    def differentiate_species(self):
        for s in range(self._species_n):
            r = np.random.rand(11)
            self._palette[s] = ti.Vector([r[0],r[1],r[2], 1.0])
            self.separate[s] = 1.0  * r[3]  + 0.3
            self.align[s]    = 1.0  * r[4]  + 0.3
            self.cohere[s]   = 1.0  * r[5]  + 0.3
            self.fear[s]     = 50.0 * r[6]  + 0
            self.dt[s]       = 3.0  * r[7]  + 0.2
            self.radius[s]   = 100  * r[8]  + 5
            self.speed[s]    = 2.0  * r[9]  + 0.2
            self.size[s]     = 3.0  * r[10] + 3

    @ti.kernel
    def move(self):
        """
        Apply updated velocities to positions
        """
        # TODO: merging this into step loop doesn't improve perf?
        # TODO: abstract out walls vs wrap, re: membranes
        for b in range(self._n):
            self._pos[b] = self._pos[b] + self.dt[self._species[0, b]] * self._vel[b]
            x = self._pos[b][0]
            y = self._pos[b][1]
            if   (x > self._x): self._pos[b][0] = 1
            elif (y > self._y): self._pos[b][1] = 1
            elif (x < 0):       self._pos[b][0] = self._x-1
            elif (y < 0):       self._pos[b][1] = self._y-1

    @ti.kernel
    def step(self):
        for b in range(self._n):
            self._step(b) # parallelize inner step loop

    @ti.func
    def _step(self, b: int):
        # TODO: neighbour search should wrap around edges?
        # TODO: do these behaviours need to eventually be fields?
        separate = ti.Vector([0.,0.])
        align    = ti.Vector([0.,0.])
        cohere   = ti.Vector([0.,0.])
        sep_species = ti.Vector([0.,0.])
        n = 0 # neighbour count
        for bn in range(self._n): # boid neighbour
            if b!=bn: # if not me
                dis = self._pos[b] - self._pos[bn]
                dis_norm = dis.norm()
                if dis_norm < self.radius[self._species[0,b]]:
                    # TODO: refactor so that behaviour/speed ops grouped?
                    self._vel[b] += dis.normalized()/dis_norm * self.speed[self._species[0,b]]
                    if (self._species[0,b] == self._species[0,bn]):
                        separate += (self._pos[b] - self._pos[bn])
                        align  += self._vel[bn]
                        cohere += self._pos[bn]
                        n += 1
                    else:
                        sep_species += (self._pos[b] - self._pos[bn]) / dis_norm
        if n != 0:
            separate = separate/n * self.separate[self._species[0,b]]
            align    = align/n    * self.align[self._species[0,b]]
            cohere   = (cohere/n - self._pos[b]) * self.cohere[self._species[0,b]]
            sep_species = sep_species/n * self.fear[self._species[0,b]]
            self._vel[b] += (cohere + align + separate + sep_species).normalized()
            self.limit_speed(b)

    @ti.func
    def limit_speed(self, b: int):
        if self._vel[b].norm() > self.speed[self._species[0,b]]:
            self._vel[b] = self._vel[b].normalized()*self.speed[self._species[0,b]]

    @ti.kernel
    def avoid(self, obs: ti.template()):
        for b in range(self._n):
            self._avoid(b, obs)

    @ti.func
    def _avoid(self, b: int, obs: ti.template()):
        obs_to_avoid = [] # ti.field?
        for o in obs:
            # dis = self._pos[b] - o._pos
            # dis_norm = dis.norm()
            # if dis_norm <= self._fov[b] # field of view
            #   obs_to_avoid.append(o)
            pass
        for o in obs_to_avoid:
            # self._vel[b] = ... # update velocity to avoid
            pass

    @ti.kernel
    def render_g(self):
        for i,j in ti.ndrange((0, self._x),(0, self._y)):
            self.px_g[0, i, j] = 0.0
        for b in range(self._n):
            xi = ti.cast(self._pos[b][0], ti.i32) - self.size[self._species[0,b]]
            xj = ti.cast(self._pos[b][0], ti.i32) + self.size[self._species[0,b]]
            yi = ti.cast(self._pos[b][1], ti.i32) - self.size[self._species[0,b]]
            yj = ti.cast(self._pos[b][1], ti.i32) + self.size[self._species[0,b]]
            for x in range(xi, xj):
                for y in range(yi, yj):
                    # TODO: grascaling
                    # _x = self._vel[i][0]
                    # _y = self._vel[i][1]
                    # avg = (_x+_y)/2
                    # scaled = (3+avg)/6
                    # self.px_g[0, x, y] = 1-scaled
                    self.px_g[0, x, y] = self._palette[self._species[0, b]][0] *0.5 + 0.5#1.0

    @ti.kernel
    def render_rgb(self):
        # TODO: Fix rgb vs rgba
        for i,j in ti.ndrange((0, self._x),(0, self._y)):
            # self.px_rgb[i, j] = ti.Vector([0.0,0.0,0.0])
            self.px_rgba[i, j] = ti.Vector([0.0,0.0,0.0,1.0])
        for b in range(self._n):
            xi = ti.cast(self._pos[b][0], ti.i32) - self.size[self._species[0,b]]# *self._vel[b].norm()
            xj = ti.cast(self._pos[b][0], ti.i32) + self.size[self._species[0,b]]# *self._vel[b].norm()
            yi = ti.cast(self._pos[b][1], ti.i32) - self.size[self._species[0,b]]# *self._vel[b].norm()
            yj = ti.cast(self._pos[b][1], ti.i32) + self.size[self._species[0,b]]# *self._vel[b].norm()
            for x in range(xi, xj):
                for y in range(yi, yj):
                    p = self._palette[self._species[0, b]] * (1-self._vel[b].norm()*0.5)
                    # p[3] = 1-
                    self.px_rgba[x, y] = p
                    # self.px_rgb[x, y] = ti.Vector([
                    #     ti.cast(self._vel[b][1]     * 255, ti.i32),
                    #     ti.cast(self._vel[b].norm() * 255, ti.i32),
                    #     ti.cast(self._vel[b][0]     * 255, ti.i32)])
 
    def get_px(self):
        if self.colormode == 'rgb':
            # return self.px_rgb.to_numpy()
            return self.px_rgba.to_numpy()
        elif self.colormode == 'g':
            return self.px_g.to_numpy()[0]
        return False

    def process(self):#, obs: ti.template()):
        if self.pause == False:
            self.step()
            # self.avoid(obs)
            self.move()
            if self.colormode == 'rgb':
                self.render_rgb()
            elif self.colormode == 'g':
                self.render_g()
        return self.get_px()
    
    @ti.kernel
    def set_params(self, i: ti.i32, p: ti.template()):
        self.separate[i] = p[0]
        self.align[i]    = p[1]
        self.cohere[i]   = p[2]
        self.fear[i]     = p[3]
        self.dt[i]       = p[4]
        self.radius[i]   = p[5]
        self.speed[i]    = p[6]

    def get_params(self, i):
        # TODO: Replace with @ti.dataclass?
        params = ti.field(dtype=ti.f32, shape=(7))
        params[0] = self.separate[i]
        params[1] = self.align[i]
        params[2] = self.cohere[i]
        params[3] = self.fear[i]
        params[4] = self.dt[i]
        params[5] = self.radius[i]
        params[6] = self.speed[i]
        return params

    def reset(self):
        self.init()


# `jurigged -v tulvera/tulvera/vera/_boids.py`
def update(b):
    # b.separate[None] = separate
    # b.align[None]    = align
    # b.cohere[None]   = cohere
    # b.dt[None]       = dt
    # b.radius[None]   = radius
    b.speed[None]    = 3.0

def main():
    t = int(time.time())
    print(t)
    ti.init(arch=ti.vulkan, random_seed=t)
    x = 1920
    y = 1080
    n = 4096
    # obstacles = Obstacles(x, y)
    boids = BoidsMulti(x, y, n, colormode='rgb', species=4)
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()
    while window.running:
        # update(boids) # jurigged: costs 10fps
        canvas.set_image(boids.process())
        window.show()

if __name__ == '__main__':
    main()
