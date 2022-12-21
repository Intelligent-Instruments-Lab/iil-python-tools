import numpy as np
import taichi as ti
import numpy as np
import time

from tolvera.vera._particle import Particles

# TODO: px_rgb mode

# TODO: multispecies
"""
move
    sense
        need to sense other species and avoid
deposit
    deposit colour based on species
diffuse

"""

@ti.data_oriented
class PhysarumMulti(Particles):
    def __init__(self,
                 x=1024,
                 y=1024,
                 n=1024,
                 sense_angle=0.20 * np.pi,
                 sense_dist=4.0,
                 evaporation=0.6,
                 move_angle=0.2 * np.pi,
                 move_step=4.0,
                 species=3,
                 colormode='g',
                 substep=2):
        super().__init__(x, y, n)
        self._ang = ti.field(dtype=ti.f32, shape=(self._n))
        self._i = 0
        self.px_g = ti.field(dtype=ti.f32, shape=(1, self._x, self._y))
        self.px_rgba = ti.Vector.field(4, dtype=ti.f32, shape=(self._x, self._y))
        self._species_n = species
        self._species = ti.field(dtype=ti.i32, shape=(1, self._n))
        self._palette = ti.Vector.field(4, ti.f32, shape=(self._species_n, ))
        self.colormode = colormode
        self.sense_angle = ti.field(ti.f32, shape=(self._species_n))
        self.sense_dist  = ti.field(ti.f32, shape=(self._species_n))
        self.evaporation = ti.field(ti.f32, shape=(self._species_n))
        self.move_angle  = ti.field(ti.f32, shape=(self._species_n))
        self.move_step   = ti.field(ti.f32, shape=(self._species_n))
        self.substep     = ti.field(ti.i32, ())
        self.sense_angle.fill(sense_angle)
        self.sense_dist.fill(sense_dist)
        self.evaporation.fill(evaporation)
        self.move_angle.fill(move_angle)
        self.move_step.fill(move_step)
        self.substep[None] = substep
        self.init()

    def init(self):
        np.random.seed(int(time.time()))
        self.differentiate_species()
        self.randomise()
        self.px_rgba.fill(1.0)

    @ti.kernel
    def randomise(self):
        for p in self._pos:
            species = p % self._species_n
            self._species[0, p] = species
            self._pos[p] = ti.Vector([ti.random()*self._x, ti.random()*self._y])
            self._ang[p] = ti.random() * np.pi * 2.0

    # @ti.func
    def differentiate_species(self):
        for s in range(self._species_n):
            r = np.random.rand(11)
            ri = np.random.randint(1,8)
            self._palette[s] = ti.Vector([r[0],r[1],r[2], 1.0])
            self.sense_angle[s] = r[3] * 0.5 + 0.10 * np.pi
            self.sense_dist[s]  = r[4] * 16.0 + 1.0
            self.evaporation[s] = r[5] * 0.35 + 0.6
            self.move_angle[s]  = r[6] * 0.5 + 0.1 * np.pi
            self.move_step[s]   = r[7] * 4.0 + 1.0

    @ti.func
    def sense(self, pi, pos, ang):
        """
        """
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * self.sense_dist[self._species[0, pi]]
        px = ti.cast(p[0], ti.i32) % self._x
        py = ti.cast(p[1], ti.i32) % self._y
        return self.px_rgba[px, py]

    @ti.kernel
    def move(self):
        """
        """
        for p in self._pos:
            pos, ang = self._pos[p], self._ang[p]
            l = self.sense(p, pos, ang - self.sense_angle[self._species[0, p]]).norm()
            c = self.sense(p, pos, ang).norm()
            r = self.sense(p, pos, ang + self.sense_angle[self._species[0, p]]).norm()
            if l < c < r:
                ang += self.move_angle[self._species[0, p]]
            elif l > c > r:
                ang -= self.move_angle[self._species[0, p]]
            elif r > c and c < l:
                ang += self.move_angle[self._species[0, p]] * (2 * (ti.random() < 0.5) - 1)
            pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * self.move_step[self._species[0, p]]
            self._pos[p], self._ang[p] = pos, ang

    @ti.kernel
    def deposit(self, _pos: ti.template(), weight: float):
        """
        """
        for p in _pos:
            iposx = ti.cast(_pos[p][0], ti.i32) % self._x
            iposy = ti.cast(_pos[p][1], ti.i32) % self._y
            self.px_rgba[iposx, iposy] += self._palette[self._species[0, p]] * weight
            # self.px_g[0, iposx, iposy] += 1.0*weight

    @ti.kernel
    def diffuse(self):
        for i, j in ti.ndrange(self._x, self._y):
            a = ti.Vector([0.0,0.0,0.0,0.0])
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    dx = (i + di) % self._x
                    dy = (j + dj) % self._y
                    # a += self.px_g[0, dx, dy]
                    a += self.px_rgba[dx, dy]
            a *= self.evaporation[self._species[0, 0]] / 9.0
            self.px_rgba[i, j] = a

    def step(self):
        self.move()
        self.deposit(self._pos, 1.0)
        self.diffuse()

    def get_image(self):
        # return self.px_g.to_numpy()[0]
        return self.px_rgba.to_numpy()

    def process(self):
        if self.pause == False:
            for _ in range(int(self.substep[None])):
                self.step()
                self._i += 1
        return self.get_image()

    @ti.kernel
    def set_params(self, p: ti.template()):
        self.sense_angle[None] = p[0] * np.pi
        self.sense_dist[None]  = p[1]
        self.evaporation[None] = p[2]
        self.move_angle[None]  = p[3] * np.pi
        self.move_step[None]   = p[4]
        self.substep[None]     = ti.cast(p[5], ti.i32)
    
    def get_params(self):
        params = ti.field(dtype=ti.f32, shape=(6))
        params[0] = self.sense_angle[None]
        params[1] = self.sense_dist[None]
        params[2] = self.evaporation[None]
        params[3] = self.move_angle[None]
        params[4] = self.move_step[None]
        params[5] = self.substep[None]
        return params
    
    def reset(self):
        self.init()

# `jurigged -v tulvera/tulvera/vera/_physarum.py`
def update(p):
    pass
    # p.sense_angle[None] = 0.2 * np.pi

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    # n = 8392704 # 4196352 # 2098176 # 1049088 # 524544 # 262272 # 131136 # 65568 # 32784
    n = 65568
    physarum = Physarum(x, y, n, colormode='rgb', species=5)
    physarum.pause = False
    window = ti.ui.Window("Physarum", (x, y))
    canvas = window.get_canvas()
    while window.running:
        # update(physarum) # jurigged
        canvas.set_image(physarum.process())
        window.show()

if __name__ == '__main__':
    main()
