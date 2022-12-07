import numpy as np
import taichi as ti
import numpy as np

from tolvera.vera._particle import Particles

# TODO: px_rgb mode
# TODO: multispecies

@ti.data_oriented
class Physarum(Particles):
    def __init__(self,
                 x=1024,
                 y=1024,
                 n=1024,
                 sense_angle=0.20 * np.pi,
                 sense_dist=4.0,
                 evaporation=0.97,
                 move_angle=0.2 * np.pi,
                 move_step=4.0,
                 substep=2):
        super().__init__(x, y, n)
        self._ang = ti.field(dtype=ti.f32, shape=(self._n))
        self._i = 0
        self.px_g = ti.field(dtype=ti.f32, shape=(1, self._x, self._y))
        self.sense_angle = ti.field(ti.f32, ())
        self.sense_dist  = ti.field(ti.f32, ())
        self.evaporation = ti.field(ti.f32, ())
        self.move_angle  = ti.field(ti.f32, ())
        self.move_step   = ti.field(ti.f32, ())
        self.substep     = ti.field(ti.i32, ())
        self.sense_angle[None] = sense_angle
        self.sense_dist[None]  = sense_dist
        self.evaporation[None] = evaporation
        self.move_angle[None]  = move_angle
        self.move_step[None]   = move_step
        self.substep[None]     = substep
        self.init()

    @ti.kernel
    def init(self):
        self.randomise()

    @ti.func
    def randomise(self):
        for i in self._pos:
            self._pos[i] = ti.Vector([ti.random()*self._x, ti.random()*self._y])
            self._ang[i] = ti.random() * np.pi * 2.0

    @ti.func
    def sense(self, pos, ang):
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * self.sense_dist[None]
        px = ti.cast(p[0], ti.i32) % self._x
        py = ti.cast(p[1], ti.i32) % self._y
        return self.px_g[0, px, py]

    @ti.kernel
    def move(self):
        for i in self._pos:
            pos, ang = self._pos[i], self._ang[i]
            l = self.sense(pos, ang - self.sense_angle[None])
            c = self.sense(pos, ang)
            r = self.sense(pos, ang + self.sense_angle[None])
            if l < c < r:
                ang += self.move_angle[None]
            elif l > c > r:
                ang -= self.move_angle[None]
            elif c < l and c < r:
                ang += self.move_angle[None] * (2 * (ti.random() < 0.5) - 1)
            pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * self.move_step[None]
            self._pos[i], self._ang[i] = pos, ang

    # @ti.kernel
    # def deposit(self):
    #     for i in self._pos:
    #         iposx = ti.cast(self._pos[i][0], ti.i32) % self._x
    #         iposy = ti.cast(self._pos[i][1], ti.i32) % self._y
    #         self.px_g[0, iposx, iposy] += 1.0

    @ti.kernel
    def deposit(self, _pos: ti.template(), weight: float):
        # TODO: deposit clashes with substep
        for i in _pos:
            iposx = ti.cast(_pos[i][0], ti.i32) % self._x
            iposy = ti.cast(_pos[i][1], ti.i32) % self._y
            self.px_g[0, iposx, iposy] += 1.0*weight
    
    @ti.kernel
    def deposit_swarm(self, _pos: ti.template(), _vel: ti.template(), weight: float):
        # TODO: Why does this seem to alter boids behaviour?
        for i in _pos:
            iposx = ti.cast(_pos[i][0], ti.i32) % self._x
            iposy = ti.cast(_pos[i][1], ti.i32) % self._y
            # self.px_g[0, iposx, iposy] += 1.0*weight
            px_g = self.px_g[0, iposx, iposy]
            self.px_g[0, iposx, iposy] = px_g * _vel[i][0] * weight

    @ti.kernel
    def stamp(self, img: ti.template(), threshold: float, weight: float):
        # TODO: 
        """
        Take an 'image' input and 'stamp' (deposit) it
        """
        x = img.shape[0]
        y = img.shape[1]
        for i, j in ti.ndrange(x, y):
            if img[i, j] < threshold*255:
                _x = ti.cast(i/x*self._x, ti.i32)
                _y = ti.cast(j/y*self._y, ti.i32)
                img_px = img[i, j]/255
                px_g = self.px_g[0, _x, _y]
                self.px_g[0, _x, _y] += px_g * img_px * weight
                # self.px_g[0, _x, _y] += px_g * px * weight

    @ti.kernel
    def stamp_pxg(self, pxg: ti.template(), threshold: float, weight: float):
        # TODO: 
        """
        Take an 'image' input and 'stamp' (deposit) it
        """
        x = pxg.shape[0]
        y = pxg.shape[1]
        for i, j in ti.ndrange(x, y):
            # if pxg[0, i, j] < threshold*255:
            _x = ti.cast(i/x*self._x, ti.i32)
            _y = ti.cast(j/y*self._y, ti.i32)
            px = pxg[0, i, j]/255
            px_g = self.px_g[0, _x, _y]
            # self.px_g[0, _x, _y] += px/255*weight
            self.px_g[0, _x, _y] += px * weight

    @ti.kernel
    def diffuse(self):
        for i, j in ti.ndrange(self._x, self._y):
            a = 0.0
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    dx = (i + di) % self._x
                    dy = (j + dj) % self._y
                    a += self.px_g[0, dx, dy]
            a *= self.evaporation[None] / 9.0
            self.px_g[0, i, j] = a

    def step(self):
        self.move()
        self.deposit(self._pos, 1.0)
        self.diffuse()

    def get_image(self):
        return self.px_g.to_numpy()[0]

    def get_image_isaac(self):
        return self.px_g.to_numpy()[0]

    def process(self):
        if self.pause == False:
            for _ in range(int(self.substep[None])):
                self.step()
                self._i += 1
        return self.get_image()

    @ti.kernel
    def update(self, p: ti.template()):
        self.sense_angle[None] = p[0] * np.pi
        self.sense_dist[None]  = p[1]
        self.evaporation[None] = p[2]
        self.move_angle[None]  = p[3] * np.pi
        self.move_step[None]   = p[4]
        self.substep[None]     = ti.cast(p[5], ti.i32)

# `jurigged -v tulvera/tulvera/vera/_physarum.py`
def update(p):
    pass
    # p.sense_angle[None] = 0.2 * np.pi

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 16785408 # 8392704 # 4196352 # 2098176 # 1049088 # 524544 # 262272 # 131136 # 65568 # 32784
    physarum = Physarum(x, y, n)
    physarum.pause = False
    window = ti.ui.Window("Physarum", (x, y))
    canvas = window.get_canvas()
    while window.running:
        # update(physarum) # jurigged
        canvas.set_image(physarum.process())
        window.show()

if __name__ == '__main__':
    main()
