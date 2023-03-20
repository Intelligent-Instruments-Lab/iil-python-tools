import taichi as ti
import numpy as np
import math

from tolvera.vera._particle import Particles
from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

@ti.data_oriented
class Obstacles(Particles):
    def __init__(self,
                 x=1024,
                 y=1024,
                 n=8,
                 r=32):
        super().__init__(x, y, n)
        self._vel = ti.Vector.field(2, dtype=ti.f32, shape=(self._n))
        self.px_rgb = ti.Vector.field(3, dtype=ti.f32, shape=(self._x, self._y))
        # self.dt
        # self.speed = ti.field(dtype=ti.f32, shape=(self._n))
        self.radius = ti.field(dtype=ti.f32, shape=(self._n))
        self.attract = ti.field(dtype=ti.f32, shape=(self._n))
        self.repel   = ti.field(dtype=ti.f32, shape=(self._n))
        self.inertia = ti.field(dtype=ti.f32, shape=(self._n))
        self.init()

    def init(self):
        self.randomise()
    
    @ti.kernel
    def randomise(self):
        for b in range(self._n):
            self._pos[b] = ti.Vector([ti.random(ti.f32)*self._x, ti.random(ti.f32)*self._y])
            self._vel[b] = ti.Vector([ti.random(ti.f32)-0.5, ti.random(ti.f32)-0.5])
            self.radius[b]  = ti.random(ti.f32) * 200.0
            self.attract[b] = ti.random(ti.f32)
            self.repel[b]   = ti.random(ti.f32)
            self.inertia[b] = ti.random(ti.f32)

    # TODO: inertia
    @ti.kernel
    def move(self):
        for b in range(self._n):
            self._pos[b] = self._pos[b] + 0.5 * self._vel[b] # * self.inertia[b]
            x = self._pos[b][0]
            y = self._pos[b][1]
            if   (x > self._x): self._pos[b][0] = 1
            elif (y > self._y): self._pos[b][1] = 1
            elif (x < 0):       self._pos[b][0] = self._x-1
            elif (y < 0):       self._pos[b][1] = self._y-1

    @ti.kernel
    def render_clear(self):
        for i,j in ti.ndrange((0, self._x),(0, self._y)):
            self.px_rgb[i, j] = ti.Vector([0.0,0.0,0.0])

    # TODO: Abstract out
    @ti.kernel
    def render_evaporate(self):
        for i,j in ti.ndrange((0, self._x),(0, self._y)):
            self.px_rgb[i, j] *= 0.8 

    @ti.kernel
    def render(self):
        for b in range(self._n):
            xi = ti.cast(self._pos[b][0], ti.i32) - 10.0
            xj = ti.cast(self._pos[b][0], ti.i32) + 10.0
            yi = ti.cast(self._pos[b][1], ti.i32) - 10.0
            yj = ti.cast(self._pos[b][1], ti.i32) + 10.0
            for x in range(xi, xj):
                for y in range(yi, yj):
                    self.px_rgb[x, y] = ti.Vector([1.0, 1.0, 1.0])

    # TODO: Improve impact dynamics...
    # att - repel
    @ti.kernel
    def impact(self, n: int,
               pos: ti.template(),
               vel: ti.template()):
        for o in range(self._n):
            self._impact(o, n, pos, vel)

    @ti.func
    def _impact(self, o: int, n: int, 
                pos: ti.template(),
                vel: ti.template()):
        for i in range(n):
            dis = (pos[i] - self._pos[o]).norm()
            rad = self.radius[o]
            if dis <= rad:
                att = ((self._pos[o] - pos[i]) / dis) * self.attract[o]
                rep = ((pos[i] - self._pos[o]) / dis) * self.repel[o]
                vel[i] += (att + rep).normalized()
    
    def set(self, s):
        self._pos[s[0]] = [s[1], s[2]]
        self._vel[s[0]] = [s[3], s[4]]

    # def set(self, i, pos, vel):
    #     self._pos[i] = pos
    #     self._vel[i] = vel

    def get_px(self):
        return self.px_rgb

    def process(self):
        self.move()
        # self.render_clear()
        self.render_evaporate()
        self.render()
        return self.get_px()

def main(host="127.0.0.1", port=4000):
    osc = OSC(host, port, verbose=False, concurrent=True)
    # osc.create_client("Obstacles", host="127.0.0.1", port=7564)

    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    n = 8
    obstacles = Obstacles(x, y, n)
    window = ti.ui.Window("Obstacles", (x, y))
    canvas = window.get_canvas()

    set = None
    @osc.args("/tolvera/obstacles/set")
    def _(address, *args):
        nonlocal set
        set = {'update': True,'i':args[0],'px':args[1],'py':args[2],'vx':args[3],'vy':args[4]}
    def set_update():
        obstacles.set(set['i'], [set['px'], set['py']], [set['vx'], set['vy']])
        set['update'] = False

    pos = None
    @osc.args("/tolvera/obstacles/pos")
    def _(address, *args):
        nonlocal pos
        pos = {'update': True, 'i': args[0], 'x': args[1], 'y': args[2]}
    def pos_update():
        obstacles._pos[pos['i']] = [pos['x'], pos['y']]
        pos['update'] = False

    vel = None
    @osc.args("/tolvera/obstacles/vel")
    def _(address, *args):
        nonlocal vel
        vel = {'update': True, 'i': args[0], 'x': args[1], 'y': args[2]}
    def vel_update():
        obstacles._vel[vel['i']] = [vel['x'], vel['y']]
        vel['update'] = False

    count   = 5
    counter = 0

    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                if set is not None and set['update'] is True:
                    set_update()
                if pos is not None and pos['update'] is True:
                    pos_update()
                if vel is not None and vel['update'] is True:
                    vel_update()
            counter +=1
            canvas.set_image(obstacles.process())
            window.show()

if __name__ == '__main__':
    main()
