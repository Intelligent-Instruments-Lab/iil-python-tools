# https://github.com/zojize/taichi-flocking

"""
taichi flocking simulation
credits: https://www.youtube.com/watch?v=mhjuuHl6qHM
"""
import taichi as ti
import numpy as np
from numpy.random import default_rng


@ti.func
def set_mag(v: ti.template(), mag: ti.f32):
    return (v / v.norm()) * mag


@ti.func
def limit(v, mag):
    norm = v.norm()
    return (v / norm) * mag if norm > 0 and norm > mag else v


def deg_to_rad(deg):
    return deg / 180 * np.pi


def rotation_mat2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def randomvector(n):
    components = [np.random.normal() for _ in range(n)]
    r = np.sqrt(sum(x * x for x in components))
    v = np.array([x / r for x in components])
    return v


# https://stackoverflow.com/questions/8904694/how-to-normalize-a-2-dimensional-np-array-in-python-less-verbose
def normalized(m):
    return m / np.linalg.norm(m, axis=-1)[:, np.newaxis]


@ti.data_oriented
class Flock:
    def __init__(
        self,
        n,
        pos=None,
        vel=None,
        acc=None,
        pa=None,
        pr=None,
        max_acc=None,
        max_spd=None,
    ):
        self.n = n

        self.pos = ti.Vector.field(n=2, dtype=ti.f32, shape=self.n)
        self.vel = ti.Vector.field(n=2, dtype=ti.f32, shape=self.n)
        self.acc = ti.Vector.field(n=2, dtype=ti.f32, shape=self.n)
        self.pr = ti.field(dtype=ti.f32, shape=self.n)
        self.pa = ti.field(dtype=ti.f32, shape=self.n)
        self.max_acc = max_acc
        self.max_spd = max_spd

        self.init_field(self.pos, pos)
        self.init_field(self.vel, vel)
        self.init_field(self.acc, acc)
        self.init_field(self.pa, pa)
        self.init_field(self.pr, pr)

    def init_field(self, field, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                field.from_numpy(value)
            else:
                field.from_numpy(
                    np.full(fill_value=value, dtype=np.float32, shape=self.n))

    @ti.kernel
    def compute_force(self, a: ti.f32, b: ti.f32, c: ti.f32):
        self.clear_acc()
        for i in range(self.n):
            # steering forces
            alignment = ti.Vector([0.0, 0.0])
            separation = ti.Vector([0.0, 0.0])
            cohesion = ti.Vector([0.0, 0.0])

            # number of boids in perception radius
            n = 0
            for j in range(self.n):
                if i != j and (d := (self.pos[i] -
                                     self.pos[j]).norm()) < self.pr[i]:
                    alignment += self.vel[j]
                    separation += (self.pos[i] - self.pos[j])
                    cohesion += self.pos[j]

                    n += 1
            if n > 0:
                alignment = limit(
                    set_mag((alignment / n), self.max_spd) - self.vel[i],
                    self.max_acc) * a
                separation = limit(
                    set_mag((separation / n), self.max_spd) - self.vel[i],
                    self.max_acc) * b
                cohesion = limit(
                    set_mag(((cohesion / n) - self.pos[i]), self.max_spd) -
                    self.vel[i], self.max_acc) * c

                self.acc[i] += alignment
                self.acc[i] += separation
                self.acc[i] += cohesion

    @ti.kernel
    def update(self, h: ti.f32):
        for i in range(self.n):
            self.vel[i] += h * self.acc[i]
            self.pos[i] += h * self.vel[i]
            self.vel[i] = limit(self.vel[i], self.max_spd)

    @ti.func
    def clear_acc(self):
        for i in range(self.n):
            self.acc[i] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def wrap_edge(self):
        for i in range(self.n):
            if self.pos[i].x < 0:
                self.pos[i].x += 1.
            if self.pos[i].x > 1:
                self.pos[i].x -= 1.
            if self.pos[i].y < 0:
                self.pos[i].y += 1.
            if self.pos[i].y > 1:
                self.pos[i].y -= 1.

    def render(self, gui, size=0.02, filename=None):
        # gui.clear(0x112F41)
        gui.clear(0xffffff)

        centers = self.pos.to_numpy()

        # dirs = normalized(self.vel.to_numpy())
        # a = centers + dirs * size
        # b = centers + (dirs @ rotation_mat2d(deg_to_rad(160))) * size
        # c = centers + (dirs @ rotation_mat2d(deg_to_rad(-160))) * size
        # # gui.triangles(a, b, c, color=0xffd0da)
        # gui.triangles(a, b, c, color=0x333333)

        # gui.circles(centers, color=0xffd0da, radius=1)
        gui.circles(centers, color=0x000000, radius=1)
        if filename is None:
            gui.show()
        else:
            gui.show(filename)


if __name__ == "__main__":
    ti.init(arch=ti.gpu)

    WINDOW_HEIGHT = 540
    AR = 1
    WINDOW_WIDTH = AR * WINDOW_HEIGHT

    N = 5000

    gui = ti.GUI("flocking behavior", res=(WINDOW_WIDTH, WINDOW_HEIGHT))

    alignment = gui.slider("align", 0, 2, 0.01)
    alignment.value = 1
    separation = gui.slider("separate", 0, 2, 0.01)
    separation.value = 1
    cohesion = gui.slider("cohere", 0, 2, 0.01)
    cohesion.value = 1

    rng = default_rng(seed=42)
    flock = Flock(
        N,
        pos=rng.random(size=(N, 2), dtype=np.float32),
        vel=np.array([randomvector(2) for _ in range(N)], dtype=np.float32),
        pr=0.15,
        max_acc=0.5,
        max_spd=1,
    )

    h = 1e-2
    while gui.running:
        flock.compute_force(alignment.value, separation.value, cohesion.value)
        flock.update(h)
        flock.wrap_edge()
        flock.render(gui, AR)