import numpy as np
import taichi as ti

@ti.data_oriented
class Physarum:
    def __init__(self, size=512, n=1024):
        self.PARTICLE_N = n
        self.GRID_SIZE = size
        self.SENSE_ANGLE = 0.20 * np.pi
        self.SENSE_DIST = 4.0
        self.EVAPORATION = 0.95
        self.MOVE_ANGLE = 0.1 * np.pi
        self.MOVE_STEP = 2.0
        self.step_per_frame = 10
        self.i = 0
        self.world = ti.field(dtype=ti.f32, shape=[2, self.GRID_SIZE, self.GRID_SIZE])
        self.position = ti.Vector.field(2, dtype=ti.f32, shape=[self.PARTICLE_N])
        self.heading = ti.field(dtype=ti.f32, shape=[self.PARTICLE_N])
        self.init()

    @ti.kernel
    def init(self):
        for p in ti.grouped(self.world):
            self.world[p] = 0.0
        for i in self.position:
            self.position[i] = ti.Vector([ti.random(), ti.random()]) * self.GRID_SIZE
            self.heading[i] = ti.random() * np.pi * 2.0

    @ti.func
    def sense(self, phase, pos, ang):
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * self.SENSE_DIST
        return self.world[phase, p.cast(int) % self.GRID_SIZE]

    @ti.kernel
    def step(self, phase: ti.i32):
        # move
        for i in self.position:
            pos, ang = self.position[i], self.heading[i]
            l = self.sense(phase, pos, ang - self.SENSE_ANGLE)
            c = self.sense(phase, pos, ang)
            r = self.sense(phase, pos, ang + self.SENSE_ANGLE)
            if l < c < r:
                ang += self.MOVE_ANGLE
            elif l > c > r:
                ang -= self.MOVE_ANGLE
            elif c < l and c < r:
                ang += self.MOVE_ANGLE * (2 * (ti.random() < 0.5) - 1)
            pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * self.MOVE_STEP
            self.position[i], self.heading[i] = pos, ang

        # deposit
        for i in self.position:
            ipos = self.position[i].cast(int) % self.GRID_SIZE
            self.world[phase, ipos] += 1.0

        # diffuse
        for i, j in ti.ndrange(self.GRID_SIZE, self.GRID_SIZE):
            a = 0.0
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    a += self.world[phase, (i + di) % self.GRID_SIZE, (j + dj) % self.GRID_SIZE]
            a *= self.EVAPORATION / 9.0
            self.world[1 - phase, i, j] = a

    def update(self):
        for _ in range(int(self.step_per_frame)):
            self.step(self.i % 2)
            self.i += 1

def main():
    ti.init(arch=ti.vulkan)
    res = 1024
    n = 2048
    physarum = Physarum(res, n)
    window = ti.ui.Window("Physarum", (res, res))
    canvas = window.get_canvas()
    while window.running:
        physarum.update()
        canvas.set_image(physarum.world.to_numpy()[0])
        window.show()

if __name__ == '__main__':
    main()
