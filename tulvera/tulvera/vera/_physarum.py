import numpy as np
import taichi as ti

@ti.data_oriented
class Physarum:
    def __init__(self, 
                 size=512, 
                 n=1024,
                 sense_angle=0.20 * np.pi,
                 sense_dist=4.0,
                 evaporation=0.95,
                 move_angle=0.1 * np.pi,
                 move_step=2.0,
                 step_per_frame=10):
        self.particle_n = n
        self.grid_size = size
        self.sense_angle    = ti.field(ti.f32, ())
        self.sense_dist     = ti.field(ti.f32, ())
        self.evaporation    = ti.field(ti.f32, ())
        self.move_angle     = ti.field(ti.f32, ())
        self.move_step      = ti.field(ti.f32, ())
        self.step_per_frame = ti.field(ti.i32, ())
        self.sense_angle[None]    = sense_angle
        self.sense_dist[None]     = sense_dist
        self.evaporation[None]    = evaporation
        self.move_angle[None]     = move_angle
        self.move_step[None]      = move_step
        self.step_per_frame[None] = step_per_frame
        self.i = 0
        # TODO: is it preferable to allocate these in `init`?
        self.world = ti.field(dtype=ti.f32, shape=[2, self.grid_size, self.grid_size])
        self.position = ti.Vector.field(2, dtype=ti.f32, shape=[self.particle_n])
        self.angle = ti.field(dtype=ti.f32, shape=[self.particle_n])
        self.init()

    @ti.kernel
    def init(self):
        for p in ti.grouped(self.world):
            self.world[p] = 0.0
        for i in self.position:
            self.position[i] = ti.Vector([ti.random(), ti.random()]) * self.grid_size
            self.angle[i] = ti.random() * np.pi * 2.0

    @ti.func
    def sense(self, phase, pos, ang):
        p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * self.sense_dist[None]
        return self.world[phase, p.cast(int) % self.grid_size]

    @ti.kernel
    def step(self, phase: ti.i32):
        # move
        for i in self.position:
            pos, ang = self.position[i], self.angle[i]
            l = self.sense(phase, pos, ang - self.sense_angle[None])
            c = self.sense(phase, pos, ang)
            r = self.sense(phase, pos, ang + self.sense_angle[None])
            if l < c < r:
                ang += self.move_angle[None]
            elif l > c > r:
                ang -= self.move_angle[None]
            elif c < l and c < r:
                ang += self.move_angle[None] * (2 * (ti.random() < 0.5) - 1)
            pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * self.move_step[None]
            self.position[i], self.angle[i] = pos, ang

        # deposit
        for i in self.position:
            ipos = self.position[i].cast(int) % self.grid_size
            self.world[phase, ipos] += 1.0

        # diffuse
        for i, j in ti.ndrange(self.grid_size, self.grid_size):
            a = 0.0
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    a += self.world[phase, (i + di) % self.grid_size, (j + dj) % self.grid_size]
            a *= self.evaporation[None] / 9.0
            self.world[1 - phase, i, j] = a

    def update(self):
        for _ in range(int(self.step_per_frame[None])):
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
        update(physarum) # jurigged
        canvas.set_image(physarum.world.to_numpy()[0])
        window.show()

# `jurigged -v tulvera/tulvera/vera/_physarum.py`
def update(b):
    pass
    # p.sense_angle[None] = 0.2 * np.pi

if __name__ == '__main__':
    main()
