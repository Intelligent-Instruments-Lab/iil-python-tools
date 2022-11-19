import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

@ti.data_oriented
class JuliaSet:
    def __init__(self, size: ti.i32):
        self.pixels = ti.Vector.field(3, dtype=float, shape=(size * 2, size))
        self.size = size
        print('JuliaSet init')

    @ti.func
    def complex_sqr(self, z: tm.vec2):  # complex square of a 2D vector
        return tm.vec2(z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1])

    @ti.kernel
    def paint(self, t: ti.f32, zoom: ti.f32):
        for i, j in self.pixels:  # Parallelized over all pixels
            c = tm.vec2(-0.8, tm.cos(t) * 0.2)
            z = tm.vec2(i / self.size - 1, j / self.size - 0.5) * 2 / zoom
            iterations = 0
            while z.norm() < 50 and iterations < 50:
                z = self.complex_sqr(z) + c
                iterations += 1
            self.pixels[i, j] = 1 - tm.vec3(
                iterations * 0.02, z.norm() * 0.02, iterations * 0.015)
