import taichi as ti
from math import log

ti.init(arch=ti.gpu)

@ti.data_oriented
class Gravity:
    def __init__(self, n=40):
        self.dt = 1e-5
        self.N = n
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=self.N)  # particle velocities
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=self.N, needs_grad=True)  # particle positions
        self.U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy
        self.steps_per_frame = 50
        self.init()

    @ti.kernel
    def init(self):
        for i in self.x:
            self.x[i] = [ti.random(), ti.random()]

    @ti.kernel
    def compute_U(self):
        for i, j in ti.ndrange(self.N, self.N):
            r = self.x[i] - self.x[j]
            # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
            # This is to prevent 1/0 error which can cause wrong derivative
            self.U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|

    @ti.kernel
    def advance(self):
        for i in self.x:
            self.v[i] += self.dt * -self.x.grad[i]  # dv/dt = -dU/dx
        for i in self.x:
            self.x[i] += self.dt * self.v[i]  # dx/dt = v

    def substep(self):
        with ti.ad.Tape(loss=self.U):
            # Kernel invocations in this scope will later contribute to partial derivatives of
            # U with respect to input variables such as x.
            # The tape will automatically compute dU/dx and save the results in x.grad
            self.compute_U() 
        self.advance()
    
    def update(self):
        for i in range(self.steps_per_frame):
            self.substep()

def main():
    ti.init(arch=ti.vulkan)
    res = 1024
    n = 16
    gravity = Gravity(n)
    window = ti.ui.Window("Gravity", (res, res))
    canvas = window.get_canvas()
    while window.running:
        gravity.update()
        canvas.circles(centers=gravity.x, radius=0.005, color=(1,1,1))
        window.show()

if __name__ == "__main__":
    main()
