# https://github.com/taichi-dev/faster-python-with-taichi/blob/main/reaction_diffusion_taichi.py

import numpy as np
import taichi as ti
import taichi.math as tm

@ti.data_oriented
class ReactionDiffusion():
    def __init__(self,
                 x=1024,
                 y=1024):
        self._x, self._y = 1920, 1080
        self.px_rgb = ti.Vector.field(3, ti.f32, shape=(self._x, self._y))
        self._Du, self._Dv, self._feed, self._kill = 0.160, 0.080, 0.060, 0.062
        #self._Du, self._Dv, self._feed, self._kill = 0.210, 0.105, 0.018, 0.051
        self._uv_grid = np.zeros((2, self._x, self._y, 2), dtype=np.float32)
        self._uv_grid[0, :, :, 0] = 1.0
        rand_rows = np.random.choice(range(self._x), 50)
        rand_cols = np.random.choice(range(self._y), 50)
        self._uv_grid[0, rand_rows, rand_cols, 1] = 1.0
        self._uv = ti.Vector.field(2, ti.f32, shape=(2, self._x, self._y))
        self._uv.from_numpy(self._uv_grid)
        self._palette = ti.Vector.field(4, ti.f32, shape=(5, ))
        self._palette[0] = [0.0, 0.0, 0.0, 0.3137]
        self._palette[1] = [1.0, 0.1843, 0.53333, 0.37647]
        self._palette[2] = [0.8549, 1.0, 0.53333, 0.388]
        self._palette[3] = [0.376, 1.0, 0.478, 0.392]
        self._palette[4] = [1.0, 1.0, 1.0, 1]
        self.substep = 32
        self._i = 0

    @ti.kernel
    def compute(self, phase: int):
        for i, j in ti.ndrange(self._x, self._y):
            cen = self._uv[phase, i, j]
            lapl = self._uv[phase, i + 1, j] + self._uv[phase, i, j + 1] + self._uv[
                phase, i - 1, j] + self._uv[phase, i, j - 1] - 4.0 * cen
            du = self._Du * lapl[0] - cen[0] * cen[1] * cen[1] + self._feed * (1 - cen[0])
            dv = self._Dv * lapl[1] + cen[0] * cen[1] * cen[1] - (self._feed + self._kill) * cen[1]
            val = cen + 0.5 * tm.vec2(du, dv)
            self._uv[1 - phase, i, j] = val

    @ti.kernel
    def render(self):
        for i, j in self.px_rgb:
            value = self._uv[0, i, j].y
            color = tm.vec3(0)
            if value <= self._palette[0].w:
                color = self._palette[0].xyz
            for k in range(4):
                c0 = self._palette[k]
                c1 = self._palette[k + 1]
                if c0.w < value < c1.w:
                    a = (value - c0.w) / (c1.w - c0.w)
                    color = tm.mix(c0.xyz, c1.xyz, a)
            self.px_rgb[i, j] = color
    
    def get_image(self):
        return self.px_rgb

    def process(self):
        for _ in range(self.substep):
            self.compute(self._i % 2)
            self._i += 1
        self.render()

# `jurigged -v tulvera/tulvera/vera/_reaction-diffusion.py`
def update(rd):
    # rd.something[None]    = 3.0
    pass

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    rea_diff = ReactionDiffusion(x, y)
    window = ti.ui.Window("ReactionDiffusion", (x, y))
    canvas = window.get_canvas()
    while window.running:
        # update(rea_diff) # jurigged: costs 10fps
        rea_diff.process()
        canvas.set_image(rea_diff.get_image())
        window.show()

if __name__ == '__main__':
    main()
