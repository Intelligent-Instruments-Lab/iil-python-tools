import taichi as ti
from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

# FIXME: @ti.dataclass inheritance https://github.com/taichi-dev/taichi/issues/7422

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

@ti.dataclass
class Pixel:
    g: vec1
    rgb: vec3
    rgba: vec4
    @ti.func
    def rgba_to_g(self):# -> vec1:
        # TODO: rgba_to_g
        pass
    @ti.func
    def rgba_to_rgb(self):# -> vec3:
        # TODO: rgba_to_rgb
        pass
    @ti.func
    def g_inv(self):# -> vec3:
        # TODO: g_inv
        pass
    @ti.func
    def rgb_inv(self):# -> vec3:
        # TODO: rgb_inv
        pass
    @ti.func
    def rgba_inv(self):# -> vec3:
        # TODO: rgba_inv
        pass

@ti.data_oriented
class Pixels:
    def __init__(self,
                 x: ti.i32,
                 y: ti.i32,
                 mode='rgba',
                 evaporate=0.99):
        self.x = x
        self.y = y
        self.px = Pixel.field(shape=(x,y))
        self.mode = mode
        self.evaporate = evaporate
    def set(self, px):
        self.px.rgba = px.rgba
    def get(self):
        return self.px.rgba
    @ti.kernel
    def clear(self):
        self.px.rgba.fill(0)
    @ti.kernel
    def diffuse(self):
        for i, j in ti.ndrange(self.x, self.y):
            d = ti.Vector([0.0,0.0,0.0,0.0])
            for di in ti.static(range(-1, 2)):
                for dj in ti.static(range(-1, 2)):
                    dx = (i + di) % self.x
                    dy = (j + dj) % self.y
                    d += self.px.rgba[dx, dy]
            d *= self.evaporate/9.0
            self.px.rgba[i,j] = d
    @ti.kernel
    def decay(self):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i,j] *= self.evaporate
    @ti.kernel
    def update(self):
        pass
    def __call__(self):
        return self.get()
