import taichi as ti
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
                 evaporate=0.99,
                 fps=120,
                 name='Tolvera',
                 render=True):
        self.lock = _lock
        self.x = x
        self.y = y
        self.fps = fps
        self.px = Pixel.field(shape=(x,y))
        self.mode = mode
        self.evaporate = evaporate
        self.render = render
        if render:
            self.window = ti.ui.Window(name, (x,y), fps_limit=fps)
            self.canvas = self.window.get_canvas()
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
    @ti.func
    def point(self, x: ti.i32, y: ti.i32, rgba: vec4):
        self.px.rgba[x,y] = rgba
    @ti.func
    def points(self, x: ti.template(), y: ti.template(), rgba: vec4):
        for i in ti.static(range(len(x))):
            self.point(x[i], y[i], rgba)
    @ti.func
    def rect(self, x: ti.i32, y: ti.i32, w: ti.i32, h: ti.i32, rgba: vec4):
        for i, j in ti.ndrange(w, h):
            self.px.rgba[x+i, y+j] = rgba
    @ti.func
    def line(self, x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, rgba: vec4):
        dx = ti.abs(x1 - x0)
        dy = ti.abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                self.px.rgba[x, y] = rgba
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                self.px.rgba[x, y] = rgba
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        self.px.rgba[x, y] = rgba
    @ti.func
    def circle(self, x: ti.i32, y: ti.i32, r: ti.i32, rgba: vec4):
        for i in range(r + 1):
            d = ti.sqrt(r ** 2 - i ** 2)
            d_int = ti.cast(d, ti.i32)
            for j in range(d_int):
                self.px.rgba[x + i, y + j] = rgba
                self.px.rgba[x + i, y - j] = rgba
                self.px.rgba[x - i, y - j] = rgba
                self.px.rgba[x - i, y + j] = rgba
    # @ti.func
    # def triangle(self, x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, x2: ti.i32, y2: ti.i32, rgba: vec4):
    # @ti.func
    # def polygon(self, x: ti.template(), y: ti.template(), rgba: vec4):
    @ti.kernel
    def decay(self):
        for i, j in ti.ndrange(self.x, self.y):
            self.px.rgba[i,j] *= self.evaporate
    @ti.kernel
    def update(self):
        pass
    def reset(self):
        self.clear()
    def show(self, f):
        if self.render:
            while self.window.running:
                with self.lock:
                    f()
                    self.canvas.set_image(self.get())
                    self.window.show()
    def __call__(self):
        return self.get()
