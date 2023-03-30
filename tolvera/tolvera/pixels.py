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
                 render=True,
                 polygon_mode='crossing'):
        self.lock = _lock
        self.x = x
        self.y = y
        self.fps = fps
        self.px = Pixel.field(shape=(x,y))
        self.mode = mode
        self.evaporate = evaporate
        self.render = render
        self._polygon_mode = polygon_mode
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
    def background(self,r,g,b):
        bg = ti.Vector([r,g,b,1.0])
        self.rect(0, 0, self.x, self.y, bg)
    @ti.func
    def point(self, x: ti.i32, y: ti.i32, rgba: vec4):
        self.px.rgba[x,y] = rgba
    @ti.func
    def points(self, x: ti.template(), y: ti.template(), rgba: vec4):
        for i in ti.static(range(len(x))):
            self.point(x[i], y[i], rgba)
    @ti.func
    def rect(self, x: ti.i32, y: ti.i32, w: ti.i32, h: ti.i32, rgba: vec4):
        # TODO: fill arg
        for i, j in ti.ndrange(w, h):
            self.px.rgba[x+i, y+j] = rgba
    @ti.func
    def line(self, x0: ti.i32, y0: ti.i32, x1: ti.i32, y1: ti.i32, rgba: vec4):
        '''
        Bresenham's line algorithm
        TODO: thickness
        TODO: anti-aliasing
        '''
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
    @ti.func
    def triangle(self, a, b, c, rgba: vec4):
        # TODO: fill arg
        x = ti.Vector([a[0], b[0], c[0]])
        y = ti.Vector([a[1], b[1], c[1]])
        self.polygon(x,y,rgba)
    @ti.func
    def polygon(self, x: ti.template(), y: ti.template(), rgba: vec4):
        # TODO: fill arg
        # after http://www.dgp.toronto.edu/~mac/e-stuff/point_in_polygon.py
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        l = len(x)
        for i, j in ti.ndrange(x_max-x_min, y_max-y_min):
            p = [x_min+i, y_min+j]
            if self._is_inside(p,x,y,l) != 0:
                self.px.rgba[p[0],p[1]] = rgba
    @ti.func
    def _is_inside(self,p,x,y,l):
        is_inside = 0
        if self._polygon_mode == 'crossing':
            is_inside = self._is_inside_crossing(p,x,y,l)
        elif self._polygon_mode == 'winding':
            is_inside = self._is_inside_winding(p,x,y,l)
        return is_inside
    @ti.func
    def _is_inside_crossing(self,p,x,y,l):
        n = 0
        v0, v1 = ti.Vector([0.0,0.0]), ti.Vector([0.0,0.0])
        for i in range(l):
            i1 = i + 1 if i < l - 1 else 0
            v0, v1 = [x[i], y[i]], [x[i1], y[i1]]
            if (v0[1] <= p[1] and v1[1] > p[1]) or \
                (v0[1] > p[1] and v1[1] <= p[1]):
                vt = (p[1] - v0[1]) / (v1[1] - v0[1])
                if p[0] < v0[0] + vt * (v1[0] - v0[0]):
                    n += 1
        return n % 2
    @ti.func
    def _is_inside_winding(self,p,x,y,l):
        n = 0
        v0, v1 = ti.Vector([0.0,0.0]), ti.Vector([0.0,0.0])
        for i in range(l):
            i1 = i + 1 if i < l - 1 else 0
            v0, v1 = [x[i], y[i]], [x[i1], y[i1]]
            if v0[1] <= p[1] and v1[1] > p[1] and \
                (v0 - v1).cross(p - v1) > 0:
                n += 1
            elif v1[1] <= p[1] and (v0-v1).cross(p-v1) < 0:
                n -= 1
        return n
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
