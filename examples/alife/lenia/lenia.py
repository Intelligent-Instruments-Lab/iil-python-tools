"""
Based on https://github.com/metachow/Taichi_Lenia (MIT License)
"""

import taichi as ti
import time

from taichi.ui import canvas

ti.init(arch=ti.vulkan)
# ti.init(arch=ti.cuda)


# color map is copy from: https://forum.taichi.graphics/t/vortex-method-demo/775
class ColorMap:
    def __init__(self, h, wl, wr, c):
        self.h = h
        self.wl = wl
        self.wr = wr
        self.c = c

    @ti.func
    def clamp(self, x):
        return max(0.0, min(1.0, x))

    @ti.func
    def map(self, x):
        w = 0.0
        if x < self.c:
            w = self.wl
        else:
            w = self.wr
        return self.clamp((w - abs(self.clamp(x) - self.c)) / w * self.h)


jetR = ColorMap(1.5, .37, .37, .75)
jetG = ColorMap(1.5, .37, .37, .5)
jetB = ColorMap(1.5, .37, .37, .25)


@ti.func
def color_map(c):
    return ti.Vector([jetR.map(c), jetG.map(c), jetB.map(c)])


@ti.data_oriented
class Lenia:
    def __init__(self,
                 res,
                 scatter,
                 conv_r,
                 time,
                 miu,
                 sig,
                 kr=1,
                 kb=ti.Vector([1])):
        #Display
        self.res = res
        self.scatter = scatter
        self.step = 1
        self.paused = True
        self.time = time
        self.dt = 1 / self.time

        self.max_conv_r = 30
        self.conv_r = conv_r

        self.kernel_alpha = 4.0
        self.kernel_rank = kr

        self.grow_miu = ti.field(ti.f32, ())
        self.grow_sig = ti.field(ti.f32, ())
        self.grow_miu[None] = miu
        self.grow_sig[None] = sig

        self.total = ti.field(ti.f32, ())
        self.brush = ti.field(ti.f32, ())
        self.brush[None] = 0.03

        # self.kernel_beta = ti.field(ti.f32, self.kernel_rank)
        self.kernel_beta = ti.Vector([1/2, 1, 1/3])
        self.kernel_beta = kb

        self.world_old = ti.field(ti.f32, (self.res, self.res))
        self.world_slice = ti.field(ti.f32, (self.res))
        self.world_save = ti.field(ti.f32, (self.res, self.res))
        self.world_new = ti.field(ti.f32, (self.res, self.res))

        self.kernel = ti.field(ti.f32,
                               (2 * self.max_conv_r, 2 * self.max_conv_r))
        self.pixels = ti.Vector.field(3, ti.f32,
                                      (res * scatter, res * scatter))

        self.cursor = ti.field(dtype=float, shape=2)

    @ti.func
    def clip(self, x, min, max):
        return ti.min(ti.max(x, min), max)

    @ti.func
    def kernel_core(self, r):
        return ti.exp(self.kernel_alpha - self.kernel_alpha / (4.0 * r *
                                                               (1.0 - r)))

    @ti.func
    def kernel_shell(self, x, y):
        c = 0.0
        center = ti.Vector([float(self.max_conv_r), float(self.max_conv_r)])
        xy = ti.Vector([x, y])
        # r = (xy - center).norm() / self.conv_r
        r = self.clip((xy - center).norm() / self.conv_r, 0, 1)
        br = r * self.kernel_rank
        for i in ti.static(range(self.kernel_rank)):
            if ti.floor(br) == i:
                c = self.clip(
                    (r < 1) * self.kernel_beta[i] * self.kernel_core(br - i),
                    0.0, 1.0)
        return c

    @ti.kernel
    def kernel_build(self):

        self.total[None] = 0.0

        for i, j in ti.ndrange(2 * self.max_conv_r, 2 * self.max_conv_r):
            self.kernel[i, j] += self.kernel_shell(float(i), float(j))
            self.total[None] += self.kernel_shell(float(i), float(j))

        # print("kernel building completed!,kernel total is {}".format(self.total[None]))

    @ti.kernel
    def kernel_norm(self, mat: ti.template()):

        for i, j in ti.ndrange(2 * self.max_conv_r, 2 * self.max_conv_r):
            mat[i, j] /= self.total[None]

        self.total[None] = 0.0

        for i, j in ti.ndrange(2 * self.max_conv_r, 2 * self.max_conv_r):
            self.total[None] += mat[i, j]

        print("kernel normalization:{}".format(self.total[None]))

    @ti.func
    def growth_mapping(self, u):
        return 2.0 * ti.exp(-(u - self.grow_miu[None])**2.0 /
                            (2.0 * (self.grow_sig[None]**2.0))) - 1.0

    @ti.kernel
    def world_conv(self):
        for i, j in ti.ndrange(self.res, self.res):
            tmp = 0.0
            for m, n in ti.ndrange(2 * self.max_conv_r, 2 * self.max_conv_r):
                tmp += self.kernel[m, n] * self.world_old[
                    (i - self.max_conv_r + m) % self.res,
                    (j - self.max_conv_r + n) % self.res]
            self.world_new[i, j] = self.growth_mapping(tmp)

    @ti.kernel
    def world_update(self):
        for i, j in ti.ndrange(self.res, self.res):
            t = 0.0
            t = self.clip(
                self.dt * self.world_new[i, j] + self.world_old[i, j], 0.0,
                1.0)
            self.world_new[i, j] = t

    @ti.kernel
    def world_init(self):
        for i, j in ti.ndrange(self.res, self.res):
            # if i < 127 and j < 127 :
            #     self.world_old[i, j] = self.clip(ti.random(), 0.2, 0.9)
            # else:
            #     self.world_old[i, j] = 0
            self.world_old[i, j] = 0

    @ti.kernel
    def render(self):
        for i, j in ti.ndrange(self.res, self.res):
            for k, l in ti.ndrange(self.scatter, self.scatter):
                self.pixels[i * self.scatter + k, j * self.scatter +
                            l] = color_map(self.world_old[i, j])
                # self.pixels[i*self.scatter+k, j*self.scatter+l] = self.get_color(self.world_old[i, j], 0.0, 1.0)
                # self.pixels[i*self.scatter+k, j*self.scatter+l] = ti.Vector([self.world_old[i, j],self.world_old[i, j],self.world_old[i, j]])

    @ti.kernel
    def draw(self):
        center = ti.Vector([self.cursor[0], self.cursor[1]])
        for i, j in ti.ndrange(self.res, self.res):
            dis = (ti.Vector([i, j]) / self.res - center).norm()
            if dis < self.brush[None]:
                self.world_old[i, j] = self.clip(ti.random(), 0.2, 0.8)

    @ti.kernel
    def erase(self):
        center = ti.Vector([self.cursor[0], self.cursor[1]])
        for i, j in ti.ndrange(self.res, self.res):
            dis = (ti.Vector([i, j]) / self.res - center).norm()
            if dis < self.brush[None]:
                self.world_old[i, j] = 0.0

    @ti.kernel
    def slice_world(self, index: ti.i32):
        for i in ti.ndrange(self.res):
            self.world_slice[i] = self.world_old[index, i]

    def save_world(self):
        self.world_save.copy_from(self.world_old)

    def load_world(self):
        self.world_old.copy_from(self.world_save)

    def init(self):
        self.world_init()

        self.kernel_build()

        self.kernel_norm(self.kernel)

        self.render()
        print(
            "Current parameter: kernel radius:{}, dt:{}, miu:{}, sig:{}, kr:{}, kb:{}"
            .format(self.conv_r, self.dt, self.grow_miu[None],
                    self.grow_sig[None], self.kernel_rank, self.kernel_beta))

    def update(self):
        for i in range(self.step):
            self.world_conv()
            self.world_update()
            self.world_old.copy_from(self.world_new)
        # print(self.world_old)

        # self.paused = not self.paused
        self.render()


if __name__ == "__main__":

    res = 256
    scatter = 4

    window = ti.ui.Window("Taichi-Lenia", (res * scatter, res * scatter))
    canvas = window.get_canvas()
    lenia = Taichi_Lenia(res=res,
                         scatter=scatter,
                         conv_r=20,
                         time=10,
                         miu=0.15,
                         sig=0.016,
                         kr=1,
                         kb=ti.Vector([1]))

    lenia.init()

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key in [ti.ui.ESCAPE]:
                exit()
            elif e.key == 'p':
                lenia.paused = not lenia.paused
                print('Pause state:{}'.format(lenia.paused))
            elif e.key == 'r':
                lenia.init()
                print("Reset world")
            elif e.key == 's':
                lenia.save_world()
                print("Saved current world")
                lenia.render()
            elif e.key == 'l':
                lenia.load_world()
                print("Loaded saved world")
                lenia.render()

        if window.is_pressed(ti.ui.LMB):
            lenia.cursor[1] = window.get_cursor_pos()[1]
            lenia.cursor[0] = window.get_cursor_pos()[0]
            lenia.draw()
            lenia.render()
        elif window.is_pressed(ti.ui.RMB):
            lenia.cursor[1] = window.get_cursor_pos()[1]
            lenia.cursor[0] = window.get_cursor_pos()[0]
            lenia.erase()
            lenia.render()

        canvas.set_image(lenia.pixels)

        window.GUI.begin("Taichi Lenia", 0.01, 0.01, 0.6, 0.15)
        window.GUI.text("LB press: draw, RB press clear")
        window.GUI.text("r : Reset, SPACE : pause")
        window.GUI.text("S : save, L : load")
        # lenia.conv_r = window.GUI.slider(
        #     "Convolution kernel radius",
        #     lenia.conv_r, 5, 40
        # )
        # lenia.time = window.GUI.slider(
        #     "time step",
        #     lenia.time, 1, 20
        # )
        lenia.grow_miu[None] = window.GUI.slider_float("Growth function miu",
                                                       lenia.grow_miu[None],
                                                       0.01, 0.30)
        lenia.grow_sig[None] = window.GUI.slider_float("Growth function sigma",
                                                       lenia.grow_sig[None],
                                                       0.001, 0.0300)
        lenia.brush[None] = window.GUI.slider_float("Brush radius",
                                                    lenia.brush[None], 0.01,
                                                    0.06)
        window.GUI.end()

        if not lenia.paused:
            lenia.update()

        window.show()

    # gui = ti.GUI("Lenia", (res*scatter, res*scatter))
    # lenia = Taichi_Lenia(res=res, scatter=scatter, conv_r=13)
    # lenia.init()

    # while gui.running:
    # # for i in range(1000):
    #     for e in gui.get_events(ti.GUI.PRESS):
    #         if e.key == ti.GUI.ESCAPE:
    #             exit()
    #         elif e.key == ti.GUI.SPACE:
    #             lenia.paused = not lenia.paused
    #             print('state:{}'.format(lenia.paused))
    #         elif e.key == 'r':
    #             print('Reset')
    #             lenia.init()

    #     gui.get_event()
    #     if(gui.is_pressed(ti.GUI.LMB)):
    #         lenia.cursor[1]=gui.get_cursor_pos()[1]
    #         lenia.cursor[0]=gui.get_cursor_pos()[0]
    #         lenia.draw()
    #         lenia.render()

    #     gui.set_image(lenia.pixels)

    #     if not lenia.paused:
    #         lenia.update()
    #     gui.show()
