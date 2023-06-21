import taichi as ti
import numpy as np
from iipyper import OSC, run, repeat, cleanup
from iipyper.state import _lock

@ti.data_oriented
class TiSensel():
    def __init__(self,
                 px_x,
                 px_y,
                 debug=True):
        self._px_x = px_x
        self._px_y = px_y
        self._debug = debug
        self._dims = {
            'x': 230.0,
            'y': 125.0,
            'f': 4096.0,
            'a': 4096.0
        }
        self._max_contacts = 16
        # TODO: Replace with 0D fields
        self._radius = 100.0
        self._brightness = 5.0
        self.contacts = ti.field(dtype=ti.f32, shape=(self._max_contacts, 10))
        self._px_g = ti.field(dtype=ti.f32, shape=(1, self._px_x, self._px_y))

    @ti.func
    def handle_contacts(self, c: ti.template()):
        """
        Basic values for each registered contact.

        /contact index id state x y force area dist wdist 
                orientation  major_axis  minor_axis
        """
        # print(self.contacts.shape, c.shape)
        for id in range(self._max_contacts):
            for cn in range(10):
                self.contacts[id, cn] = c[id, cn]
        # id = args[1] # [int]   contact id (0..15)
        # self.contacts[id, 0] = args[0] # 'state' [int]   one of invalid(0), start(1), move(2), end(3)  
        # self.contacts[id, 1] = args[1]/self._dims['x'] # 'x' [float] x-coordinate in [mm]
        # self.contacts[id, 2] = args[2]/self._dims['y'] # 'y' [float] y-coordinate in [mm]
        # self.contacts[id, 3] = args[3]/self._dims['f'] # 'f' [float] sum of pressure applied [g]
        # self.contacts[id, 4] = args[4]/self._dims['a'] # 'a' [int]   covered area [sensels]
        # self.contacts[id, 5] = args[5] # 'dist'        [float] distance to average position [mm]
        # self.contacts[id, 6] = args[6] # 'wdist'       [float] distance to weighted average position [mm]
        # self.contacts[id, 7] = args[7] # 'orientation' [float] orientation of bounding elipsis [deg] (0..360)
        # self.contacts[id, 8] = args[8] # 'major_axis'  [float] major axis length of bounding elipsis [mm]
        # self.contacts[id, 9] = args[9] # 'minor_axis'  [float] minor axis length of bounding elipsis [mm]

    @ti.kernel
    def raster(self):
        for i,j in ti.ndrange((0, self._px_x),(0, self._px_y)):
            self._px_g[0, i, j] = 0.0
        for n in range(self._max_contacts):
            _x = self.contacts[n,1] * self._px_x
            _y = self._px_y - self.contacts[n,2] * self._px_y
            _a = self.contacts[n,3] * self._radius
            _f = self.contacts[n,4]
            xi = ti.cast(_x, ti.i32) - _a
            xj = ti.cast(_x, ti.i32) + _a
            yi = ti.cast(_y, ti.i32) - _a
            yj = ti.cast(_y, ti.i32) + _a
            for x in range(xi, xj):
                for y in range(yi, yj):
                    self._px_g[0, x, y] = _f*self._brightness

    def get_contacts(self):
        return self.contacts

    @ti.kernel
    def set_contacts(self, c: ti.template()):
        self.handle_contacts(c)

    def get_image(self):
        return self._px_g.to_numpy()[0]

    def process(self):
        if self._debug is True:
            self.raster()
            return self.get_image()

def main(host="127.0.0.1", port=7006):
    osc = OSC(host, port, concurrent=True)
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    sensel = TiSensel(x, y)
    contacts = sensel.get_contacts()
    window = ti.ui.Window("Sensel", (x, y))
    canvas = window.get_canvas()

    # TODO: Why is this such a mess? Try numpy arrays again?

    @osc.args("/contact")
    def _(address, *args):
        nonlocal contacts
        contacts[args[1], 0] = args[2]
        contacts[args[1], 1] = args[3]
        contacts[args[1], 2] = args[4]
        contacts[args[1], 3] = args[5]
        contacts[args[1], 4] = args[6]
        contacts[args[1], 5] = args[7]
        contacts[args[1], 6] = args[8]
        contacts[args[1], 7] = args[9]
        contacts[args[1], 8] = args[10]
        contacts[args[1], 9] = args[11]
        print(args[1], contacts[args[1], 1], contacts[args[1], 2])

    count   = 5
    counter = 0
    
    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                # sensel.set_contacts(contacts)
            counter +=1
            canvas.set_image(sensel.process())
            window.show()

if __name__ == '__main__':
    run(main)
