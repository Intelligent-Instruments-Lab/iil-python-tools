import taichi as ti
import numpy as np
from pysensel import PySensel

@ti.data_oriented
class TiSensel(): # MultiTouchPressureSurfaceInputs...?
    def __init__(self,
                 px_x,
                 px_y):
        self._px_x = px_x
        self._px_y = px_y
        self._device = PySensel()
        self._x = self._device.info.width
        self._y = self._device.info.height
        # TODO: Handle dynamic # contacts
        self._max_contacts = 5
        # TODO: Replace with 0D fields
        self._radius = 1000.0
        self._brightness = 2.0
        self._contacts_test = np.empty([5, 4], dtype=np.float32)
        self._contact_ids_test = np.empty([5], dtype=int)
        self.contact_ids = ti.field(dtype=ti.i32, shape=(self._max_contacts))
        self.contacts = ti.field(dtype=ti.f32, shape=(self._max_contacts,4))
        self._px_g = ti.field(dtype=ti.f32, shape=(1, self._px_x, self._px_y))
    
    def contacts_test(self):
        self._contacts_test[0, 0] = 130 # x
        self._contacts_test[0, 1] = 55 # y
        self._contacts_test[0, 2] = 1024 # a - 4096
        self._contacts_test[0, 3] = 1024 # f - 4096
        self._contacts_test[1, 0] = 30 # x
        self._contacts_test[1, 1] = 100 # y
        self._contacts_test[1, 2] = 512 # a - 4096
        self._contacts_test[1, 3] = 2048 # f - 4096
        # self.parse_frame(self._contacts_test, self._contacts_ids_test)

    @ti.kernel
    def raster(self):
        for i,j in ti.ndrange((0, self._px_x),(0, self._px_y)):
            self._px_g[0, i, j] = 0.0
        for n in range(self._max_contacts):
            _x = self.contacts[n,0]/self._x*self._px_x
            _y = self.contacts[n,1]/self._y*self._px_y
            _a = self.contacts[n,2]/4096.0*self._radius
            _f = self.contacts[n,3]/4096.0
            xi = ti.cast(_x, ti.i32) - _a
            xj = ti.cast(_x, ti.i32) + _a
            yi = ti.cast(_y, ti.i32) - _a
            yj = ti.cast(_y, ti.i32) + _a
            for x in range(xi, xj):
                for y in range(yi, yj):
                    self._px_g[0, x, y] = _f*self._brightness
    
    def parse_frame(self, contacts, ids):
        self.contacts.from_numpy(contacts)
        self.contact_ids.from_numpy(ids)

    def get_image(self):
        return self._px_g.to_numpy()[0]

    def process(self):
        self._device.scanFrames()
        self.parse_frame(self._device.np_contacts, self._device.np_contact_ids)
        self.raster()
        return self.get_image()

def main():
    ti.init(arch=ti.vulkan)
    x = 1920
    y = 1080
    ti_sensel = TiSensel(x, y)
    window = ti.ui.Window("Sensel", (x, y))
    canvas = window.get_canvas()
    while window.running:
        canvas.set_image(ti_sensel.process())
        window.show()

if __name__ == '__main__':
    main()
