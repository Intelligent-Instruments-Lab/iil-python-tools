import taichi as ti

@ti.data_oriented
class Particles:
    def __init__(self,
                 x=1024,
                 y=1024,
                 n=1024):
        self._x = x
        self._y = y
        self._n = n
        self._pos = ti.Vector.field(2, dtype=ti.f32, shape=(self._n))
        self.pause = False

    # @ti.kernel
    # def _process(self):
    #     # only outer-most loops parallelize
    #     for s in range(self._substeps):
    #         pass

    # @ti.kernel
    # def process(self):
    #     if self._paused is not False:
    #         self._process()
            
    # def cleanup(self):
    #     pass

