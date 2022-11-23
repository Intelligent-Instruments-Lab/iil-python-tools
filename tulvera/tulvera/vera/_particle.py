import taichi as ti

# @ti.dataclass
# class Particle:
#     x : ti.i32
#     y : ti.i32
#     n : ti.i32
#     pos : ti.types.vector(2, ti.f32)

# Particle.field(shape=(n))

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
        self.pause = True

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

