from tulvera.vera import *

# TODO: Use Profiler to understand why this cuts framerate by half:

# import taichi as ti
# import numpy as np
# from tulvera.vera import *

# @ti.data_oriented
# class World:
#     def __init__(self,
#                  x=1024,
#                  y=1024,
#                  n=1024):
#         self.boids = Boids(x, y, n)
#         self.physarum = Physarum(x, n)
#         # self.window = ti.ui.Window("Tulvera", (x, y))
#         # self.canvas = self.window.get_canvas()
#         # self.init()

#     @ti.kernel
#     def init(self):
#         pass

#     def process(self):
#         self.boids.update()
#         self.physarum.update()
#         # self.canvas.set_image(self.boids.world.to_numpy().astype(np.uint8))
#         # self.window.show()
    
#     def cleanup():
#         pass
