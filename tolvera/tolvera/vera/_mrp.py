import taichi as ti

# TODO: Abstract out Trigger? Add to Obstacle?

@ti.dataclass
class MRPNote:
    # TODO: Setter functions with clamping?
    number: ti.i32
    brightness: ti.f32
    intensity: ti.f32
    pitch: ti.f32
    harmonics: ti.types.vector(16, float)

@ti.data_oriented
class TiMRP:
    def __init__(self,
                 x=1920,
                 y=1080):
        self.x = x
        self.y = y
        self.keys = 88
        self.range_start = 21
        self.range_end = 108
        self.harmonics = 16
        self.notes = MRPNote.field(shape=(self.keys))
        self.voices_n = 16
        self.voices = ti.field(dtype=ti.i32, shape=(self.voices_n))
        self.trigger_n = 8
        self.trigger_width = 12.0
        self.trigger_pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.trigger_n))
        self.trigger_rect = ti.Vector.field(4, dtype=ti.f32, shape=(self.trigger_n))
        self.trigger_note = ti.Vector.field(1, dtype=ti.f32, shape=(self.trigger_n))
        self.px_g = ti.field(dtype=ti.f32, shape=(1, self.x, self.y))
        self.init()
    
    # @ti.kernel
    def init(self):
        self.init_notes()
        self.init_triggers()

    @ti.kernel
    def init_notes(self):
        for n in range(self.keys):
            self.notes[n].number = self.range_start + n
            self.notes[n].harmonics[0] = 1.0

    @ti.kernel
    def init_triggers(self):
        for i,j in ti.ndrange((0, self.x),(0, self.y)):
            self.px_g[0, i, j] = 0.0
        for t in range(self.trigger_n):
            self.trigger_pos[t] = ti.Vector([ti.random(ti.f32)*self.x, ti.random(ti.f32)*self.y])
            self.trigger_note[t] = ti.Vector([ti.cast(ti.random(ti.f32)*self.keys+self.range_start, ti.i32)])
            x1 = ti.cast(self.trigger_pos[t][0], ti.i32) - self.trigger_width
            x2 = ti.cast(self.trigger_pos[t][1], ti.i32) + self.trigger_width
            y1 = ti.cast(self.trigger_pos[t][0], ti.i32) - self.trigger_width
            y2 = ti.cast(self.trigger_pos[t][1], ti.i32) + self.trigger_width
            self.trigger_rect[t] = ti.Vector([x1,x2,y1,y2])
            for x in range(x1, x2):
                for y in range(y1, y2):
                    self.px_g[0, x, y] = 1.0

    @ti.kernel
    def boids(self, pos: ti.template(), vel: ti.template()):
        for p in pos:
            px, py, px_p = p[0], p[1], self.px_g[0, p[0], p[1]]
            vx, vy, px_v = vel[p][0], vel[p][1], self.px_g[0, vel[p][0], vel[p][1]]
            if px_p == 0.0 and px_v == 1.0: # entering hit area
                note_on = self.which_trigger(vx, vy)
            elif px_p == 1.0 and px_v == 1.0: # staying in hit area
                note = self.which_trigger(vx, vy)
            elif px_p == 1.0 and px_v == 0.0: # leaving hit area
                note_off = self.which_trigger(px, py)

    @ti.func
    def which_trigger(self, x, y) -> ti.i32:
        triggered = -1
        for t in range(self.trigger_n):
            tx1 = self.trigger_rect[t][0]
            tx2 = self.trigger_rect[t][1]
            ty1 = self.trigger_rect[t][2]
            ty2 = self.trigger_rect[t][3]
            if tx1 < x and x < tx2:
                if ty1 < y and y < ty2:
                    triggered = t
        return self.trigger_note[triggered]
    
    def get_image(self):
        return self.px_g.to_numpy()[0]
    
    def process(self):
        return self.get_image()

def main():
    ti.init(arch=ti.vulkan)
    x = 512
    y = 1080
    # n = 4096
    mrp = TiMRP()
    window = ti.ui.Window("MRP", (x, x))
    canvas = window.get_canvas()
    while window.running:
        
        # update(boids) # jurigged: costs 10fps
        canvas.set_image(mrp.process())
        window.show()

if __name__ == '__main__':
    main()
