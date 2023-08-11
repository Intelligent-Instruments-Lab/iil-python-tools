import taichi as ti

# TODO: color palette
# FIXME: @ti.dataclass inheritance https://github.com/taichi-dev/taichi/issues/7422

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

# TODO: Decouple Species from Particle?
# @ti.dataclass
# class Species:
#     size:      ti.f32
#     speed:     ti.f32
#     max_speed: ti.f32
#     mass:      ti.f32
#     rgba:      vec4
#     diffusion: ti.f32 # for pixels

@ti.dataclass
class Particle:
    # g:    vec1
    # rgb:  vec3
    rgba: vec4
    pos:  vec2
    vel:  vec2
    ang:  ti.f32
    # acc:  vec2
    mass:      ti.f32
    size:      ti.f32
    speed:     ti.f32
    max_speed: ti.f32
    active:    ti.f32
    species:   ti.i32
    decay:     ti.f32
    inertia:   ti.f32
    nearby:    ti.i32 # boids/NN
    left:      ti.f32 # physarum/sense
    centre:    ti.f32 # physarum/sense
    right:     ti.f32 # physarum/sense
    @ti.func
    def dist(self, other):
        # FIXME: wrap around walls
        return self.pos - other.pos
    @ti.func
    def dist_norm(self, other):
        return self.dist(self.pos - other.pos).norm()
    @ti.func
    def dist_normalized(self, other):
        return self.dist(self.pos - other.pos).normalized()

@ti.data_oriented
class Particles:
    def __init__(self,
                 x: ti.i32, y: ti.i32,
                 max_n: ti.i32,
                 species: ti.i32 = 1,
                 wall_margin: ti.i32 = 50,
                 turn_factor: ti.f32 = 0.8,
                 substep: ti.i32 = 1):
        self.max_n = max_n
        self.species_n = species
        self.species_c = ti.Vector.field(4, ti.f32, shape=(species))
        self.x = x
        self.y = y
        self.turn_factor = ti.field(ti.f32, shape=())
        self.turn_factor[None] = turn_factor
        self.wall_margin = ti.field(ti.f32, shape=())
        self.wall_margin[None] = wall_margin
        self.field = Particle.field(shape=(self.max_n))
        self.substep = substep
        self.tmp_pos = ti.Vector.field(2, ti.f32, shape=(max_n)) # FIXME: hack
        self.active_indexes = ti.field(ti.i32, shape=(self.max_n))
        self.active_count = ti.field(ti.i32, shape=())
        self.init()
    @ti.kernel
    def init(self):
        self.randomise()
        self.speciate()
    @ti.kernel
    def update_active(self):
        j = 0
        for i in range(self.max_n):
            p = self.field[i]
            if p.active > 0.0:
                self.active_indexes[j] = i
                j += 1
        self.active_count[None] = j
    @ti.func
    def randomise(self):
        for i in range(self.max_n):
            self.field[i] = Particle(
                pos=[self.x*ti.random(ti.f32),self.y*ti.random(ti.f32)],
                vel=[2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)],
                ang=2.0 * ti.math.pi * ti.random(ti.f32),
                active=1)
    @ti.func
    def speciate(self):
        for i in range(self.species_n):
            self.species_c[i] = [ti.random(ti.f32)*ti.random(ti.f32),ti.random(ti.f32),ti.random(ti.f32),1]
        for i in range(self.max_n):
            s = i % self.species_n
            self.field[i].species = s
            self.field[i].rgba    = self.species_c[s]
            self.field[i].size      = 3.0 + 1     * ti.random(ti.f32)
            self.field[i].speed     = 1.0     * ti.random(ti.f32)
            self.field[i].max_speed = 1.0 + 5     * ti.random(ti.f32)
            self.field[i].mass      = 1.0 + 5     * ti.random(ti.f32)
            self.field[i].decay     = 0.9 + 0.099 * ti.random(ti.f32)
    @ti.kernel
    def move(self):
        # TOOD: collisions
        for i in range(self.max_n):
            p = self.field[i]
            if p.active > 0.0:
                self.wall_repel(i)
                self.limit_speed(i)
                self.update_position(i)
        # for i in range(self.active_count[None]):
        #     idx = self.active_indexes[i]
        #     self.wall_repel(idx)
        #     self.limit_speed(idx)
        #     self.update_position(idx)
    @ti.func
    def update_position(self, i):
        p = self.field[i]
        self.field[i].pos += p.vel * p.speed * p.active
    @ti.func
    def wall_repel(self, i):
        p = self.field[i]
        w, t = self.wall_margin[None], self.turn_factor[None]
        x, y = p.pos[0], p.pos[1]
        if  (x > self.x-w): # left
            self.field.vel[i][0] -= t
            self.field.pos[i][0] -= ti.random(ti.f32)*5.0
        elif(x < w):        # right
            self.field.vel[i][0] += t
            self.field.pos[i][0] += ti.random(ti.f32)*5.0
        elif(y > self.y-w): # up
            self.field.vel[i][1] -= t
            self.field.pos[i][1] -= ti.random(ti.f32)*5.0
        elif(y < w):        # down
            self.field.vel[i][1] += t
            self.field.pos[i][1] += ti.random(ti.f32)*5.0
    @ti.func
    def limit_speed(self, i: int):
        p = self.field[i]
        if p.vel.norm() > p.max_speed:
            self.field[i].vel = p.vel.normalized() * p.max_speed
    @ti.kernel
    def activity_decay(self):
        for i in range(self.active_count[None]):
            idx = self.active_indexes[i]
            self.field[idx].active *= self.field[i].decay
    def step(self):
        # self.activity_decay()
        pass
    def process(self):
        for i in range(self.substep):
            self.update_active()
            self.step()
            self.move()
    @ti.kernel
    def render(self, pixels: ti.template()):
        # FIXME: low activity particles rendering...
        # TODO: support g, rgb, rgba
        # TODO: render shapes
        # l = len(px[0,0])
        for i in range(self.max_n):
            p = self.field[i]
            if p.active > 0.0:
                x = ti.cast(p.pos[0], ti.i32)
                y = ti.cast(p.pos[1], ti.i32)
                pixels.circle(x, y, p.size, p.rgba * p.active)        
        # for i in range(self.active_count[None]):
        #     idx = self.active_indexes[i]
        #     p = self.field[idx]
        #     x = ti.cast(p.pos[0], ti.i32)
        #     y = ti.cast(p.pos[1], ti.i32)
        #     pixels.circle(x, y, p.size, p.rgba * p.active)
    def seeks(self, attractors):
        [self.seek(a) for a in attractors]
    def seek(self, attractor): # attractor: Attractor
        self._seek(attractor.p.pos, attractor.p.mass, attractor.radius)
    @ti.kernel
    def _seek(self, pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
        for i in range(self.field.shape[0]):
            if self.field[i].active > 0.0:
                target_distance = (pos-self.field[i].pos).norm()
                if target_distance < radius:
                    factor = (radius-target_distance)/radius
                    self.field[i].vel += (pos-self.field[i].pos).normalized() * mass * factor
    def avoids(self, attractors):
        [self.avoid(a) for a in attractors]
    def avoid(self, attractor):
        self._avoid(attractor.p.pos, attractor.p.mass, attractor.radius)
    @ti.kernel
    def _avoid(self, pos: ti.math.vec2, mass: ti.f32, radius: ti.f32):
        for i in range(self.field.shape[0]):
            if self.field[i].active > 0.0:
                target_distance = (pos-self.field[i].pos).norm()
                if target_distance < radius:
                    factor = (target_distance-radius)/radius
                    self.field[i].vel += (pos-self.field[i].pos).normalized() * mass * factor
    @ti.kernel
    def osc_set_species_speed(self, i: ti.i32, speed: ti.f32, max_speed: ti.f32):
        for j in range(self.max_n):
            if self.field[j].species == i:
                self.field[j].speed = speed
                self.field[j].max_speed = max_speed
    @ti.kernel
    def osc_set_species_color(self, i: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
        for j in range(self.max_n):
            if self.field[j].species == i:
                self.field[j].rgba = [r,g,b,1]
    @ti.kernel
    def osc_set_species_size(self, i: ti.i32, size: ti.f32):
        for j in range(self.max_n):
            if self.field[j].species == i:
                self.field[j].size = size
    @ti.kernel
    def osc_set_active(self, a: ti.i32):
        for i in range(self.field.shape[0]):
            if i > a:
                self.field[i].active = 0
            else:
                self.field[i].active = 1
    @ti.kernel
    def osc_set_species_active(self, i: ti.i32, a: ti.i32):
        for j in range(self.field.shape[0]):
            if self.field[j].species == i:
                if j > a:
                    self.field[j].active = 0
                else:
                    self.field[j].active = 1
    def osc_set_pos(self, i, x, y):
        self.field[i].pos = [x, y]
    def osc_set_vel(self, i, x, y):
        self.field[i].vel = [x, y]
    def osc_set_speed(self, i, s, m):
        self.field[i].speed = s
        self.field[i].max_speed = m
    def osc_set_size(self, i, s):
        self.field[i].size = s
    def osc_set_wall_repel(self, m, t):
        self.wall_margin[None] = m
        self.turn_factor[None] = t
    def osc_get_pos(self, i):
        return self.field[i].pos.to_numpy().tolist()
    def osc_get_vel(self, i):
        return self.field[i].vel.to_numpy().tolist()
    def osc_get_pos_all(self):
        self._osc_get_pos_all()
        return self.tmp_pos.to_numpy().flatten().tolist()
    @ti.kernel
    def _osc_get_pos_all(self):
        # for i in range(self.active_count[None]):
        #     idx = self.active_indexes[i]
        #     p = self.field[idx]
        #     self.tmp_pos[i] = p.pos / [self.x, self.y]
        for i in range(self.max_n):
            p = self.field[i]
            if p.active > 0.0:
                self.tmp_pos[i] = p.pos / [self.x, self.y]
    def reset(self):
        self.init()
    def __call__(self, pixels):
        self.process()
        self.render(pixels)
