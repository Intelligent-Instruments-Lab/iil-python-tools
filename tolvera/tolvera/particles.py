'''
TODO: move rendering outside of particles/make it optional/overrideable
TODO: color palette
TODO: global speed scalar
TODO: walls
    move default behaviour to Vera base class?
    default wrap vs avoid flags per wall
    then per algorithm override (e.g. boids)
FIXME: @ti.dataclass inheritance https://github.com/taichi-dev/taichi/issues/7422
FIXME: Fix tmp_pos / tmp_vel
'''

import taichi as ti
from .utils import Options
from .pixels import Pixels
from .rules import Rules

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

# Physical properties and attributes of a particle
@ti.dataclass
class Particle:
    species: ti.i32
    active:  ti.f32
    pos:     vec2
    vel:     vec2
    mass:    ti.f32
    size:    ti.f32
    speed:   ti.f32
    @ti.func
    def dist(self, other):
        return self.pos - other.pos
    @ti.func
    def dist_norm(self, other):
        return self.dist(self.pos - other.pos).norm()
    @ti.func
    def dist_normalized(self, other):
        return self.dist(self.pos - other.pos).normalized()
    @ti.func
    def dist_wrap(self, other, x, y):
        # TODO: test
        dx = self.pos[0] - other.pos[0]
        dy = self.pos[1] - other.pos[1]
        dx = min(dx, x - abs(dx)) if abs(dx) > y / 2 else dx
        dy = min(dy, x - abs(dy)) if abs(dy) > y / 2 else dy
        return ti.Vector([dx, dy])
    @ti.func
    def randomise(self, x, y):
        self.randomise_pos(x, y)
        self.randomise_vel()
    @ti.func
    def randomise_pos(self, x, y):
        self.pos = [x*ti.random(ti.f32),y*ti.random(ti.f32)]
    @ti.func
    def randomise_vel(self):
        self.vel = [2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)]

@ti.data_oriented
class Particles:
    def __init__(self,
                 options: Options,
                 pixels: Pixels):
        self.o = options
        # self.s = species
        self.rules = Rules({
            'size':   (3., 8.),
            'speed':  (0., 5.),
            'mass':   (0., 1.),
            'decay':  (.9, .999),
            'r':      (0., 1.),
            'b':      (0., 1.),
            'g':      (0., 1.),
            'a':      (1., 1.),
        }, self.o.species)
        self.px = pixels
        self.x = self.o.x
        self.y = self.o.y
        self.substep = self.o.substep
        self.field = Particle.field(shape=(self.o.n))
        # Wall behaviours: top, right, bottom, left (clockwise, a la CSS margin)
        self.wall_margins = ti.field(ti.f32, shape=(4))
        self.turn_factors = ti.field(ti.f32, shape=(4))
        self.wall_margins.fill(50.0)
        self.turn_factors.fill(0.8)
        self.tmp_pos = ti.Vector.field(2, ti.f32, shape=(self.o.n)) # FIXME: hack
        self.tmp_pos_species = ti.Vector.field(2, ti.f32, shape=(self.o.species)) # FIXME: hack
        self.tmp_vel = ti.Vector.field(2, ti.f32, shape=(self.o.n)) # FIXME: hack
        self.tmp_vel_species = ti.Vector.field(2, ti.f32, shape=(self.o.species)) # FIXME: hack
        self.tmp_vel_stats = ti.Vector.field(1, ti.f32, shape=(7)) # FIXME: hack
        self.active_indexes = ti.field(ti.i32, shape=(self.o.n))
        self.active_count = ti.field(ti.i32, shape=())
        self.init()
    def init(self):
        self.assign_species()
        self.randomise()
    @ti.kernel
    def update_active(self):
        j = 0
        for i in range(self.o.n):
            p = self.field[i]
            if p.active > 0.0:
                self.active_indexes[j] = i
                j += 1
        self.active_count[None] = j
    @ti.kernel
    def assign_species(self):
        for i in range(self.o.n):
            self.field[i].species = i % self.o.species
    @ti.kernel
    def randomise(self):
        for i in range(self.o.n):
            species = self.field[i].species
            active  = 1.0
            pos     = [self.x*ti.random(ti.f32),self.y*ti.random(ti.f32)]
            vel     = [2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)]
            # TODO: init based on self.s.consts
            mass    = 1.0
            size    = 2.0
            speed   = 1.0
            self.field[i] = Particle(species=species, pos=pos, vel=vel, active=active, mass=mass, size=size, speed=speed)
    @ti.kernel
    def move(self):
        # TOOD: collisions
        for i in range(self.o.n):
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
        wt, wr, wb, wl = self.wall_margins[0], self.wall_margins[1], self.wall_margins[2], self.wall_margins[3]
        tt, tr, tb, tl = self.turn_factors[0], self.turn_factors[1], self.turn_factors[2], self.turn_factors[3]
        x, y = p.pos[0], p.pos[1]
        if  (y > self.y-wt): # top
            # self.field.vel[i][1] *= -0.9
            self.field.vel[i][1] -= tt
            self.field.pos[i][1] -= ti.random(ti.f32)*5.0
        elif(x < wr):        # right
            # self.field.vel[i][0] *= -0.9
            self.field.vel[i][0] += tr
            self.field.pos[i][0] += ti.random(ti.f32)*5.0
        elif(y < wb):        # bottom
            # self.field.vel[i][1] *= -0.9
            self.field.vel[i][1] += tb
            self.field.pos[i][1] += ti.random(ti.f32)*5.0
        elif(x > self.x-wl): # left
            # self.field.vel[i][0] *= -0.9
            self.field.vel[i][0] -= tl
            self.field.pos[i][0] -= ti.random(ti.f32)*5.0
    @ti.kernel
    def set_wall_margins(self, top: ti.f32, right: ti.f32, bottom: ti.f32, left: ti.f32):
        self.wall_margins[0] = top
        self.wall_margins[1] = right
        self.wall_margins[2] = bottom
        self.wall_margins[3] = left
    @ti.kernel
    def set_turn_factors(self, top: ti.f32, right: ti.f32, bottom: ti.f32, left: ti.f32):
        self.turn_factors[0] = top
        self.turn_factors[1] = right
        self.turn_factors[2] = bottom
        self.turn_factors[3] = left
    @ti.kernel
    def set_wall(self, wall: ti.i32, margin: ti.f32, turn: ti.f32):
        self.wall_margins[wall] = margin
        self.turn_factors[wall] = turn
    @ti.func
    def limit_speed(self, i: int):
        p = self.field[i]
        r = self.rules.field[p.species, p.species] # diagonal index
        if p.vel.norm() > r.speed:
            self.field[i].vel = p.vel.normalized() * r.speed
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
    def render(self):
        # FIXME: low activity particles rendering...
        # TODO: support g, rgb, rgba
        # TODO: render shapes
        # l = len(px[0,0])
        for i in range(self.o.n):
            p = self.field[i]
            r = self.rules.field[p.species, p.species] # diagonal index
            if p.active > 0.0:
                x = ti.cast(p.pos[0], ti.i32)
                y = ti.cast(p.pos[1], ti.i32)
                rgba = ti.Vector([r.r, r.g, r.b, r.a])
                self.px.circle(x, y, p.size, rgba * p.active)
        # for i in range(self.active_count[None]):
        #     idx = self.active_indexes[i]
        #     p = self.field[idx]
        #     x = ti.cast(p.pos[0], ti.i32)
        #     y = ti.cast(p.pos[1], ti.i32)
        #     pixels.circle(x, y, p.size, p.rgba * p.active)
    @ti.kernel
    def set_species_speed(self, i: ti.i32, speed: ti.f32, max_speed: ti.f32):
        c = self.species_consts
        for j in range(self.o.n):
            if self.field[j].species == i:
                self.field[j].speed = c['speed_min'] + c['speed_scale'] * speed
                self.field[j].max_speed = c['max_speed_min'] + c['max_speed_scale'] * max_speed
    @ti.kernel
    def set_species_color(self, i: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
        for j in range(self.o.n):
            if self.field[j].species == i:
                self.field[j].rgba = [r,g,b,1]
    @ti.kernel
    def set_species_size(self, i: ti.i32, size: ti.f32):
        c = self.species_consts
        for j in range(self.o.n):
            if self.field[j].species == i:
                self.field[j].size = c['size_min'] + c['size_scale'] * size
    @ti.kernel
    def set_active(self, a: ti.i32):
        for i in range(self.field.shape[0]):
            if i > a:
                self.field[i].active = 0
            else:
                self.field[i].active = 1
    @ti.kernel
    def set_species_active(self, i: ti.i32, a: ti.i32):
        for j in range(self.field.shape[0]):
            if self.field[j].species == i:
                if j > a:
                    self.field[j].active = 0
                else:
                    self.field[j].active = 1
    def set_pos(self, i, x, y):
        self.field[i].pos = [x, y]
    def set_vel(self, i, x, y):
        self.field[i].vel = [x, y]
    def set_speed(self, i, s):
        self.field[i].speed = s
    def set_size(self, i, s):
        self.field[i].size = s
    def set_wall_repel(self, m, t):
        self.wall_margin[None] = m
        self.turn_factor[None] = t
    def get_pos(self, i):
        return self.field[i].pos.to_numpy().tolist()
    def get_vel(self, i):
        return self.field[i].vel.to_numpy().tolist()
    def get_pos_all_1d(self):
        self._get_pos_all()
        return self.tmp_pos.to_numpy().flatten().tolist()
    def get_pos_all_2d(self):
        self._get_pos_all()
        return self.tmp_pos.to_numpy().tolist()
    def get_vel_all_1d(self):
        self._get_vel_all()
        return self.tmp_vel.to_numpy().flatten().tolist()
    def get_vel_all_2d(self):
        self._get_vel_all()
        return self.tmp_vel.to_numpy().tolist()
    @ti.kernel
    def _get_pos_all(self):
        # for i in range(self.active_count[None]):
        #     idx = self.active_indexes[i]
        #     p = self.field[idx]
        #     self.tmp_pos[i] = p.pos / [self.x, self.y]
        # TODO: Only send active particle positions...? Or inactive=-1?
        for i in range(self.o.n):
            p = self.field[i]
            if p.active > 0.0:
                self.tmp_pos[i] = p.pos / [self.x, self.y]
            # else:
            #     self.tmp_pos[i] = [0.0,0.0] # ???
    @ti.kernel
    def _get_vel_all(self):
        for i in range(self.o.n):
            p = self.field[i]
            if p.active > 0.0:
                self.tmp_vel[i] = p.vel
    def get_pos_species_1d(self, species):
        self._get_pos_species()
        return self.tmp_pos_species.to_numpy().flatten().tolist()
    def get_pos_species_2d(self, species):
        self._get_pos_species(species)
        return self.tmp_pos_species.to_numpy().tolist()
    @ti.kernel
    def _get_pos_species(self, i: ti.i32):
        for j in range(self.o.n):
            p = self.field[i]
            if self.field[j].species == i and p.active > 0.0:
                self.tmp_pos_species[j] = p.pos / [self.x, self.y]
    def get_vel_species_1d(self, species):
        self._get_vel_species(species)
        return self.tmp_vel_species.to_numpy().flatten().tolist()
    def get_vel_species_2d(self, species):
        self._get_vel_species(species)
        return self.tmp_vel_species.to_numpy().tolist()
    @ti.kernel
    def _get_vel_species(self, i: ti.i32):
        for j in range(self.o.n):
            p = self.field[i]
            if self.field[j].species == i and p.active > 0.0:
                self.tmp_vel_species[j] = p.vel
    def get_vel_stats_species_1d(self, species):
        self._species_velocity_statistics(species)
        return self.tmp_vel_stats.to_numpy().flatten().tolist()
    @ti.kernel
    def _species_velocity_statistics(self, i: ti.i32):
        '''
        Centre of Mass Velocity: This is the average velocity of all particles in the species.
        Relative Velocity: This is the average velocity of all particles in the species relative to the centre of mass velocity.
        Angular Momentum: This is the sum of the angular momentum of all particles, which is given by mass * velocity * radius for each particle.
        Kinetic Energy: This is the sum of the kinetic energy of all particles, which is given by 0.5 * mass * velocity^2 for each particle.
        Temperature: In statistical mechanics, the temperature of a system of particles is related to the average kinetic energy of the particles.
        '''
        centre_of_mass_velocity = ti.Vector([0.0,0.0])
        relative_velocity = ti.Vector([0.0,0.0])
        angular_momentum = ti.Vector([0.0])
        kinetic_energy = ti.Vector([0.0])
        for j in range(self.o.n):
            if self.field[j].species == i:
                v = self.field[j].vel
                p = self.field[j].pos
                m = self.field[j].mass
                centre_of_mass_velocity += v
                relative_velocity       += v# - centre_of_mass_velocity
                angular_momentum        += m * ti.math.cross(v,p)
                kinetic_energy          += 0.5 * m * v.norm_sqr()
        centre_of_mass_velocity = centre_of_mass_velocity / self.o.particles_per_species
        relative_velocity = (relative_velocity - centre_of_mass_velocity * self.o.particles_per_species) / self.o.particles_per_species
        temperature = 2.0 * kinetic_energy / (self.particles_per_species * 1.380649e-23)
        self.tmp_vel_stats[0] = centre_of_mass_velocity[0]
        self.tmp_vel_stats[1] = centre_of_mass_velocity[1]
        self.tmp_vel_stats[2] = relative_velocity[0]
        self.tmp_vel_stats[3] = relative_velocity[1]
        self.tmp_vel_stats[4] = angular_momentum[0]
        self.tmp_vel_stats[5] = kinetic_energy[0]
        self.tmp_vel_stats[6] = temperature[0]
    def reset(self):
        self.init()
    def __call__(self):
        self.process()
        self.render()
