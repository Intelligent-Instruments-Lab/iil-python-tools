'''
TODO: move rendering outside of particles/make it optional/overrideable
TODO: color palette - implement using State
TODO: Particle.dist debug toroidal distance wrap function
FIXME: refactor tmp_* fields and setters to use State
'''

import taichi as ti
from .utils import Options
from .pixels import Pixels
from .state import State

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
    # @ti.func
    # def dist_wrap(self, other, x, y):
    #     dx = self.pos[0] - other.pos[0]
    #     dy = self.pos[1] - other.pos[1]
    #     if abs(dx) > x / 2: # x-axis
    #         dx = x - abs(dx)
    #         if self.pos[0] > other.pos[0]: dx = -dx
    #     if abs(dy) > y / 2: # y-axis
    #         dy = y - abs(dy)
    #         if self.pos[1] > other.pos[1]: dy = -dy
    #     return ti.Vector([dx, dy])
    @ti.func
    def dist_wrap(self, other, x, y):
        dx = self.pos[0] - other.pos[0]
        dy = self.pos[1] - other.pos[1]
        # Wrap around for the x-axis
        if abs(dx) > x / 2:
            dx = x - abs(dx)
            if self.pos[0] < other.pos[0]:
                dx = -dx
        # Wrap around for the y-axis
        if abs(dy) > y / 2:
            dy = y - abs(dy)
            if self.pos[1] < other.pos[1]:
                dy = -dy
        return ti.Vector([dx, dy])
    # @ti.func
    # def dist_wrap(self, other, width, height):
    #     # Compute the element-wise absolute difference
    #     self_abs = ti.abs(self.pos)
    #     other_abs = ti.abs(other.pos)
    #     delta = self_abs - other_abs
    #     # Check if wrapping around is shorter for both the x and y components
    #     if delta[0] > width / 2:
    #         delta[0] = width - delta[0]
    #     if delta[1] > height / 2:
    #         delta[1] = height - delta[1]
    #     # Correct the signs if necessary
    #     if self.pos[0] > other.pos[0] and delta[0] > 0:
    #         delta[0] = -delta[0]
    #     if self.pos[1] > other.pos[1] and delta[1] > 0:
    #         delta[1] = -delta[1]
    #     return delta
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
        self.species = State(self.o, {
            'size':   (2., 5.),
            'speed':  (0., 4.),
            'mass':   (0., 1.),
            'decay':  (.9, .999),
            'r':      (0., 1.),
            'g':      (0., 1.),
            'b':      (0., 1.),
            'a':      (1., 1.),
        }, self.o.species, osc=('set'), name='particles_species')
        self.px = pixels
        self.x = self.o.x
        self.y = self.o.y
        self.substep = self.o.substep
        self.field = Particle.field(shape=(self.o.particles))
        # TODO: These should be possible with State
        self.tmp_pos = ti.Vector.field(2, ti.f32, shape=(self.o.particles)) # FIXME: hack
        self.tmp_pos_species = ti.Vector.field(2, ti.f32, shape=(self.o.species)) # FIXME: hack
        self.tmp_vel = ti.Vector.field(2, ti.f32, shape=(self.o.particles)) # FIXME: hack
        self.tmp_vel_species = ti.Vector.field(2, ti.f32, shape=(self.o.species)) # FIXME: hack
        self.tmp_vel_stats = ti.Vector.field(1, ti.f32, shape=(7)) # FIXME: hack
        # TODO: These should be possible with State
        self.active_indexes = ti.field(ti.i32, shape=(self.o.particles))
        self.active_count = ti.field(ti.i32, shape=())
        self.init()
    def init(self):
        self.assign_species()
        self.randomise()
    @ti.kernel
    def update_active(self):
        j = 0
        for i in range(self.o.particles):
            p = self.field[i]
            if p.active > 0.0:
                self.active_indexes[j] = i
                j += 1
        self.active_count[None] = j
    @ti.kernel
    def assign_species(self):
        for i in range(self.o.particles):
            self.field[i].species = i % self.o.species
    def _randomise(self):
        self.randomise()
    @ti.kernel
    def randomise(self):
        for i in range(self.o.particles):
            si = self.field[i].species
            s = self.species.field[si, 0]
            species = si
            active  = 1.0
            pos     = [self.x*ti.random(ti.f32),self.y*ti.random(ti.f32)]
            vel     = [2*(ti.random(ti.f32)-0.5), 2*(ti.random(ti.f32)-0.5)]
            mass    = ti.random(ti.f32) * s.mass
            size    = ti.random(ti.f32) * s.size
            speed   = ti.random(ti.f32) * s.speed
            self.field[i] = Particle(species=species, pos=pos, vel=vel, active=active, mass=mass, size=size, speed=speed)
    @ti.kernel
    def move(self):
        # TODO: collisions
        for i in range(self.o.particles):
            if self.field[i] == 0.0: continue
            self.toroidal_wrap(i)
            self.limit_speed(i)
            self.update_position(i)
    @ti.func
    def toroidal_wrap(self, i):
        p = self.field[i]
        if p.pos[0] > self.x: self.field[i].pos[0] = 0.0
        if p.pos[0] < 0.0:    self.field[i].pos[0] = self.x
        if p.pos[1] > self.y: self.field[i].pos[1] = 0.0
        if p.pos[1] < 0.0:    self.field[i].pos[1] = self.y
    @ti.func
    def limit_speed(self, i: int):
        p = self.field[i]
        s = self.species.field[p.species, 0]
        if p.vel.norm() > s.speed:
            self.field[i].vel = p.vel.normalized() * s.speed
    @ti.func
    def update_position(self, i):
        p = self.field[i]
        self.field[i].pos += p.vel * p.speed * p.active
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
        for i in range(self.o.particles):
            p = self.field[i]
            s = self.species.field[p.species, 0]
            if p.active > 0.0:
                x = ti.cast(p.pos[0], ti.i32)
                y = ti.cast(p.pos[1], ti.i32)
                rgba = ti.Vector([s.r, s.g, s.b, s.a])
                self.px.circle(x, y, p.size, rgba * p.active)
        # for i in range(self.active_count[None]):
        #     idx = self.active_indexes[i]
        #     p = self.field[idx]
        #     x = ti.cast(p.pos[0], ti.i32)
        #     y = ti.cast(p.pos[1], ti.i32)
        #     pixels.circle(x, y, p.size, p.rgba * p.active)
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
        for i in range(self.o.particles):
            p = self.field[i]
            if p.active > 0.0:
                self.tmp_pos[i] = p.pos / [self.x, self.y]
            # else:
            #     self.tmp_pos[i] = [0.0,0.0] # ???
    @ti.kernel
    def _get_vel_all(self):
        for i in range(self.o.particles):
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
        for j in range(self.o.particles):
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
        for j in range(self.o.particles):
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
        for j in range(self.o.particles):
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
