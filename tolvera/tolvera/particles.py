'''
TODO: color palette
TODO: global speed scalar
TODO: walls
    default behaviour in particles
    default wrap vs avoid flags per wall
    then per algorithm override (e.g. boids)
FIXME: @ti.dataclass inheritance https://github.com/taichi-dev/taichi/issues/7422
FIXME: Fix tmp_pos / tmp_vel
'''

import taichi as ti
import numpy as np

vec1 = ti.types.vector(1, ti.f32)
vec2 = ti.math.vec2
vec3 = ti.math.vec3
vec4 = ti.math.vec4

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
                 substep: ti.i32 = 1):
        self.max_n = max_n
        self.species_n = species
        self.species_c = ti.Vector.field(4, ti.f32, shape=(species))
        self.particles_per_species = max_n // species
        self.species_consts = {
            'size_min': 2,
            'size_scale': 2,
            'speed_min': 0.1,
            'speed_scale': 1.0,
            'max_speed_min': 1.0,
            'max_speed_scale': 3.0,
            'mass_min': 1.0,
            'mass_scale': 5.0,
            'decay_min': 0.9,
            'decay_scale': 0.099,
        }
        self.x = x
        self.y = y
        # Wall behaviours: top, right, bottom, left (clockwise, a la CSS margin)
        self.wall_margins = ti.field(ti.f32, shape=(4))
        self.turn_factors = ti.field(ti.f32, shape=(4))
        self.wall_margins.fill(50.0)
        self.turn_factors.fill(0.8)
        self.field = Particle.field(shape=(self.max_n))
        self.substep = substep
        self.tmp_pos = ti.Vector.field(2, ti.f32, shape=(max_n)) # FIXME: hack
        self.tmp_pos_species = ti.Vector.field(2, ti.f32, shape=(species)) # FIXME: hack
        self.tmp_vel = ti.Vector.field(2, ti.f32, shape=(max_n)) # FIXME: hack
        self.tmp_vel_species = ti.Vector.field(2, ti.f32, shape=(species)) # FIXME: hack
        self.tmp_vel_stats = ti.Vector.field(1, ti.f32, shape=(7)) # FIXME: hack
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
            self.species_c[i] = [ti.random(ti.f32),ti.random(ti.f32),ti.random(ti.f32),1]
        c = self.species_consts
        size      = c['size_min']      + c['size_scale']      * ti.random(ti.f32)
        speed     = c['speed_min']     + c['speed_scale']     * ti.random(ti.f32)
        max_speed = c['max_speed_min'] + c['max_speed_scale'] * ti.random(ti.f32)
        mass      = c['mass_min']      + c['mass_scale']      * ti.random(ti.f32)
        decay     = c['decay_min']     + c['decay_scale']     * ti.random(ti.f32)
        for i in range(self.max_n):
            s = i % self.species_n
            self.field[i].species   = s
            self.field[i].rgba      = self.species_c[s]
            self.field[i].size      = size
            self.field[i].speed     = speed
            self.field[i].max_speed = max_speed
            self.field[i].mass      = mass
            self.field[i].decay     = decay
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
    @ti.kernel
    def seeks(self, attractors: ti.template()):
        for i in range(attractors.n):
            a = attractors.field[i]
            self._seek(a.p.pos, a.p.mass, a.radius)
    @ti.kernel
    def seek(self, attractor: ti.template()): # attractor: Attractor
        self._seek(attractor.p.pos, attractor.p.mass, attractor.radius)
    @ti.func
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
    def set_species(self, i: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32, size: ti.f32, speed: ti.f32, max_speed: ti.f32):
        c = self.species_consts
        for j in range(self.max_n):
            if self.field[j].species == i:
                self.field[j].rgba = [r,g,b,1]
                self.field[j].size = c['size_min'] + c['size_scale'] * size
                self.field[j].speed = c['speed_min'] + c['speed_scale'] * speed
                self.field[j].max_speed = c['max_speed_min'] + c['max_speed_scale'] * max_speed
    @ti.kernel
    def set_all_species(self, r: ti.template(), g: ti.template(), b: ti.template(), size: ti.template(), speed: ti.template(), max_speed: ti.template()):
        c = self.species_consts
        for i in range(self.max_n):
            s = i % self.species_n
            self.field[i].rgba = [r[s],g[s],b[s],1]
            self.field[i].size = c['size_min'] + c['size_scale'] * size[s]
            self.field[i].speed = c['speed_min'] + c['speed_scale'] * speed[s]
            self.field[i].max_speed = c['max_speed_min'] + c['max_speed_scale'] * max_speed[s]
    @ti.kernel
    def set_all_species_from_list(self, species: ti.types.ndarray()):
        c = self.species_consts
        for i in range(self.max_n):
            s = i % self.species_n
            self.field[i].rgba = [species[s+0],species[s+1],species[s+2],1]
            self.field[i].size = c['size_min'] + c['size_scale'] * species[s+3]
            self.field[i].speed = c['speed_min'] + c['speed_scale'] * species[s+4]
            self.field[i].max_speed = c['max_speed_min'] + c['max_speed_scale'] * species[s+5]
    @ti.kernel
    def set_species_speed(self, i: ti.i32, speed: ti.f32, max_speed: ti.f32):
        c = self.species_consts
        for j in range(self.max_n):
            if self.field[j].species == i:
                self.field[j].speed = c['speed_min'] + c['speed_scale'] * speed
                self.field[j].max_speed = c['max_speed_min'] + c['max_speed_scale'] * max_speed
    @ti.kernel
    def set_species_color(self, i: ti.i32, r: ti.f32, g: ti.f32, b: ti.f32):
        for j in range(self.max_n):
            if self.field[j].species == i:
                self.field[j].rgba = [r,g,b,1]
    @ti.kernel
    def set_species_size(self, i: ti.i32, size: ti.f32):
        c = self.species_consts
        for j in range(self.max_n):
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
    def set_speed(self, i, s, m):
        self.field[i].speed = s
        self.field[i].max_speed = m
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
        for i in range(self.max_n):
            p = self.field[i]
            if p.active > 0.0:
                self.tmp_pos[i] = p.pos / [self.x, self.y]
            # else:
            #     self.tmp_pos[i] = [0.0,0.0] # ???
    @ti.kernel
    def _get_vel_all(self):
        for i in range(self.max_n):
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
        for j in range(self.max_n):
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
        for j in range(self.max_n):
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
        for j in range(self.max_n):
            if self.field[j].species == i:
                v = self.field[j].vel
                p = self.field[j].pos
                m = self.field[j].mass
                centre_of_mass_velocity += v
                relative_velocity       += v# - centre_of_mass_velocity
                angular_momentum        += m * ti.math.cross(v,p)
                kinetic_energy          += 0.5 * m * v.norm_sqr()
        centre_of_mass_velocity = centre_of_mass_velocity / self.particles_per_species
        relative_velocity = (relative_velocity - centre_of_mass_velocity * self.particles_per_species) / self.particles_per_species
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
    def __call__(self, pixels=None):
        self.process()
        if pixels is not None:
            self.render(pixels)
