# original: https://forum.taichi-lang.cn/t/homework0-boids/563/6

import taichi as ti
import numpy as np
import math

@ti.data_oriented
class Boids:
    def __init__(self, x=512, y=512, n=1024):
        self.dt = 1
        self.vis_radius = 40.0
        self.max_speed = 3.0
        self.boids_size = n
        self.bg_color = 255
        self.boids_radius = 1
        self.resx = x
        self.resy = y
        self.boids_pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.boids_size))
        self.boids_velocity = ti.Vector.field(2, dtype=ti.f32, shape=(self.boids_size))
        self.world = ti.Vector.field(3, dtype=ti.i32, shape=(self.resx, self.resy))
        self.init()

    @ti.kernel
    def init(self):
        for x in range(0, self.boids_size):
            self.boids_pos[x] = ti.Vector([ti.random(ti.f32)*self.resx, ti.random(ti.f32)*self.resy])
            self.boids_velocity[x] = ti.Vector([self.max_speed*(ti.random(ti.f32)-0.5), self.max_speed*(ti.random(ti.f32)-0.5)])

    @ti.kernel
    def render(self):
        for i,j in ti.ndrange((0, self.resx),(0, self.resy)):
            self.world[i, j] = ti.Vector([255 - self.bg_color,255 - self.bg_color,255 - self.bg_color])
        for i in range(self.boids_size):
            for x in range(ti.cast(self.boids_pos[i][0],ti.i32) - self.boids_radius, ti.cast(self.boids_pos[i][0],ti.i32) + self.boids_radius):
                for y in range(ti.cast(self.boids_pos[i][1],ti.i32) - self.boids_radius,ti.cast(self.boids_pos[i][1],ti.i32) + self.boids_radius):
                    self.world[x, y] = ti.Vector([self.boids_velocity[i][1]*255,self.boids_velocity[i].norm()*255,self.boids_velocity[i][0]*255])
    
    @ti.kernel
    def update_pos(self):
        for x in range(self.boids_size):
            self.boids_pos[x] = self.boids_pos[x] + self.dt * self.boids_velocity[x];
            if (self.boids_pos[x][0] > self.resx): self.boids_pos[x][0] = 1
            elif (self.boids_pos[x][1] > self.resy): self.boids_pos[x][1] = 1
            elif (self.boids_pos[x][0] < 0): self.boids_pos[x][0] = self.resx-1
            elif (self.boids_pos[x][1] < 0): self.boids_pos[x][1] = self.resy-1

    @ti.kernel
    def update_by_rules(self):
        for i in range(self.boids_size):
            avoid = ti.Vector([0,0])
            cnt=0
            avoid = ti.Vector([0.,0.])
            follow = ti.Vector([0.,0.])
            middle = ti.Vector([0.,0.])
            for j in range(self.boids_size):
                if i!=j:
                    dis = self.boids_pos[i] - self.boids_pos[j]
                    if dis.norm() < self.vis_radius:
                        cnt += 1
                        self.boids_velocity[i] += dis.normalized()/dis.norm()*self.max_speed
                        follow += self.boids_velocity[j]
                        middle = middle + self.boids_pos[j]
            if cnt != 0:
                middle = middle/cnt - self.boids_pos[i]            
                self.boids_velocity[i] += (middle + (follow/cnt)).normalized()
                if self.boids_velocity[i].norm() > self.max_speed:
                    self.boids_velocity[i] = self.boids_velocity[i].normalized()*self.max_speed
    
    def update(self):
        self.update_by_rules()
        self.update_pos()
        self.render() 