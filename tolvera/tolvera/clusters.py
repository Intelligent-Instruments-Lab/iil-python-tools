import taichi as ti
import numpy as np

from iipyper import run

class Clusters:
    def __init__(self, max_n, k) -> None:
        self.max_n = max_n
        self.k = k
        self.centroids = ti.field(ti.f32, shape=(k, 2))
        self.cluster_assignment = ti.field(ti.i32, shape=self.max_n)

    @staticmethod
    def _random_centroids(particle_positions, k):
        return particle_positions[np.random.choice(particle_positions.shape[0], k, replace=False)]

    def initialize_centroids(self, method='random'):
        '''
        Initialize centroids for k-means clustering.

        Args:
            method (str, optional): Method to initialize centroids. Currently only 'random' is supported. Defaults to 'random'.
        '''
        particle_positions = self.get_positions_of_active()
        particle_positions_np = np.array(particle_positions).reshape((-1, 2))

        if method == 'random':
            centroids_np = self._random_centroids(particle_positions_np, self.k)

        self.centroids.from_numpy(centroids_np)

    @ti.kernel
    def assign_clusters(self):
        '''
        Assign particles to the nearest centroid, updating the cluster_assignment field.
        '''
        for i in range(self.active_count[None]):
            p = self.pos_of_active[i]
            
            min_distance = ti.inf
            for k in range(self.k):
                centroid = self.centroids[k]
                distance = (p - centroid).norm_sqr()
                
                if distance < min_distance:
                    min_distance = distance
                    self.cluster_assignment[i] = k

    @ti.kernel
    def update_centroids(self):
        '''
        Update the centroids based on the particles assigned to them (average of particle positions in each cluster).
        '''
        new_centroids = ti.Vector.field(2, ti.f32, self.k)()
        clusters_count = ti.field(ti.i32, self.k)()

        for i in range(self.active_count[None]):
            p = self.pos_of_active[i]
            cluster_idx = self.cluster_assignment[i]
            new_centroids[cluster_idx] += p
            clusters_count[cluster_idx] += 1

        for k in range(self.k):
            if clusters_count[k] != 0:
                self.centroids[k] = new_centroids[k] / clusters_count[k]

    def perform_clustering(self, max_iter=10):
        '''
        Performs k-means clustering on active particles for a given number of iterations.

        Args:
            max_iter (int, optional): Maximum number of iterations to perform k-means clustering. Defaults to 10.
        '''
        self.initialize_centroids()

        for i in range(max_iter):
            self.assign_clusters()
            self.update_centroids()

