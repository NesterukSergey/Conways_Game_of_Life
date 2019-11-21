from mpi4py import MPI
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
import imageio
from IPython.display import HTML
import cmath
from scipy import signal
import os


class Game:
    def __init__(self,
                 field_size=10,
                 dimensions=2,
                 agents_percent=0.5,
                 is_periodic=True,
                 save_maps=True,
                 init_map=None):

        self.size = field_size
        self.dimensions = dimensions
        self.agents_percent = agents_percent
        self.is_periodic = is_periodic
        self.save_maps = save_maps
        self.files_count = 0

        if init_map:
            self.map = init_map
            self.dimensions = init_map.shape[0]
        else:
            self.map = (
                    np.random.random([self.size for i in range(self.dimensions)]) < self.agents_percent
            ).astype(int)

        self.shape = tuple([self.size for i in range(self.dimensions)])
        self.file_location = './data/'
        self.file_name = (
                'Life' +
                '_s' + str(self.size) +
                '_' + str(self.dimensions) + 'd')

    def save_gif(self):
        assert self.files_count > 0
        gif_name = self.file_location + self.file_name + '.gif'

        with imageio.get_writer(gif_name, mode='I') as writer:
            for i in range(self.files_count):
                image_path = self.file_location + self.file_name + 'i_' + str(i) + '.jpg'
                writer.append_data(imageio.imread(image_path))
                os.remove(os.path.join(image_path))

        return gif_name

    def play(self, iterations):
        for i in range(iterations):
            self.step()
            if self.save_maps:
                self.show(save=True,
                          file_name=self.file_location + self.file_name + 'i_' + str(i))

    def step(self):
        neighbours = self._count_neighbours()
        self.map = (((self.map == 0) & (neighbours == 3)) |
                    (self.map == 1) & ((neighbours == 2) | (neighbours == 3))
                    ).astype(int)

    def show(self, save=False, file_name=None):
        plt.ioff()
        plt.matshow(self.map, cmap=cm.binary)
        plt.title('City map')
        plt.axis('off')

        if save:
            plt.savefig(file_name + '.jpg')
            self.files_count += 1
        else:
            plt.show()

        plt.close()

    def _count_neighbours(self):
        mp = self.add_padding(self.map)
        result = np.zeros(self.shape)

        if self.dimensions == 2:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self.sum_matrix(mp[i:i + 3, j:j + 3])
            return (result - self.map).astype(int)

        if self.dimensions == 3:
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(self.shape[2]):
                        result[i, j, k] = self.sum_matrix(mp[i:i + 3, j:j + 3, k:k + 3])
            return (result - self.map).astype(int)

    def add_padding(self, mp):
        padding = np.zeros([i + 2 for i in self.shape])

        if self.dimensions == 2:
            padding[1:-1, 1:-1] = mp

            if self.is_periodic:
                padding[:, -1] = padding[:, 1]
                padding[:, 0] = padding[:, -2]

        if self.dimensions == 3:
            padding[1:-1, 1:-1, 1:-1] = mp

            if self.is_periodic:
                padding[:, :, -1] = padding[:, :, 1]
                padding[:, :, 0] = padding[:, :, -2]

        return padding

    def remove_padding(self, mp):
        if self.dimensions == 2:
            return padding[1:-1, 1:-1]
        if self.dimensions == 3:
            return padding[1:-1, 1:-1, 1:-1]

    def sum_matrix(self, matrix):
        return sum(np.ravel(matrix))
