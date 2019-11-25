from mpi4py import MPI
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
import imageio
import cmath
import os
import time

from game_of_life import Game

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
amode = MPI.MODE_RDWR|MPI.MODE_CREATE

def measure_time(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()

        if rank == 0:
            file = open("./data/game_time.txt", "a")
            file.writelines(str(size) + ' ' + str((time2-time1)*1000.0) + '\n')
            file.close()
        return ret
    return wrap


class Game_Parallel():
    def __init__(self,
                 field_size=10,
                 dimensions=2,
                 agents_percent=0.5,
                 from_template=False,
                 is_periodic=True,
                 save_maps=True,
                 init_map=None):

        self.from_template = from_template
        self.size = field_size
        self.dimensions = dimensions
        self.agents_percent = agents_percent
        self.is_periodic = is_periodic
        self.save_maps = save_maps
        self.files_count = 0

        self.init_map(init_map)

        if rank == 0:
            self.agents_count = [self.sum_matrix(self.map)]

        self.shape = self.map.shape
        self.file_location = './data/'
        self.file_name = (
                'Life' +
                '_s' + str(self.size) +
                '_' + str(self.dimensions) + 'd')

        assert (1 < size) and (size <= field_size)
        self.split()

    def init_map(self, init_map):
        shape = None

        if rank == 0:
            if self.from_template:
                self.map = init_map.astype(int)
                self.size = init_map.shape[0]
            else:
                self.map = (
                        np.random.random([self.size for i in range(self.dimensions)]) < self.agents_percent
                ).astype(int)
            shape = self.map.shape

        comm.Barrier() # Wait while map is initialised on the 0-th node
        shape = comm.bcast(shape, root=0)
        common_map = self.map.astype(int) if rank == 0 else np.zeros(shape).astype(int)

        comm.Barrier() # Wait for common variable inititalisation
        comm.Bcast([common_map, MPI.INT], root=0)
        self.map = common_map
        comm.Barrier()

    def split(self):
        k, m = divmod(self.map.shape[1], size)
        a = list(range(self.map.shape[1]))

        partitions_indexes = [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(size)]
        self.partitions_indexes = partitions_indexes
        part = self.map[:, partitions_indexes[rank]]
        self.part = part

    def collect(self):
        ''' Custom GatherV '''
        if rank == 0:
            received_map = np.zeros((self.size, self.size))
            received_map[:, self.partitions_indexes[0]] = self.part

        for node in range(1, size):
            if rank == node:
                comm.send(self.part.tolist(), dest=0, tag=node)

            if rank == 0:
                data = comm.recv(source=node, tag=node)
                received_map[:, self.partitions_indexes[node]] = np.array(data)

        comm.Barrier()
        if rank == 0:
            self.collected_map = received_map.astype(int)

    def share(self):
        for node in range(size):
            if rank == node:
                self.padded = self.add_padding(self.part)
                left_neighbor = (node - 1) % size
                right_neighbor = (node + 1) % size

                comm.send(self.part[:, 0], dest=left_neighbor, tag=0)
                comm.send(self.part[:, -1], dest=right_neighbor, tag=1)

                right = comm.recv(source=right_neighbor, tag=0)
                left = comm.recv(source=left_neighbor, tag=1)

                self.padded[1:-1, 0] = left
                self.padded[1:-1, -1] = right
        comm.Barrier()

    def add_padding(self, mp):
        padding = np.zeros((mp.shape[0] + 2, mp.shape[1] + 2))
        padding[1:-1, 1:-1] = mp
        return padding

    def count_neighbours(self):
        mp = self.padded
        result = np.zeros(self.part.shape)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = self.sum_matrix(mp[i:i + 3, j:j + 3])
        return (result - self.part).astype(int)

    def step(self):
        neighbors = self.count_neighbours()
        self.part = (((self.part == 0) & (neighbors == 3)) |
                    (self.part == 1) & ((neighbors == 2) | (neighbors == 3))
                    ).astype(int)

    @measure_time
    def play(self, iterations):
        self.show(save=True,
                  file_name=self.file_location + self.file_name + 'i_0')
        for i in range(iterations):
            self.share()
            self.step()
            self.collect()

            if rank == 0:
                self.map = self.collected_map
                self.agents_count.append(self.sum_matrix(self.map))
            self.show(save=True,
                      file_name=self.file_location + self.file_name + 'i_' + str(i+1))
            pass

    def show(self, save=False, file_name=None):
        if rank == 0:
            plt.ioff()
            # plt.figure(figsize=(20, 20))
            plt.matshow(self.map, cmap=cm.binary)
            plt.title('City map')
            plt.axis('off')

            if save:
                plt.savefig(file_name + '.jpg')
                self.files_count += 1
            else:
                plt.show()
            plt.close()

    def sum_matrix(self, matrix):
        return sum(np.ravel(matrix))

    def save_gif(self):
        assert self.files_count > 0
        gif_name = self.file_location + self.file_name + '.gif'

        with imageio.get_writer(gif_name, mode='I') as writer:
            for i in range(self.files_count):
                image_path = self.file_location + self.file_name + 'i_' + str(i) + '.jpg'
                writer.append_data(imageio.imread(image_path))
                os.remove(os.path.join(image_path))

    def show_agents(self):
        if rank == 0:
            plt.ioff()
            plt.plot(self.agents_count)
            plt.title('Agents evolution')
            plt.xlabel('Time, n')
            plt.ylabel('Agents count')
            plt.savefig(self.file_location + self.file_name  + '_agents_count.jpg')
            plt.close()



##########################################################
# Random map
gp = Game_Parallel(field_size=100)
gp.play(100)

comm.Barrier()
if rank == 0:
    gp.save_gif()
    gp.show_agents()
##########################################################



# ##########################################################
# # Gosper's Glider Gun map
# # Sample map form: http://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
# glider_gun =\
# [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
#  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
#  [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#  [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#  [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
#
# X = np.zeros((100, 100))
# X[1:10,1:37] = glider_gun
#
# gg = Game_Parallel(init_map=X, from_template=True, field_size=100)
# gg.play(200)
# comm.Barrier()
# if rank == 0:
#     gg.save_gif()
#     gg.show_agents()
# ##########################################################




# ##########################################################
# small_map =\
# [[0, 1, 1],
#  [1, 1, 0],
#  [0, 1, 0]]
#
# X = np.zeros((100, 100))
# X[49:52, 49:52] = small_map
#
# gs = Game_Parallel(init_map=X, from_template=True, field_size=100)
# gs.play(200)
# comm.Barrier()
# if rank == 0:
#     gs.save_gif()
#     gs.show_agents()
# ##########################################################




##########################################################
# # Big random map
# gp = Game_Parallel(field_size=400)
# gp.play(300)
#
# comm.Barrier()
# if rank == 0:
#     gp.save_gif()
#     gp.show_agents()
##########################################################



