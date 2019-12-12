# Conways'Game of Life

In this repository, I provide several implementations of Conway's' Game of Life.

[General description](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)

Rules for 3D version picked from this [article](https://wpmedia.wolfram.com/uploads/sites/13/2018/02/01-3-1.pdf)

[Numba 3D version](https://github.com/NesterukSergey/Conways_Game_of_Life/blob/master/3D.ipynb) shows some initial map samples optimized with numba jit.

## Sample 3D maps:


* Random map

![Random 3d map](https://github.com/NesterukSergey/Conways_Game_of_Life/blob/master/data/3dLife_random_s10g.gif)



* Glider Gun

![Glider 3d map](https://github.com/NesterukSergey/Conways_Game_of_Life/blob/master/data/3dLifeglider_s10.gif)



* Stable map

![Stable 3d map](https://github.com/NesterukSergey/Conways_Game_of_Life/blob/master/data/3dLifecollision_s10.gif)



Larger maps are not easy to visualise good.


[MPI version](https://github.com/NesterukSergey/Conways_Game_of_Life/blob/master/game_of_life_parallel.ipynb) is an implementation that distributes computations between several nodes. It is tested on Skolkovo [Zhores supercomputer](https://sk.ru/news/b/pressreleases/archive/2019/01/18/uchenye-skolteha-sozdali-superkompyuter-zhores.aspx).

To run parallel scripts mpi4py is required:
`[sudo] pip install mpi4py`

To run parallel scripts enter in command line: `mpirun -n [processes_number] python [filename]`


## Sample 2D maps:


* Glider Gun

![game of life](https://github.com/NesterukSergey/Conways_Game_of_Life/blob/master/data/Life_s100_2d_gun.gif)


* Random map

![game of life](https://github.com/NesterukSergey/Conways_Game_of_Life/blob/master/data/Life_s400_2d_big.gif)


