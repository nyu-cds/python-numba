#
# CUDA kernel to multiple the elements of a 2D array by 2
#
from __future__ import division
from numba import cuda
import numpy
import math

@cuda.jit
def my_kernel_2D(io_array):
    x, y = cuda.grid(2)
    if x < io_array.shape[0] and y < io_array.shape[1]:
        io_array[x, y] *= 2 # do the computation
        
data = numpy.ones((16, 16))
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(data.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(data.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
my_kernel_2D[blockspergrid, threadsperblock](data)
print(data)
