
from numba import cuda
import numpy as np
from .cuda_kernels import precondition_kernel

def precondition_block(A_block):
    """
    Applies preconditioning to a single block using CUDA.
    """
    A_device = cuda.to_device(A_block)
    M_device = cuda.device_array_like(A_block)
    
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(A_block.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A_block.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    precondition_kernel[blockspergrid, threadsperblock](A_device, M_device)
    M = M_device.copy_to_host()
    return M
    