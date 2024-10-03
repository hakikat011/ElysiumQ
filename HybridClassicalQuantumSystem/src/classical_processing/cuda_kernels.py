
from numba import cuda
import numpy as np

@cuda.jit
def precondition_kernel(A, M):
    """
    Simple diagonal preconditioning kernel.
    """
    row, col = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        if row == col:
            if A[row, col] != 0:
                M[row, col] = 1.0 / A[row, col]
            else:
                M[row, col] = 0.0
        else:
            M[row, col] = 0.0
    