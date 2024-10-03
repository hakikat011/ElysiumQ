
import numpy as np

def partition_matrix(A, b, block_size):
    """
    Partitions matrix A and vector b into blocks.
    """
    n = A.shape[0]
    blocks = []
    for i in range(0, n, block_size):
        A_block = A[i:i+block_size, i:i+block_size]
        b_block = b[i:i+block_size]
        blocks.append({'A_block': A_block, 'b_block': b_block})
    return blocks
    