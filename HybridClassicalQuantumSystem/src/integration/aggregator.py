
import numpy as np

def aggregate_solutions(solutions, total_size, block_size):
    x = np.zeros(total_size)
    for idx, sol in enumerate(solutions):
        start = idx * block_size
        end = start + len(sol)
        x[start:end] = sol
    return x
    