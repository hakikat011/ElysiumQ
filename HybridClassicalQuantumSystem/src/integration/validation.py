
import numpy as np

def validate_solution(A, x, b, tolerance=1e-3):
    residual = np.linalg.norm(A @ x - b)
    return residual < tolerance
    