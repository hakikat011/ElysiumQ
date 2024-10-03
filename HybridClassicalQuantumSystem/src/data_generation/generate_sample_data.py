import numpy as np
import pandas as pd
import os

def generate_sparse_matrix(size, sparsity=0.8):
    """
    Generates a diagonally dominant sparse matrix.
    """
    A = np.random.rand(size, size)
    # Apply sparsity
    mask = np.random.rand(size, size) < sparsity
    A[mask] = 0
    # Make diagonally dominant
    for i in range(size):
        A[i, i] = np.sum(np.abs(A[i])) + 1
    # Ensure Hermitian
    A = (A + A.T) / 2
    return A

def generate_vector(A):
    """
    Generates vector b such that A x = b for a known x.
    """
    x_true = np.random.rand(A.shape[1])
    b = A @ x_true
    return b, x_true

def save_data(A, b, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    pd.DataFrame(A).to_csv(os.path.join(data_dir, 'matrix_A.csv'), index=False)
    pd.DataFrame({'b': b}).to_csv(os.path.join(data_dir, 'vector_b.csv'), index=False)

def main():
    size = 16  # Adjust size as needed
    data_dir = 'examples/sample_data/raw/'
    A = generate_sparse_matrix(size)
    b, x_true = generate_vector(A)
    save_data(A, b, data_dir)
    print(f"Sample data generated in {data_dir}")

if __name__ == "__main__":
    main()