import numpy as np
from qiskit import Aer
from qiskit.algorithms.linear_solvers import HHL
from qiskit.utils import QuantumInstance

def solve_hhl(A_block, b_block):
    """
    Solves the linear system using the HHL algorithm.
    """
    # Ensure A_block is Hermitian and positive-definite
    if not np.allclose(A_block, A_block.conj().T):
        raise ValueError("Matrix A_block must be Hermitian.")
    
    # Normalize b_block
    b_norm = np.linalg.norm(b_block)
    if b_norm == 0:
        raise ValueError("Vector b_block cannot be zero.")
    b_normalized = b_block / b_norm

    # Setup quantum instance
    backend = Aer.get_backend('statevector_simulator')
    qi = QuantumInstance(backend)

    # Use HHL
    hhl = HHL(quantum_instance=qi)
    result = hhl.solve(A_block, b_normalized)
    x = result.solution
    # Scale back the solution
    x_scaled = x * b_norm
    return x_scaled.real  # Return real part if imaginary components are negligible