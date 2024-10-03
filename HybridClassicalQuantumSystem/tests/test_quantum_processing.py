
import unittest
import numpy as np
from src.quantum_processing.hhl_circuit import solve_hhl

class TestQuantumProcessing(unittest.TestCase):
    def test_solve_hhl(self):
        A_block = np.array([[1, 0], [0, 2]], dtype=np.float64)
        b_block = np.array([1, 1], dtype=np.float64)
        x = solve_hhl(A_block, b_block)
        expected = np.linalg.solve(A_block, b_block)
        np.testing.assert_array_almost_equal(x, expected, decimal=1)

if __name__ == '__main__':
    unittest.main()
    