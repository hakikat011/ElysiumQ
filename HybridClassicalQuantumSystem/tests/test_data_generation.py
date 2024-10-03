
import unittest
from src.data_generation.generate_sample_data import generate_sparse_matrix, generate_vector

class TestDataGeneration(unittest.TestCase):
    def test_generate_sparse_matrix(self):
        A = generate_sparse_matrix(10, 0.8)
        self.assertEqual(A.shape, (10, 10))
        # Add more assertions

    def test_generate_vector(self):
        A = generate_sparse_matrix(10, 0.8)
        b, x_true = generate_vector(A)
        self.assertEqual(b.shape, (10,))
        self.assertEqual(x_true.shape, (10,))
        # Add more assertions

if __name__ == '__main__':
    unittest.main()
    