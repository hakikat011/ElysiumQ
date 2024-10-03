
from src.data_generation.generate_sample_data import generate_sparse_matrix, generate_vector
from src.data_preparation.data_structuring import load_matrix, load_vector
from src.data_preparation.partitioning import partition_matrix
from src.classical_processing.preconditioning import precondition_block
from src.quantum_processing.hhl_circuit import solve_hhl
from src.integration.aggregator import aggregate_solutions
from src.integration.validation import validate_solution
from src.workflow.logger import setup_logger
import numpy as np

def orchestrate_workflow():
    logger = setup_logger()
    try:
        # Load data
        A = load_matrix('examples/sample_data/raw/matrix_A.csv')
        b = load_vector('examples/sample_data/raw/vector_b.csv')
        
        # Partition data
        block_size = 4  # Adjust as needed
        blocks = partition_matrix(A, b, block_size)
        
        solutions = []
        for idx, block in enumerate(blocks):
            A_block = block['A_block']
            b_block = block['b_block']
            # Precondition
            M_block = precondition_block(A_block)
            # Quantum solve
            x_block = solve_hhl(M_block, b_block)
            solutions.append(x_block)
        
        # Aggregate solutions
        x = aggregate_solutions(solutions, A.shape[0], block_size)
        
        # Validate solution
        is_valid = validate_solution(A, x, b)
        if is_valid:
            logger.info("Solution is valid.")
        else:
            logger.warning("Solution is invalid.")
        
        # Save solution
        np.save('examples/sample_data/processed/solution.npy', x)
        logger.info("Solution saved.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    