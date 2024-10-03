import pandas as pd
import numpy as np

def load_matrix(file_path):
    A = pd.read_csv(file_path, header=None).values
    return A

def load_vector(file_path):
    b = pd.read_csv(file_path, header=None).values.flatten()
    return b
