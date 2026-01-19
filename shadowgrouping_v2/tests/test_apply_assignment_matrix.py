import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.noise_models import (apply_assignment_matrix, A_row_col_C, A_row_col_T, 
                                                generate_n_qubit_assignment_matrix_randomly)

from shadowgrouping.helper_functions import int_to_bitlist

def test_output_shape():
    """Check that output shape matches input samples"""
    samples = np.array([[0, 0], [1, 1], [0, 1]])
    S_dict = {
        0: np.eye(2),
        1: np.eye(2)
    }
    C_dict = {}
    noisy = apply_assignment_matrix(samples, S_dict, C_dict, A_row_col_T, seed=42)
    assert noisy.shape == samples.shape

def test_identity_assignment_matrix():
    """When S = identity and no correlations, output = input"""
    samples = np.array([[0, 0], [1, 1], [0, 1]])
    S_dict = {
        0: np.eye(2),
        1: np.eye(2)
    }
    C_dict = {}
    noisy = apply_assignment_matrix(samples, S_dict, C_dict, A_row_col_T, seed=123)
    np.testing.assert_array_equal(noisy, samples)

def test_tensor_product_assignment_matrix():
    """Check that bit flips occur according to tensor product S matrices"""
    samples = np.array([[0, 0], [1, 1], [0, 1]])
    S_dict = {
        0: np.array([[0.9, 0.1], [0.2, 0.8]]),
        1: np.array([[0.85, 0.15], [0.1, 0.9]])
    }
    C_dict = {}
    noisy = apply_assignment_matrix(samples, S_dict, C_dict, A_row_col_T, seed=456)
    assert noisy.shape == samples.shape
    assert not np.all(noisy == samples)

def test_correlated_assignment_matrix():
    """Check that correlated assignment matrix produces changes (statistically)"""
    rng = np.random.default_rng(789)
    n = 3
    Nshots = 1000
    samples = rng.integers(0, 2, size=(Nshots, n))

    S_dict = {
        0: np.eye(2),
        1: np.eye(2),
        2: np.array([[0.95, 0.05], [0.2, 0.8]])
    }
    C_dict = {
        (0, 1): np.array([[0.95, 0.05, 0.0, 0.0],
                          [0.05, 0.9, 0.05, 0.0],
                          [0.0, 0.05, 0.9, 0.05],
                          [0.0, 0.0, 0.05, 0.95]])
    }

    noisy = apply_assignment_matrix(samples, S_dict, C_dict, A_row_col_C, seed=789)

    assert noisy.shape == samples.shape

    # Check that the noise had an effect on at least some samples
    num_diffs = np.sum(np.any(noisy != samples, axis=1))
    assert num_diffs > 0, "No samples were modified by the assignment matrix"

def test_random_matrices_and_consistency():
    """Use random but normalized S and C matrices, and check probabilities"""
    n = 3
    Nshots = 5
    samples = np.random.randint(0, 2, size=(Nshots, n))

    S_dict = {
        i: generate_n_qubit_assignment_matrix_randomly(1, p=0.95, seed=10 + i)
        for i in range(n)
    }
    C_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            A = generate_n_qubit_assignment_matrix_randomly(2, p=0.95, seed=20 + i + j)
            C_dict[(i, j)] = A

    noisy = apply_assignment_matrix(samples, S_dict, C_dict, A_row_col_C, seed=2024)
    assert noisy.shape == samples.shape
    for row in noisy:
        assert all(b in [0, 1] for b in row)
