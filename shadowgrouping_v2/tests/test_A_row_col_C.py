import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.noise_models import (A_row_col_C, A_row_col_T, generate_n_qubit_assignment_matrix_randomly)

def test_single_correlated_pair_only():
    """Test when only one correlated pair is present, no S_matrix_dict needed."""
    C = np.array([[0.9, 0.05, 0.03, 0.02],
                  [0.05, 0.85, 0.05, 0.05],
                  [0.03, 0.05, 0.9, 0.02],
                  [0.02, 0.05, 0.02, 0.91]])
    C_dict = {(0, 1): C}
    S_dict = {0: np.eye(2), 1: np.eye(2)}  # doesn't affect result here

    for r in range(4):
        for c in range(4):
            row = [(r >> 1) & 1, r & 1]
            col = [(c >> 1) & 1, c & 1]
            expected = C[r, c]
            result = A_row_col_C(row, col, S_dict, C_dict)
            assert np.isclose(result, expected), f"Mismatch at {row}, {col}"

def test_correlated_pair_plus_one_extra_qubit():
    """Check that the third qubit is treated independently"""
    C = np.ones((4, 4)) * 0.25  # uniform assignment matrix
    C_dict = {(0, 1): C}

    S2 = np.array([[0.9, 0.1],
                   [0.2, 0.8]])
    S_dict = {
        0: np.eye(2),
        1: np.eye(2),
        2: S2
    }

    row = [0, 1, 1]
    col = [1, 0, 0]

    expected = C[1, 2] * S2[1, 0]
    result = A_row_col_C(row, col, S_dict, C_dict)
    assert np.isclose(result, expected)

def test_multiple_correlated_pairs():
    """Sum over multiple correlated pairs"""
    C01 = np.array([[0.95, 0.05, 0.0, 0.0],
                    [0.05, 0.9, 0.05, 0.0],
                    [0.0, 0.05, 0.9, 0.05],
                    [0.0, 0.0, 0.05, 0.95]])

    C02 = np.array([[0.9, 0.1, 0.0, 0.0],
                    [0.1, 0.85, 0.05, 0.0],
                    [0.0, 0.05, 0.85, 0.1],
                    [0.0, 0.0, 0.1, 0.9]])

    C_dict = {(0, 1): C01, (0, 2): C02}
    S_dict = {
        0: np.eye(2),
        1: np.array([[0.9, 0.1], [0.2, 0.8]]),
        2: np.array([[0.92, 0.08], [0.15, 0.85]])
    }

    row = [1, 0, 1]
    col = [1, 1, 0]

    # Compute expected manually
    term_01 = C01[2, 3] * S_dict[2][1, 0]  # correlated on (0,1), independent qubit 2
    term_02 = C02[3, 2] * S_dict[1][0, 1]  # correlated on (0,2), independent qubit 1
    expected = 1/2*(term_01 + term_02)

    result = A_row_col_C(row, col, S_dict, C_dict)
    assert np.isclose(result, expected, atol=1e-12)

def test_output_is_probability():
    """Ensure matrix element is in [0, 1]"""
    n = 3
    rng = np.random.default_rng(123)
    S_dict = {
        i: rng.random((2, 2)) for i in range(n)
    }
    # Normalize columns of each S
    for S in S_dict.values():
        S /= S.sum(axis=0, keepdims=True)

    C_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            C = rng.random((4, 4))
            C /= C.sum(axis=0, keepdims=True)
            C_dict[(i, j)] = C

    for _ in range(10):
        row = rng.integers(0, 2, size=n).tolist()
        col = rng.integers(0, 2, size=n).tolist()
        val = A_row_col_C(row, col, S_dict, C_dict)
        assert 0.0 <= val <= 1.0, f"Value out of range: {val}"

def test_invalid_inputs_raise_error():
    S_dict = {0: np.eye(2), 1: np.eye(2)}
    C_dict = {(0, 0): np.eye(4)}  # invalid: duplicate qubit index

    with pytest.raises(AssertionError):
        A_row_col_C([0, 1], [1, 0], S_dict, C_dict)

    with pytest.raises(AssertionError):
        A_row_col_C([0], [0, 1], S_dict, C_dict)
        
def test_A_row_col_C_matches_T_for_one_pair_and_tensor_C():
    """Test A_row_col_C matches A_row_col_T when C_ij = S_i ⊗ S_j and all other qubits use S_k."""

    S0 = np.array([[0.95, 0.05], [0.1, 0.9]])
    S1 = np.array([[0.9, 0.1], [0.2, 0.8]])
    S2 = np.array([[0.93, 0.07], [0.15, 0.85]])

    S_dict = {0: S0, 1: S1, 2: S2}
    C_dict = {(0, 1): np.kron(S0, S1)}

    for _ in range(10):
        row = np.random.randint(0, 2, size=3).tolist()
        col = np.random.randint(0, 2, size=3).tolist()

        val_T = A_row_col_T(row, col, S_dict)
        val_C = A_row_col_C(row, col, S_dict, C_dict)

        assert np.isclose(val_T, val_C, atol=1e-12), f"Mismatch: {val_T} vs {val_C}"

def test_A_row_col_C_with_random_matrices():
    """Test A_row_col_C using randomly generated assignment matrices"""

    n = 3
    rng = np.random.default_rng(42)

    # Generate 2x2 matrices for each qubit
    S_dict = {
        i: generate_n_qubit_assignment_matrix_randomly(1, p=0.95, seed=100 + i)
        for i in range(n)
    }

    # Generate 4x4 matrices for each pair
    C_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            A = generate_n_qubit_assignment_matrix_randomly(2, p=0.95, seed=200 + i + j)
            C_dict[(i, j)] = A

    # Evaluate random entries
    for _ in range(10):
        row = rng.integers(0, 2, size=n).tolist()
        col = rng.integers(0, 2, size=n).tolist()
        val = A_row_col_C(row, col, S_dict, C_dict)
        assert 0.0 <= val <= 1.0, f"Invalid output: {val}"