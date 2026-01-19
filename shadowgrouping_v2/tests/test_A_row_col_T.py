import pytest, sys, numpy as np, itertools

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.noise_models import A_row_col_T, generate_n_qubit_assignment_matrix_randomly

def test_A_row_col_T_identity_assignment():
    # All assignment matrices are identity (perfect readout)
    S_dict = {0: np.eye(2), 1: np.eye(2)}
    # Should give 1.0 only when row == col
    for row in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        for col in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            expected = 1.0 if row == col else 0.0
            assert A_row_col_T(row, col, S_dict) == expected

def test_A_row_col_T_matches_full_assignment_matrix():
    """Compare against full matrix for n=2"""
    S_dict = {
        0: np.array([[0.9, 0.1], [0.2, 0.8]]),
        1: np.array([[0.95, 0.05], [0.1, 0.9]])
    }
    # Build full 4x4 assignment matrix A = S1 ⊗ S0
    A_full = np.kron(S_dict[0], S_dict[1])

    for row_idx in range(4):
        for col_idx in range(4):
            row = [(row_idx >> 1) & 1, row_idx & 1]
            col = [(col_idx >> 1) & 1, col_idx & 1]
            approx = A_row_col_T(row, col, S_dict)
            exact = A_full[row_idx, col_idx]
            assert np.isclose(approx, exact), f"Mismatch at ({row}, {col})"
            
def test_tensor_product_consistency_large_n():
    """Test A_row_col_T against explicit tensor product for n = 3, 4, 5"""
    for n in [3, 4, 5]:
        # Generate separate 2x2 assignment matrix for each qubit
        S_dict = {
            i: generate_n_qubit_assignment_matrix_randomly(1, p=0.95, seed=100 + i)
            for i in range(n)
        }

        # Compute full tensor product matrix explicitly
        S_full = S_dict[0]
        for i in range(1, n):
            S_full = np.kron(S_full, S_dict[i])  # note: reverse order to match row/col convention

        d = 2**n
        # Randomly sample 10 (row, col) index pairs to compare
        rng = np.random.default_rng(999)
        for _ in range(10):
            row_idx = rng.integers(0, d)
            col_idx = rng.integers(0, d)

            # Convert to bitstrings
            row = [(row_idx >> i) & 1 for i in reversed(range(n))]
            col = [(col_idx >> i) & 1 for i in reversed(range(n))]

            approx = A_row_col_T(row, col, S_dict)
            exact = S_full[row_idx, col_idx]
            assert np.isclose(approx, exact, atol=1e-12), (
                f"Mismatch at row {row}, col {col}, "
                f"computed: {approx:.6f}, expected: {exact:.6f}"
            )

def test_output_is_probabilistic():
    """Ensure output is between 0 and 1 for valid random local assignment matrices"""
    n = 3
    S_dict = {
        i: generate_n_qubit_assignment_matrix_randomly(1, p=0.95, seed=100 + i)
        for i in range(n)
    }
    for _ in range(10):
        row = np.random.randint(0, 2, size=n).tolist()
        col = np.random.randint(0, 2, size=n).tolist()
        val = A_row_col_T(row, col, S_dict)
        assert 0.0 <= val <= 1.0, f"Matrix element out of bounds: {val}"

def test_invalid_inputs_raise_errors():
    S = {0: np.eye(2), 1: np.eye(2)}
    with pytest.raises(AssertionError):
        A_row_col_T([0], [0, 1], S)  # mismatched length
    with pytest.raises(AssertionError):
        A_row_col_T([0, 1], [1, 0], {0: np.eye(2)})  # missing qubit 1