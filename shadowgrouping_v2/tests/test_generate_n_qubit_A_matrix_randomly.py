import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.noise_models import generate_n_qubit_assignment_matrix_randomly

def test_matrix_shape():
    for n in [1, 2, 3]:
        A = generate_n_qubit_assignment_matrix_randomly(n, p=0.9, seed=42)
        d = 2**n
        assert A.shape == (d, d), f"Matrix shape mismatch for {n} qubits"

def test_column_stochasticity():
    A = generate_n_qubit_assignment_matrix_randomly(3, p=0.95, seed=123)
    col_sums = A.sum(axis=0)
    assert np.allclose(col_sums, 1.0), "Each column should sum to 1"

def test_non_negative_entries():
    A = generate_n_qubit_assignment_matrix_randomly(2, p=0.95, seed=21)
    assert np.all(A >= 0), "All matrix entries should be non-negative"

def test_diagonal_average_close_to_p():
    p_target = 0.92
    A = generate_n_qubit_assignment_matrix_randomly(3, p=p_target, seed=7)
    diag_mean = np.mean(np.diag(A))
    assert abs(diag_mean - p_target) < 0.02, f"Diagonal mean {diag_mean:.3f} deviates too much from target {p_target}"

def test_diagonal_entries_in_range():
    A = generate_n_qubit_assignment_matrix_randomly(4, p=0.95, seed=99)
    diag = np.diag(A)
    assert np.all((diag >= 0) & (diag <= 1)), "Diagonal entries must be in [0, 1]"