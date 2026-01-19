import pytest, sys, numpy as np, random

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.noise_models import apply_global_depolarizing_noise_end_of_circuit

def test_no_noise_when_p_zero():
    """p = 0 → output must be identical to input"""
    samples = np.array([[1, -1], [-1, 1], [1, 1]])
    noisy = apply_global_depolarizing_noise_end_of_circuit(samples, p=0.0)
    np.testing.assert_array_equal(noisy, samples)

def test_full_noise_when_p_one():
    """p = 1 → every row should be replaced with random bitstring"""
    samples = np.ones((5, 3), dtype=int)
    random.seed(42)
    noisy = apply_global_depolarizing_noise_end_of_circuit(samples, p=1.0)
    assert not np.all(noisy == samples), "All samples were expected to change at p=1"
    for row in noisy:
        assert set(row).issubset({-1, 1})

def test_output_shape_matches_input():
    samples = np.ones((100, 4), dtype=int)
    noisy = apply_global_depolarizing_noise_end_of_circuit(samples, p=0.3)
    assert noisy.shape == samples.shape

def test_statistical_global_noise_rate():
    """With p = 0.2, expect ~20% of rows to be replaced"""
    nshots = 10000
    nqubits = 5
    samples = np.ones((nshots, nqubits), dtype=int)
    p = 0.2
    random.seed(123)

    noisy = apply_global_depolarizing_noise_end_of_circuit(samples, p)
    num_changed = np.sum(np.any(noisy != samples, axis=1))
    empirical_rate = num_changed / nshots

    assert abs(empirical_rate - p) < 0.02, f"Expected flip rate ≈ {p}, got {empirical_rate}"

def test_output_values_still_in_plus_minus_one():
    samples = np.ones((1000, 4), dtype=int)
    noisy = apply_global_depolarizing_noise_end_of_circuit(samples, p=0.8)
    assert np.all(np.isin(noisy, [-1, 1])), "Outputs must remain in {-1, 1}"