import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.noise_models import apply_local_depolarizing_noise_end_of_circuit

def test_identity_channel_no_effect():
    """If p=0 for all qubits, samples must remain unchanged"""
    samples = np.array([[1, -1], [-1, 1], [1, 1]])
    meas_basis = "XZ"
    p_array = [0.0, 0.0]
    noisy = apply_local_depolarizing_noise_end_of_circuit(samples, meas_basis, p_array)
    np.testing.assert_array_equal(noisy, samples)

def test_identity_basis_ignored():
    """'I' basis qubits should not be affected, even with nonzero p"""
    samples = np.array([[1, -1], [-1, 1], [1, 1]])
    meas_basis = "IZ"
    p_array = [0.5, 0.0]  # nonzero p for 'I', should be ignored
    noisy = apply_local_depolarizing_noise_end_of_circuit(samples, meas_basis, p_array)
    # qubit 0 should remain unchanged
    assert np.all(noisy[:, 0] == samples[:, 0])

def test_shape_preserved():
    """Check that output shape matches input shape"""
    nshots, nqubits = 100, 4
    samples = np.ones((nshots, nqubits), dtype=int)
    meas_basis = "XZYY"
    p_array = [0.1, 0.1, 0.1, 0.1]
    noisy = apply_local_depolarizing_noise_end_of_circuit(samples, meas_basis, p_array)
    assert noisy.shape == samples.shape

def test_statistical_flip_rate():
    """Verify that the flip rate approximates 2p/3 for nonzero p"""
    nshots = 10000
    samples = np.ones((nshots, 1), dtype=int)  # All +1
    meas_basis = "Z"
    p = 0.15  # expect ~10% flips
    p_array = [p]

    noisy = apply_local_depolarizing_noise_end_of_circuit(samples, meas_basis, p_array)
    flips = np.sum(noisy[:, 0] == -1)
    empirical_rate = flips / nshots
    expected = 2 * p / 3

    assert abs(empirical_rate - expected) < 0.02, f"Expected ~{expected}, got {empirical_rate}"

def test_invalid_basis_raises():
    """Invalid Pauli string should raise ValueError"""
    samples = np.ones((3, 2), dtype=int)
    meas_basis = "AZ"
    p_array = [0.1, 0.2]
    with pytest.raises(ValueError):
        apply_local_depolarizing_noise_end_of_circuit(samples, meas_basis, p_array)
        
def test_multi_qubit_noise_application():
    """Test that noise is applied nontrivially to multiple qubits"""
    nshots = 10000
    samples = np.ones((nshots, 3), dtype=int)  # Start with all +1s
    meas_basis = "XYZ"
    p_array = [0.15, 0.30, 0.45]  # Different noise levels on all qubits

    noisy = apply_local_depolarizing_noise_end_of_circuit(samples, meas_basis, p_array)

    # Compute empirical flip rates
    flip_rates = np.mean(noisy != 1, axis=0)
    expected_rates = np.array([2*p/3 for p in p_array])

    for q in range(3):
        assert abs(flip_rates[q] - expected_rates[q]) < 0.02, (
            f"Qubit {q} flip rate off: expected ~{expected_rates[q]}, got {flip_rates[q]}"
        )