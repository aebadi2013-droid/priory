import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.noise_models import apply_stochastic_bit_flips

def test_no_flips_when_prob_zero():
    samples = np.array([[0, 1], [1, 0]])
    p0to1 = np.array([0.0, 0.0])
    p1to0 = np.array([0.0, 0.0])
    noisy = apply_stochastic_bit_flips(samples, p0to1, p1to0, seed=42)
    assert np.array_equal(noisy, samples), "Output should be identical when all probabilities are zero"

def test_uncorrelated_bit_flips_only():
    samples = np.zeros((1000, 2), dtype=int)
    p0to1 = np.array([0.2, 0.0])
    p1to0 = np.array([0.0, 0.0])
    noisy = apply_stochastic_bit_flips(samples, p0to1, p1to0, seed=0)
    # Expect around 20% of column 0 to be flipped
    flipped = noisy[:, 0].sum()
    assert 150 < flipped < 250, f"Unexpected flip rate: {flipped/10:.1f}%"

def test_correlated_bit_flips_only():
    samples = np.zeros((1000, 4), dtype=int)
    p0to1 = np.zeros(4)
    p1to0 = np.zeros(4)
    correlated_groups = [[0, 1], [2, 3]]
    p_corr = [0.1, 0.05]
    noisy = apply_stochastic_bit_flips(samples, p0to1, p1to0, correlated_groups, p_corr, seed=1)
    # Expect about 10% of rows to have both [0, 1] flipped, 5% for [2, 3]
    assert np.mean((noisy[:, 0] == 1) & (noisy[:, 1] == 1)) > 0.08
    assert np.mean((noisy[:, 2] == 1) & (noisy[:, 3] == 1)) > 0.03

def test_combined_flips():
    samples = np.zeros((500, 3), dtype=int)
    p0to1 = np.array([0.1, 0.2, 0.3])
    p1to0 = np.array([0.0, 0.0, 0.0])
    correlated_groups = [[0, 2]]
    p_corr = [0.1]
    noisy = apply_stochastic_bit_flips(samples, p0to1, p1to0, correlated_groups, p_corr, seed=123)
    # Flip rates should roughly reflect both uncorrelated and correlated noise
    assert np.any(noisy.sum(axis=0) > 0), "Some bits should be flipped"

def test_reproducibility_with_seed():
    samples = np.zeros((100, 2), dtype=int)
    p0to1 = np.array([0.1, 0.2])
    p1to0 = np.array([0.0, 0.0])
    out1 = apply_stochastic_bit_flips(samples, p0to1, p1to0, seed=42)
    out2 = apply_stochastic_bit_flips(samples, p0to1, p1to0, seed=42)
    assert np.array_equal(out1, out2), "Results should be identical with same seed"
    
def test_realistic_samples_uncorrelated():
    # Random bitstring samples (1000 shots, 4 qubits)
    rng = np.random.default_rng(42)
    samples = rng.integers(0, 2, size=(1000, 4))

    p0to1 = np.array([0.1, 0.2, 0.1, 0.0])
    p1to0 = np.array([0.05, 0.0, 0.1, 0.2])

    noisy = apply_stochastic_bit_flips(samples, p0to1, p1to0, seed=123)

    # Expect nontrivial changes
    assert not np.array_equal(samples, noisy), "Noisy output should differ from input"

    # Check that both flips occurred
    flips_0to1 = np.logical_and(samples == 0, noisy == 1)
    flips_1to0 = np.logical_and(samples == 1, noisy == 0)
    assert flips_0to1.any(), "Some 0→1 flips should occur"
    assert flips_1to0.any(), "Some 1→0 flips should occur"

def test_realistic_samples_with_correlation():
    rng = np.random.default_rng(123)
    samples = rng.integers(0, 2, size=(1000, 5))

    p0to1 = np.zeros(5)
    p1to0 = np.zeros(5)

    correlated_groups = [[0, 1], [2, 4]]
    p_corr = [0.05, 0.08]

    noisy = apply_stochastic_bit_flips(samples, p0to1, p1to0, correlated_groups, p_corr, seed=1)

    # Check that entire correlated groups were flipped simultaneously
    g1 = np.logical_and(noisy[:, 0] != samples[:, 0], noisy[:, 1] != samples[:, 1])
    g2 = np.logical_and(noisy[:, 2] != samples[:, 2], noisy[:, 4] != samples[:, 4])
    assert np.mean(g1) > 0.03
    assert np.mean(g2) > 0.05

