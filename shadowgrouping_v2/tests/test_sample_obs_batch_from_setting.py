import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import (
    sample_obs_batch_from_setting, sample_obs_batch_from_setting_numba, sample_obs_batch_from_setting_standalone)

def test_sample_obs_batch_from_setting_match():
    O_batch = np.array([[0, 2, 0], [1, 0, 3], [0, 0, 0]])
    P = np.array([1, 2, 3])
    result_1 = sample_obs_batch_from_setting(O_batch, P)
    result_2 = sample_obs_batch_from_setting_numba(O_batch, P)
    result_3 = sample_obs_batch_from_setting_standalone(O_batch, P)
    expected = np.array([True, True, True])
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)

def test_sample_obs_batch_from_setting_mismatch():
    O_batch = np.array([[1, 2, 3], [1, 1, 1], [0, 1, 2]])
    P = np.array([1, 0, 3])
    result_1 = sample_obs_batch_from_setting(O_batch, P)
    result_2 = sample_obs_batch_from_setting_numba(O_batch, P)
    result_3 = sample_obs_batch_from_setting_standalone(O_batch, P)
    expected = np.array([False, False, False])
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)
    
def test_sample_obs_batch_from_setting_mixed():
    O_batch = np.array([[1, 0, 3], [1, 2, 3], [0, 0, 0]])
    P = np.array([1, 0, 3])
    result_1 = sample_obs_batch_from_setting(O_batch, P)
    result_2 = sample_obs_batch_from_setting_numba(O_batch, P)
    result_3 = sample_obs_batch_from_setting_standalone(O_batch, P)    
    expected = np.array([True, False, True])
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)
    
def test_sample_obs_batch_from_setting_edge_cases():
    O_batch = np.empty((0, 3), dtype=int)  # 0 observables of length 3
    P = np.array([1, 2, 3])
    result_1 = sample_obs_batch_from_setting(O_batch, P)
    result_2 = sample_obs_batch_from_setting_numba(O_batch, P)
    result_3 = sample_obs_batch_from_setting_standalone(O_batch, P)
    assert result_1.shape == (0,)
    assert result_2.shape == (0,)
    assert result_3.shape == (0,)

    O_batch = np.array([[0], [2], [3]])
    P = np.array([2])
    result_1 = sample_obs_batch_from_setting(O_batch, P)
    result_2 = sample_obs_batch_from_setting_numba(O_batch, P)
    result_3 = sample_obs_batch_from_setting_standalone(O_batch, P)
    expected = np.array([True, True, False])
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)
    
def test_sample_obs_batch_from_setting_invalid_type():
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_standalone([[1, 2, 3], [1, 2, 'Z']], [1, 2, 3])
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_standalone([[1, 2, 5], [1, 2, 3]], [1, 2, 3])
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_standalone([[1, 2, 3]], [1, 2, -1])

def test_sample_obs_batch_from_setting_invalid_shape():
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_standalone([1, 2, 3], [1, 2, 3])  # O_batch not 2D
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_standalone([[1, 2], [3, 4]], [1, 2, 3])  # Length mismatch
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_standalone([[1, 2, 3]], [[1, 2, 3]])  # P not 1D