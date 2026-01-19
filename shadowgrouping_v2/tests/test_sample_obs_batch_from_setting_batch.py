import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import (
    sample_obs_batch_from_setting_batch, sample_obs_batch_from_setting_batch_numba, sample_obs_batch_from_setting_batch_standalone)

def test_sample_obs_batch_from_setting_batch_basic():
    O = [[0, 2, 0], [1, 0, 3], [0, 0, 0]]
    P = [[1, 2, 3], [1, 0, 3]]
    expected = np.array([[True, False], [True, True], [True, True]])
    result_1 = sample_obs_batch_from_setting_batch(O, P)
    result_2 = sample_obs_batch_from_setting_batch_numba(np.array(O), np.array(P))
    result_3 = sample_obs_batch_from_setting_batch_standalone(O, P)
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)

def test_sample_obs_batch_from_setting_batch_all_false():
    O = [[1, 2, 3], [3, 2, 1]]
    P = [[0, 0, 0]]
    expected = np.array([[False], [False]])
    result_1 = sample_obs_batch_from_setting_batch(O, P)
    result_2 = sample_obs_batch_from_setting_batch_numba(np.array(O), np.array(P))
    result_3 = sample_obs_batch_from_setting_batch_standalone(O, P)
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)

def test_sample_obs_batch_from_setting_batch_mixed_results():
    O = [[1, 2, 3], [0, 2, 0]]
    P = [[1, 2, 3], [1, 2, 0]]
    expected = np.array([[True, False], [True, True]])
    result_1 = sample_obs_batch_from_setting_batch(O, P)
    result_2 = sample_obs_batch_from_setting_batch_numba(np.array(O), np.array(P))
    result_3 = sample_obs_batch_from_setting_batch_standalone(O, P)
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)

def test_sample_obs_batch_from_setting_batch_identity_obs():
    O = [[0, 0, 0], [0, 0, 0]]
    P = [[1, 1, 1], [2, 2, 2]]
    result_1 = sample_obs_batch_from_setting_batch(O, P)
    result_2 = sample_obs_batch_from_setting_batch_numba(np.array(O), np.array(P))
    result_3 = sample_obs_batch_from_setting_batch_standalone(O, P)
    assert np.all(result_1)
    assert np.all(result_2)
    assert np.all(result_3)

def test_sample_obs_batch_from_setting_batch_invalid_type():
    with pytest.raises(TypeError):
        sample_obs_batch_from_setting_batch_standalone("not_array", [[1, 2, 3]])
    with pytest.raises(TypeError):
        sample_obs_batch_from_setting_batch_standalone([[1, 2, 3]], "also_not_array")

def test_sample_obs_batch_from_setting_batch_invalid_shape():
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_batch_standalone([[1, 2]], [[1, 2, 3]])  # mismatched qubits
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_batch_standalone([1, 2, 3], [[1, 2, 3]])  # O_batch not 2D
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_batch_standalone([[1, 2, 3]], [1, 2, 3])  # P_batch not 2D

def test_sample_obs_batch_from_setting_batch_invalid_entries():
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_batch_standalone([[1, 2, 5]], [[1, 2, 3]])  # 5 is invalid
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_batch_standalone([[1, -1, 3]], [[1, 2, 3]])  # -1 is invalid
    with pytest.raises(ValueError):
        sample_obs_batch_from_setting_batch_standalone([[1, 2, 3]], [[1, 2, "Z"]])  # "Z" is invalid