import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import (
    sample_obs_from_setting, sample_obs_from_setting_numba, sample_obs_from_setting_standalone)

def test_sample_obs_from_setting_match():
    assert sample_obs_from_setting([1, 0, 3], [1, 0, 3]) == True
    assert sample_obs_from_setting([0, 0, 0], [1, 2, 3]) == True
    assert sample_obs_from_setting([0, 2, 0], [1, 2, 3]) == True
    assert sample_obs_from_setting_numba(np.array([1, 0, 3]), np.array([1, 0, 3])) == True
    assert sample_obs_from_setting_numba(np.array([0, 0, 0]), np.array([1, 2, 3])) == True
    assert sample_obs_from_setting_numba(np.array([0, 2, 0]), np.array([1, 2, 3])) == True
    assert sample_obs_from_setting_standalone([1, 0, 3], [1, 0, 3]) == True
    assert sample_obs_from_setting_standalone([0, 0, 0], [1, 2, 3]) == True
    assert sample_obs_from_setting_standalone([0, 2, 0], [1, 2, 3]) == True

def test_sample_obs_from_setting_mismatch():
    assert sample_obs_from_setting([1, 2, 3], [1, 0, 3]) == False
    assert sample_obs_from_setting([1, 1, 1], [1, 2, 1]) == False
    assert sample_obs_from_setting([0, 1, 2], [0, 0, 1]) == False
    assert sample_obs_from_setting_numba(np.array([1, 2, 3]), np.array([1, 0, 3])) == False
    assert sample_obs_from_setting_numba(np.array([1, 1, 1]), np.array([1, 2, 1])) == False
    assert sample_obs_from_setting_numba(np.array([0, 1, 2]), np.array([0, 0, 1])) == False
    assert sample_obs_from_setting_standalone([1, 2, 3], [1, 0, 3]) == False
    assert sample_obs_from_setting_standalone([1, 1, 1], [1, 2, 1]) == False
    assert sample_obs_from_setting_standalone([0, 1, 2], [0, 0, 1]) == False

def test_sample_obs_from_setting_all_identity():
    assert sample_obs_from_setting([0, 0, 0], [0, 0, 0]) == True
    assert sample_obs_from_setting_numba(np.array([0, 0, 0]), np.array([0, 0, 0])) == True
    assert sample_obs_from_setting_standalone([0, 0, 0], [0, 0, 0]) == True

def test_sample_obs_from_setting_edge_cases():
    assert sample_obs_from_setting([0], [1]) == True
    assert sample_obs_from_setting([2], [2]) == True
    assert sample_obs_from_setting([3], [1]) == False
    assert sample_obs_from_setting_numba(np.array([0]), np.array([1])) == True
    assert sample_obs_from_setting_numba(np.array([2]), np.array([2])) == True
    assert sample_obs_from_setting_numba(np.array([3]), np.array([1])) == False
    assert sample_obs_from_setting_standalone([0], [1]) == True
    assert sample_obs_from_setting_standalone([2], [2]) == True
    assert sample_obs_from_setting_standalone([3], [1]) == False

def test_sample_obs_from_setting_invalid_type():
    with pytest.raises(TypeError):
        sample_obs_from_setting_standalone("123", [1, 2, 3])
    with pytest.raises(TypeError):
        sample_obs_from_setting_standalone([1, 2, 3], "XYZ")

def test_sample_obs_from_setting_invalid_length():
    with pytest.raises(ValueError):
        sample_obs_from_setting_standalone([1, 2], [1, 2, 3])

def test_sample_obs_from_setting_invalid_elements():
    with pytest.raises(ValueError):
        sample_obs_from_setting_standalone([1, 2, 5], [1, 2, 3])
    with pytest.raises(ValueError):
        sample_obs_from_setting_standalone([1, -1, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        sample_obs_from_setting_standalone([1, 2, 3], [1, 2, 'Z'])