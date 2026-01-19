import pytest, sys
import numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import hit_by_batch_batch, hit_by_batch_batch_numba, hit_by_batch_batch_standalone 

def test_hit_by_batch_batch_valid_same_number():
    O_batch = np.array([
        [1, 2, 3],
        [0, 0, 0],
        [1, 0, 2]
    ])
    P_batch = np.array([
        [1, 2, 3],
        [1, 0, 0],
        [1, 2, 0]
    ])
    result_1 = hit_by_batch_batch(O_batch, P_batch)
    result_2 = hit_by_batch_batch_numba(O_batch, P_batch)
    result_3 = hit_by_batch_batch_standalone(O_batch, P_batch)
    expected = np.array([
        [True, True, True],
        [True, True, True],
        [False, True, True]
    ])
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)

def test_hit_by_batch_batch_valid_different_number():
    O_batch = np.array([
        [1, 2, 3],
        [0, 0, 0],
        [1, 0, 2]
    ])
    P_batch = np.array([
        [1, 0, 2],
        [3, 0, 0]
    ])
    result_1 = hit_by_batch_batch(O_batch, P_batch)
    result_2 = hit_by_batch_batch_numba(O_batch, P_batch)
    result_3 = hit_by_batch_batch_standalone(O_batch, P_batch)
    expected = np.array([
        [False, False],
        [True, True],
        [True, False]
    ])
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)

def test_invalid_types():
    with pytest.raises(TypeError):
        hit_by_batch_batch_standalone("123", [[1, 2, 3]])
    with pytest.raises(TypeError):
        hit_by_batch_batch_standalone([[1, 2, 3]], "XYZ")

def test_invalid_shapes():
    with pytest.raises(ValueError):
        hit_by_batch_batch_standalone([[1, 2]], [[1, 2, 3]])
    with pytest.raises(ValueError):
        hit_by_batch_batch_standalone([1, 2, 3], [[1, 2, 3]])

def test_invalid_values():
    with pytest.raises(ValueError):
        hit_by_batch_batch_standalone([[1, 2, 5]], [[1, 2, 3]])  # 5 is out of range
    with pytest.raises(ValueError):
        hit_by_batch_batch_standalone([[1, -1, 3]], [[1, 2, 3]])  # -1 is invalid
    with pytest.raises(ValueError):
        hit_by_batch_batch_standalone([[1, 2, 3]], [[1, 2, 'Z']])  # 'Z' is invalid

def test_output_shape_and_type():
    O_batch = np.array([[1, 2, 3], [3, 1, 1]])
    P_batch = np.array([[1, 2, 3], [1, 0, 3]])
    result_1 = hit_by_batch_batch(O_batch, P_batch)
    result_2 = hit_by_batch_batch_numba(O_batch, P_batch)
    result_3 = hit_by_batch_batch_standalone(O_batch, P_batch)
    assert result_1.shape == (2, 2)
    assert result_1.dtype == bool
    assert result_2.shape == (2, 2)
    assert result_2.dtype == bool
    assert result_3.shape == (2, 2)
    assert result_3.dtype == bool