import pytest, sys
import numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import hit_by_batch, hit_by_batch_standalone, hit_by_batch_numba

def test_hit_by_batch_valid():
    observables = np.array([
        [0, 2, 0],
        [1, 2, 3],
        [3, 3, 3]
    ])
    setting = np.array([1, 2, 3])
    result_1 = hit_by_batch(observables, setting)
    result_2 = hit_by_batch_numba(observables, setting)
    result_3 = hit_by_batch_standalone(observables, setting)
    expected = [True, True, False]
    assert result_1.tolist() == expected
    assert result_2.tolist() == expected
    assert result_3.tolist() == expected
    
def test_hit_by_batch_commutation():
    O_batch = np.array([
        [1, 2, 3],
        [1, 0, 3],
        [0, 0, 0]
    ])
    P = np.array([1, 2, 3])
    result_1 = hit_by_batch(O_batch, P)
    result_2 = hit_by_batch_numba(O_batch, P)
    result_3 = hit_by_batch_standalone(O_batch, P)
    expected = np.array([True, True, True])
    np.testing.assert_array_equal(result_1, expected)
    np.testing.assert_array_equal(result_2, expected)
    np.testing.assert_array_equal(result_3, expected)

def test_hit_by_batch_invalid_type():
    with pytest.raises(TypeError):
        hit_by_batch_standalone("123", np.array([1, 2, 3]))
    with pytest.raises(TypeError):
        hit_by_batch_standalone(np.array([[1, 2, 3]]), "XYZ")

def test_hit_by_batch_invalid_shape():
    with pytest.raises(ValueError):
        hit_by_batch_standalone(np.array([[1, 2]]), np.array([1, 2, 3]))  # mismatched qubit counts
    with pytest.raises(ValueError):
        hit_by_batch_standalone(np.array([1, 2, 3]), np.array([1, 2, 3]))  # O_batch not 2D

def test_hit_by_batch_invalid_values():
    with pytest.raises(ValueError):
        hit_by_batch_standalone(np.array([[1, 2, 5]]), np.array([1, 2, 3]))  # 5 is out of range
    with pytest.raises(ValueError):
        hit_by_batch_standalone(np.array([[1, -1, 3]]), np.array([1, 2, 3]))  # -1 is invalid
    with pytest.raises(ValueError):
        hit_by_batch_standalone(np.array([[1, 2, 3]]), np.array([1, 2, 'Z']))  # 'Z' is invalid