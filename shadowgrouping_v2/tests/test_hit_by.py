import pytest, sys, numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import hit_by, hit_by_standalone, hit_by_numba

def test_hit_by_identical():
    assert hit_by(np.array([1, 2, 3]), np.array([1, 2, 3])) == True
    assert hit_by_numba(np.array([1, 2, 3]), np.array([1, 2, 3])) == True
    assert hit_by_standalone(np.array([1, 2, 3]), np.array([1, 2, 3])) == True

def test_hit_by_with_identities():
    assert hit_by(np.array([1, 2, 3]), np.array([0, 0, 0])) == True
    assert hit_by(np.array([0, 2, 0]), np.array([1, 2, 3])) == True
    assert hit_by_numba(np.array([1, 2, 3]), np.array([0, 0, 0])) == True
    assert hit_by_numba(np.array([0, 2, 0]), np.array([1, 2, 3])) == True
    assert hit_by_standalone(np.array([1, 2, 3]), np.array([0, 0, 0])) == True
    assert hit_by_standalone(np.array([0, 2, 0]), np.array([1, 2, 3])) == True

def test_hit_by_incompatible():
    assert hit_by(np.array([1, 2, 3]), np.array([3, 1, 2])) == False
    assert hit_by(np.array([1, 1, 1]), np.array([2, 2, 2])) == False
    assert hit_by_numba(np.array([1, 2, 3]), np.array([3, 1, 2])) == False
    assert hit_by_numba(np.array([1, 1, 1]), np.array([2, 2, 2])) == False
    assert hit_by_standalone(np.array([1, 2, 3]), np.array([3, 1, 2])) == False
    assert hit_by_standalone(np.array([1, 1, 1]), np.array([2, 2, 2])) == False

def test_hit_by_mixed():
    assert hit_by(np.array([1, 0, 3]), np.array([1, 2, 3])) == True   # Middle qubit is identity in O
    assert hit_by(np.array([1, 2, 3]), np.array([1, 0, 3])) == True   # Middle qubit is identity in P
    assert hit_by_numba(np.array([1, 0, 3]), np.array([1, 2, 3])) == True   # Middle qubit is identity in O
    assert hit_by_numba(np.array([1, 2, 3]), np.array([1, 0, 3])) == True   # Middle qubit is identity in P
    assert hit_by_standalone(np.array([1, 0, 3]), np.array([1, 2, 3])) == True   # Middle qubit is identity in O
    assert hit_by_standalone(np.array([1, 2, 3]), np.array([1, 0, 3])) == True   # Middle qubit is identity in P

def test_hit_by_edge_cases():
    assert hit_by([0], [0]) == True
    
    assert hit_by_standalone([0], [0]) == True
    
def test_invalid_type_inputs():
    with pytest.raises(TypeError):
        hit_by_standalone("123", [1, 2, 3])
    with pytest.raises(TypeError):
        hit_by_standalone([1, 2, 3], "XYZ")

def test_mismatched_length():
    with pytest.raises(ValueError):
        hit_by_standalone([1, 2, 3], [1, 2])

def test_invalid_elements():
    with pytest.raises(ValueError):
        hit_by_standalone([1, 2, 5], [1, 2, 3])  # 5 is invalid
    with pytest.raises(ValueError):
        hit_by_standalone([1, -1, 3], [1, 2, 3])  # -1 is invalid
    with pytest.raises(ValueError):
        hit_by_standalone([1, 2, 3], [1, 2, 'Z'])  # 'Z' is invalid
