import numpy as np, sys, pytest

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import index_to_string, index_to_string_standalone

def test_index_to_string_typical_case():
    assert index_to_string([0, 1, 2, 3]) == "IXYZ"
    assert index_to_string_standalone([0, 1, 2, 3]) == "IXYZ"

def test_index_to_string_all_I():
    assert index_to_string([0, 0, 0]) == "III"
    assert index_to_string_standalone([0, 0, 0]) == "III"

def test_index_to_string_all_Z():
    assert index_to_string([3, 3, 3, 3]) == "ZZZZ"
    assert index_to_string_standalone([3, 3, 3, 3]) == "ZZZZ"

def test_index_to_string_mixed():
    assert index_to_string([1, 2, 1, 0]) == "XYXI"
    assert index_to_string_standalone([1, 2, 1, 0]) == "XYXI"

def test_index_to_string_numpy_array_input():
    input_array = np.array([2, 0, 3])
    assert index_to_string(input_array) == "YIZ"
    assert index_to_string_standalone(input_array) == "YIZ"

def test_index_to_string_invalid_negative():
    with pytest.raises(ValueError, match="All elements must be in {0,1,2,3}."):
        index_to_string_standalone([0, -1, 1])

def test_index_to_string_invalid_large():
    with pytest.raises(ValueError, match="All elements must be in {0,1,2,3}."):
        index_to_string_standalone([1, 4, 2])

def test_index_to_string_non_integer():
    with pytest.raises(ValueError, match="All entries must be integers."):
        index_to_string_standalone(["0", "1", "Z"])
