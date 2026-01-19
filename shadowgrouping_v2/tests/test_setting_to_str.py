import pytest, sys

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import setting_to_str, setting_to_str_standalone

def test_setting_to_str_valid():
    assert setting_to_str([1, 2, 3]) == "XYZ"
    assert setting_to_str([0, 1, 1, 0, 3, 2]) == "IXXIZY"
    assert setting_to_str([0, 0, 0]) == "III"
    assert setting_to_str([]) == ""
    assert setting_to_str([3, 2, 1, 0]) == "ZYXI"
    assert setting_to_str_standalone([1, 2, 3]) == "XYZ"
    assert setting_to_str_standalone([0, 1, 1, 0, 3, 2]) == "IXXIZY"
    assert setting_to_str_standalone([0, 0, 0]) == "III"
    assert setting_to_str_standalone([]) == ""
    assert setting_to_str_standalone([3, 2, 1, 0]) == "ZYXI"

def test_setting_to_str_invalid_type():
    with pytest.raises(TypeError):
        setting_to_str_standalone("XYZ")  # Not a list
    with pytest.raises(TypeError):
        setting_to_str_standalone(None)
    with pytest.raises(TypeError):
        setting_to_str_standalone((1, 2, 3))  # Tuple, not list

def test_setting_to_str_invalid_elements():
    with pytest.raises(ValueError, match=r"P\[0\] = '-1' is not an integer in \[0, 3\]"):
        setting_to_str_standalone([-1, 2, 3])
    with pytest.raises(ValueError, match=r"P\[1\] = '4' is not an integer in \[0, 3\]"):
        setting_to_str_standalone([0, 4, 2])
    with pytest.raises(ValueError, match=r"P\[2\] = 'X' is not an integer in \[0, 3\]"):
        setting_to_str_standalone([0, 1, 'X'])
