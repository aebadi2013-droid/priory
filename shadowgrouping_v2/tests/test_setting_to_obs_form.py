import pytest, sys

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import setting_to_obs_form, setting_to_obs_form_standalone

def test_setting_to_obs_form_valid():
    assert setting_to_obs_form("XYZ") == [1, 2, 3]
    assert setting_to_obs_form("IXXIZY") == [0, 1, 1, 0, 3, 2]
    assert setting_to_obs_form("IIII") == [0, 0, 0, 0]
    assert setting_to_obs_form("") == []
    assert setting_to_obs_form_standalone("XYZ") == [1, 2, 3]
    assert setting_to_obs_form_standalone("IXXIZY") == [0, 1, 1, 0, 3, 2]
    assert setting_to_obs_form_standalone("IIII") == [0, 0, 0, 0]
    assert setting_to_obs_form_standalone("") == []

def test_setting_to_obs_form_invalid_characters():
    with pytest.raises(ValueError, match=r"Invalid character 'A' at position 0"):
        setting_to_obs_form_standalone("AXYZ")
    with pytest.raises(ValueError, match=r"Invalid character 'T' at position 3"):
        setting_to_obs_form_standalone("IXYT")
    with pytest.raises(ValueError, match=r"Invalid character '\*' at position 1"):
        setting_to_obs_form_standalone("I*YZ")

def test_setting_to_obs_form_invalid_type():
    with pytest.raises(TypeError):
        setting_to_obs_form_standalone(123)
    with pytest.raises(TypeError):
        setting_to_obs_form_standalone(['X', 'Y', 'Z'])
    with pytest.raises(TypeError):
        setting_to_obs_form_standalone(None)