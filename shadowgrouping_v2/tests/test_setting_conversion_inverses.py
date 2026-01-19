import sys

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import (setting_to_obs_form, setting_to_str,
                                             setting_to_obs_form_standalone, setting_to_str_standalone)

def test_string_to_list_inverse():
    valid_strings = ["XYZ", "IXXIZY", "III", "", "XYZZYXXII"]
    for s in valid_strings:
        assert setting_to_str(setting_to_obs_form(s)) == s
        assert setting_to_str_standalone(setting_to_obs_form_standalone(s)) == s

def test_list_to_string_inverse():
    valid_lists = [[1, 2, 3], [0, 1, 1, 0, 3, 2], [0, 0, 0], [], [1, 2, 3, 3, 2, 1, 1, 0, 0]]
    for lst in valid_lists:
        assert setting_to_obs_form(setting_to_str(lst)) == lst
        assert setting_to_obs_form_standalone(setting_to_str_standalone(lst)) == lst
