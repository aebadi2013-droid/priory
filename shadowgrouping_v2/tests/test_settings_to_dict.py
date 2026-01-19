import numpy as np, sys, pytest

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import setting_to_str, settings_to_dict

def test_settings_to_dict_nonadaptive_basic():
    settings = np.array([[0, 1, 2], [0, 1, 2], [1, 0, 3]])
    settings_dict = {}
    settings_buffer = {}

    settings_to_dict(settings, settings_dict, settings_buffer)

    assert settings_dict == {"IXY": 2, "XIZ": 1}
    assert settings_buffer == {"IXY": 2, "XIZ": 1}

def test_settings_to_dict_with_existing_counts():
    settings = np.array([[1, 0, 3]])
    settings_dict = {"XIZ": 1}
    settings_buffer = {"XIZ": 2}

    settings_to_dict(settings, settings_dict, settings_buffer)

    assert settings_dict["XIZ"] == 2
    assert settings_buffer["XIZ"] == 3

def test_settings_to_dict_adaptive_updates_order():
    settings = np.array([[0, 1, 2], [1, 1, 1], [0, 1, 2]])
    settings_dict = {}
    settings_buffer = {}
    order = {}

    settings_to_dict(settings, settings_dict, settings_buffer, is_adaptive=True, order=order)

    assert "IXY" in order
    assert "XXX" in order
    assert set(order["IXY"]) == {0, 2}
    assert set(order["XXX"]) == {1}
    assert settings_dict == {"IXY": 2, "XXX": 1}

def test_settings_to_dict_adaptive_missing_order_raises():
    settings = np.array([[0, 1, 2]])
    settings_dict = {}
    settings_buffer = {}

    with pytest.raises(ValueError, match="order must be provided"):
        settings_to_dict(settings, settings_dict, settings_buffer, is_adaptive=True, order=None)

def test_settings_to_dict_empty_input():
    settings = np.empty((0, 3), dtype=int)
    settings_dict = {}
    settings_buffer = {}

    settings_to_dict(settings, settings_dict, settings_buffer)

    assert settings_dict == {}
    assert settings_buffer == {}