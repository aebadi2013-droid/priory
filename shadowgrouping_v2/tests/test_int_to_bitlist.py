import pytest, sys

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.helper_functions import int_to_bitlist

def test_basic_binary():
    assert int_to_bitlist(0) == [0]
    assert int_to_bitlist(1) == [1]
    assert int_to_bitlist(2) == [1, 0]
    assert int_to_bitlist(5) == [1, 0, 1]
    assert int_to_bitlist(13) == [1, 1, 0, 1]

def test_with_width_padding():
    assert int_to_bitlist(5, width=4) == [0, 1, 0, 1]  # binary 0101
    assert int_to_bitlist(5, width=6) == [0, 0, 0, 1, 0, 1]  # binary 000101
    assert int_to_bitlist(0, width=4) == [0, 0, 0, 0]

def test_width_smaller_than_bit_length():
    # Should truncate higher bits if width < actual bit length
    assert int_to_bitlist(13, width=3) == [1, 0, 1]  # last 3 bits of 1101

def test_zero_with_no_width():
    assert int_to_bitlist(0) == [0]  # special handling

def test_large_integer():
    n = 255
    expected = [1] * 8  # 255 = 0b11111111
    assert int_to_bitlist(n, width=8) == expected
