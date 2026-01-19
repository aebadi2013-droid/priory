import pytest, sys
import numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.guarantees import get_epsilon_Hoeffding_L1_sampler

@pytest.fixture(params=[0, 1, -0.01, 1.01])
def invalid_delta(request):
    return request.param

# ---------- TESTS ----------

def test_valid_input():
    weights_input = np.array([1.0, 2.0, 3.0])
    delta_input = 0.05
    shots_input = 3
    
    epsilon = get_epsilon_Hoeffding_L1_sampler(delta_input, shots_input, weights_input)
    assert np.isfinite(epsilon)
    assert epsilon > 0

def test_zero_hits_returns_1_norm_coefficients():
    weights_input = np.array([1.0, 2.0, 3.0])
    delta_input = 0.05
    shots_input = 0
    
    epsilon = get_epsilon_Hoeffding_L1_sampler(delta_input, shots_input, weights_input)
    assert epsilon == np.sum(np.abs(weights_input))

def test_very_small_delta(capsys):
    weights_input = np.array([1.0, 2.0, 3.0])
    delta_input = 1e-10
    shots_input = 3
    
    epsilon = get_epsilon_Hoeffding_L1_sampler(delta_input, shots_input, weights_input)
    assert np.isfinite(epsilon)
    
def test_very_small_shots(capsys):
    weights_input = np.array([1.0, 2.0, 3.0])
    delta_input = 1e-10
    shots_input = 1
    
    epsilon = get_epsilon_Hoeffding_L1_sampler(delta_input, shots_input, weights_input)
    assert np.isfinite(epsilon)
    assert epsilon == np.sum(np.abs(weights_input))

def test_invalid_deltas_raise_error(invalid_delta):
    weights_input = np.array([1.0, 2.0, 3.0])
    shots_input = 3
    
    with pytest.raises(ValueError, match="delta must be in the interval \\(0,1\\)"):
        get_epsilon_Hoeffding_L1_sampler(invalid_delta, shots_input, weights_input)