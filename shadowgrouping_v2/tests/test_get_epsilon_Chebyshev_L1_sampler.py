import pytest, sys
import numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.guarantees import get_epsilon_Chebyshev_L1_sampler

@pytest.fixture(params=[0, 1, -0.01, 1.01])
def invalid_delta(request):
    return request.param

def test_valid_input():
    delta_input = 0.05
    shots_input = 3
    w_input = np.array([1.0, 2.0, 3.0])
    
    epsilon = get_epsilon_Chebyshev_L1_sampler(delta_input, shots_input, w_input)
    assert np.isfinite(epsilon)
    assert epsilon > 0

def test_zero_hits_returns_1_norm_coefficients():
    delta_input = 0.05
    shots_input = 0
    w_input = np.array([1.0, 2.0, 3.0])
    
    epsilon = get_epsilon_Chebyshev_L1_sampler(delta_input, shots_input, w_input)
    assert epsilon == np.sum(np.abs(w_input))

def test_very_small_delta(capsys):
    delta_input = 1e-10
    shots_input = 3
    w_input = np.array([1.0, 2.0, 3.0])
    
    epsilon = get_epsilon_Chebyshev_L1_sampler(delta_input, shots_input, w_input)
    assert np.isfinite(epsilon)
    
def test_very_small_shots(capsys):
    delta_input = 0.05
    shots_input = 1
    w_input = np.array([1.0, 2.0, 3.0])
    
    epsilon = get_epsilon_Chebyshev_L1_sampler(delta_input, shots_input, w_input)
    assert np.isfinite(epsilon)
    assert epsilon == np.sum(np.abs(w_input))

def test_invalid_deltas_raise_error(invalid_delta):
    shots_input = 3
    w_input = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="delta must be in the interval \\(0,1\\)"):
        get_epsilon_Chebyshev_L1_sampler(invalid_delta, shots_input, w_input)