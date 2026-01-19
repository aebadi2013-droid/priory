import pytest, sys
import numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.guarantees import (
    get_epsilon_Chebyshev_scalar, get_epsilon_Chebyshev_scalar_numba)

@pytest.fixture(params=[0, 1, -0.01, 1.01])
def invalid_delta(request):
    return request.param

def test_valid_input():
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([10, 20, 30])
    delta = 0.05
    
    epsilon_1 = get_epsilon_Chebyshev_scalar(delta, N_hits, w)
    epsilon_2, _ = get_epsilon_Chebyshev_scalar_numba(delta, N_hits, w)
    assert np.isfinite(epsilon_1)
    assert np.isfinite(epsilon_2)
    assert epsilon_1 > 0
    assert epsilon_2 > 0

def test_all_zero_hits():
    delta_input = 0.05
    N_hits_input = np.array([0, 0, 0])
    w_input = np.array([1.0, 2.0, 3.0])
    
    epsilon_1 = get_epsilon_Chebyshev_scalar(delta_input, N_hits_input, w_input)
    epsilon_2, _ = get_epsilon_Chebyshev_scalar_numba(delta_input, N_hits_input, w_input)
    assert epsilon_1 == np.sum(np.abs(w_input))
    assert epsilon_2 == np.sum(np.abs(w_input))
    
def test_some_zero_hits():
    delta_input = 0.05
    N_hits_input = np.array([0, 0, 0, 10, 20])
    w_input = np.array([1.0, 2.0, 5.0, 100.0, 2000.0])
    
    epsilon_1 = get_epsilon_Chebyshev_scalar(delta_input, N_hits_input, w_input)
    epsilon_2, _ = get_epsilon_Chebyshev_scalar_numba(delta_input, N_hits_input, w_input)
    eps_sys = np.sum(np.abs(w_input[:3]))
    eps_stat = get_epsilon_Chebyshev_scalar(delta_input, N_hits_input[3:], w_input[3:])
    assert epsilon_1 == eps_sys + eps_stat
    assert epsilon_2 == eps_sys + eps_stat
    
def test_split():
    delta_input = 0.05
    N_hits_input = np.array([0, 0, 0, 10, 20])
    w_input = np.array([1.0, 2.0, 5.0, 100.0, 2000.0])
    
    eps_stat_1, eps_sys_1 = get_epsilon_Chebyshev_scalar(delta_input, N_hits_input, w_input, split=True)
    eps_stat_2, eps_sys_2 = get_epsilon_Chebyshev_scalar_numba(delta_input, N_hits_input, w_input, split=True)
    eps_sys_check = np.sum(np.abs(w_input[:3]))
    eps_stat_check = get_epsilon_Chebyshev_scalar(delta_input, N_hits_input[3:], w_input[3:])
    assert eps_stat_1 == eps_stat_check
    assert eps_sys_1 == eps_sys_check
    assert eps_stat_2 == eps_stat_check
    assert eps_sys_2 == eps_sys_check

def test_very_small_delta():
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([10, 20, 30])
    delta = 1e-10
    
    epsilon_1 = get_epsilon_Chebyshev_scalar(delta, N_hits, w)
    epsilon_2, _ = get_epsilon_Chebyshev_scalar_numba(delta, N_hits, w)
    assert np.isfinite(epsilon_1)
    assert np.isfinite(epsilon_2)

def test_invalid_deltas_raise_error(invalid_delta):
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([10, 20, 30])
    
    with pytest.raises(ValueError, match="delta must be in the interval \\(0,1\\)"):
        get_epsilon_Chebyshev_scalar(invalid_delta, N_hits, w)
