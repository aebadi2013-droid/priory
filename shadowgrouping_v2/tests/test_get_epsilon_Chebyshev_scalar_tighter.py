import pytest, sys
import numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.guarantees import (
    get_epsilon_Chebyshev_scalar_tighter, get_epsilon_Chebyshev_scalar_tighter_numba)
from shadowgrouping.helper_functions import prepare_settings_for_numba

@pytest.fixture(params=[0, 1, -0.01, 1.01])
def invalid_delta(request):
    return request.param

def test_valid_input_no_repeated_settings():
    obs = np.array([[1,1], [1,0], [1,3]])
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([1, 2, 1])
    settings_dict = {'XX': 1, 'XZ': 1}
    delta = 0.05
    settings_int, settings_reps = prepare_settings_for_numba(settings_dict)
    
    epsilon_1 = get_epsilon_Chebyshev_scalar_tighter(delta, N_hits, w, settings_dict, obs)
    epsilon_2, _ = get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits, w, settings_int, 
                                                              settings_reps, obs)
    assert np.isfinite(epsilon_1)
    assert np.isfinite(epsilon_2)
    assert epsilon_1 > 0
    assert epsilon_2 > 0
    
def test_valid_input_repeated_settings():
    obs = np.array([[1,1], [1,0], [1,3]])
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([1, 3, 2])
    settings_dict = {'XX': 1, 'XZ': 2}
    delta = 0.05
    settings_int, settings_reps = prepare_settings_for_numba(settings_dict)
    
    epsilon_1 = get_epsilon_Chebyshev_scalar_tighter(delta, N_hits, w, settings_dict, obs)
    epsilon_2, _ = get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits, w, settings_int, 
                                                              settings_reps, obs)
    assert np.isfinite(epsilon_1)
    assert np.isfinite(epsilon_2)
    assert epsilon_1 > 0
    assert epsilon_2 > 0

def test_all_zero_hits():
    obs = np.array([[1,1], [1,0], [1,3]])
    settings_dict = {}
    delta = 0.05
    N_hits = np.array([0, 0, 0])
    w = np.array([1.0, 2.0, 3.0])
    
    epsilon = get_epsilon_Chebyshev_scalar_tighter(delta, N_hits, w, settings_dict, obs)

    assert epsilon == np.sum(np.abs(w))
    
def test_some_zero_hits():
    obs = np.array([[2,0], [2,1], [3,1], [1,1], [1,3]])
    settings_dict = {'XX': 10, 'XZ': 20}
    delta = 0.05
    N_hits = np.array([0, 0, 0, 10, 20])
    w = np.array([1.0, 2.0, 5.0, 100.0, 2000.0])
    settings_int, settings_reps = prepare_settings_for_numba(settings_dict)
    
    epsilon_1 = get_epsilon_Chebyshev_scalar_tighter(delta, N_hits, w, settings_dict, obs)
    epsilon_2, _ = get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits, w, settings_int, 
                                                              settings_reps, obs)
    eps_sys = np.sum(np.abs(w[:3]))
    eps_stat_1 = get_epsilon_Chebyshev_scalar_tighter(delta, N_hits[3:], w[3:],
                                                      settings_dict, obs[3:])
    eps_stat_2, _ = get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits[3:], w[3:],
                                                               settings_int, settings_reps, obs[3:])
    assert epsilon_1 == eps_sys + eps_stat_1
    assert epsilon_2 == eps_sys + eps_stat_2
    
def test_split():
    obs = np.array([[2,0], [2,1], [3,1], [1,1], [1,3]])
    settings_dict = {'XX': 10, 'XZ': 20}
    delta = 0.05
    N_hits = np.array([0, 0, 0, 10, 20])
    w = np.array([1.0, 2.0, 5.0, 100.0, 2000.0])
    settings_int, settings_reps = prepare_settings_for_numba(settings_dict)
    
    eps_stat_1, eps_sys_1 = get_epsilon_Chebyshev_scalar_tighter(delta, N_hits, w, settings_dict, 
                                                                 obs, split=True)
    eps_stat_2, eps_sys_2 = get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits, w, settings_int, 
                                                                       settings_reps, obs, split=True)
    eps_sys_check = np.sum(np.abs(w[:3]))
    eps_stat_check_1 = get_epsilon_Chebyshev_scalar_tighter(delta, N_hits[3:], w[3:],
                                                            settings_dict, obs[3:])
    eps_stat_check_2, _ = get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits[3:], w[3:],
                                                                     settings_int, settings_reps, obs[3:])
    assert eps_stat_1 == eps_stat_check_1
    assert eps_sys_1 == eps_sys_check
    assert eps_stat_2 == eps_stat_check_2
    assert eps_sys_2 == eps_sys_check

def test_very_small_delta(capsys):
    obs = np.array([[1,1], [1,0], [1,3]])
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([1, 3, 2])
    settings_dict = {'XX': 1, 'XZ': 2}
    delta = 1e-15
    settings_int, settings_reps = prepare_settings_for_numba(settings_dict)
    
    epsilon_1 = get_epsilon_Chebyshev_scalar_tighter(delta, N_hits, w, settings_dict, obs)
    epsilon_2, _ = get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits, w, settings_int, 
                                                              settings_reps, obs)
    assert np.isfinite(epsilon_1)
    assert np.isfinite(epsilon_2)

def test_invalid_deltas_raise_error(invalid_delta):
    obs = np.array([[1,1], [1,0], [1,3]])
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([1, 3, 2])
    settings_dict = {'XX': 1, 'XZ': 2}
    
    with pytest.raises(ValueError, match="delta must be in the interval \\(0,1\\)"):
        get_epsilon_Chebyshev_scalar_tighter(invalid_delta, N_hits, w, settings_dict, obs)
