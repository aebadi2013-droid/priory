import pytest, sys
import numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.guarantees import get_epsilon_Bernstein_scalar_tighter_no_restricted_validity

@pytest.fixture(params=[0, 1, -0.01, 1.01])
def invalid_delta(request):
    return request.param

def test_valid_input_no_repeated_settings():
    obs = np.array([[1,1], [1,0], [1,3]])
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([1, 2, 1])
    settings_dict = {'XX': 1, 'XZ': 1}
    delta = 0.05
    
    epsilon = get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta, N_hits, w, 
                                                                          settings_dict, obs)
    assert np.isfinite(epsilon)
    assert epsilon > 0
    
def test_valid_input_repeated_settings():
    obs = np.array([[1,1], [1,0], [1,3]])
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([1, 3, 2])
    settings_dict = {'XX': 1, 'XZ': 2}
    delta = 0.05
    
    epsilon = get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta, N_hits, w, 
                                                                          settings_dict, obs)
    assert np.isfinite(epsilon)
    assert epsilon > 0

def test_all_zero_hits():
    obs = np.array([[1,1], [1,0], [1,3]])
    settings_dict = {}
    delta_input = 0.05
    N_hits_input = np.array([0, 0, 0])
    w_input = np.array([1.0, 2.0, 3.0])
    
    epsilon = get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta_input, N_hits_input, w_input, 
                                                                          settings_dict, obs)
    assert epsilon == np.sum(np.abs(w_input))
    
def test_some_zero_hits():
    obs = np.array([[2,0], [2,1], [3,1], [1,1], [1,3]])
    settings_dict = {'XX': 10, 'XZ': 20}
    delta_input = 0.05
    N_hits_input = np.array([0, 0, 0, 10, 20])
    w_input = np.array([1.0, 2.0, 5.0, 100.0, 2000.0])
    
    epsilon = get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta_input, N_hits_input, w_input,
                                                                          settings_dict, obs)
    eps_sys = np.sum(np.abs(w_input[:3]))
    eps_stat = get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta_input, N_hits_input[3:], w_input[3:],
                                                                           settings_dict, obs[3:])
    assert epsilon == eps_sys + eps_stat
    
def test_split():
    obs = np.array([[2,0], [2,1], [3,1], [1,1], [1,3]])
    settings_dict = {'XX': 10, 'XZ': 20}
    delta_input = 0.05
    N_hits_input = np.array([0, 0, 0, 10, 20])
    w_input = np.array([1.0, 2.0, 5.0, 100.0, 2000.0])
    
    eps_stat, eps_sys = get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta_input, N_hits_input, w_input, 
                                                                                    settings_dict, obs, split=True)
    eps_sys_check = np.sum(np.abs(w_input[:3]))
    eps_stat_check = get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta_input, N_hits_input[3:], w_input[3:],
                                                                                 settings_dict, obs[3:])
    assert eps_stat == eps_stat_check
    assert eps_sys == eps_sys_check

def test_very_small_delta(capsys):
    obs = np.array([[1,1], [1,0], [1,3]])
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([1, 3, 2])
    settings_dict = {'XX': 1, 'XZ': 2}
    delta = 1e-10
    
    epsilon = get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta, N_hits, w, 
                                                                          settings_dict, obs)
    assert np.isfinite(epsilon)

def test_invalid_deltas_raise_error(invalid_delta):
    obs = np.array([[1,1], [1,0], [1,3]])
    w = np.array([1.0, 2.0, 3.0])
    N_hits = np.array([1, 3, 2])
    settings_dict = {'XX': 1, 'XZ': 2}
    
    with pytest.raises(ValueError, match="delta must be in the interval \\(0,1\\)"):
        get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(invalid_delta, N_hits, w, 
                                                                    settings_dict, obs)