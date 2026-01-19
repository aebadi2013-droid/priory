import numpy as np, sys, pytest

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)
# defining path where Hamiltonians are stored
folder_Hamiltonians = SG_package_path + "Hamiltonians\\"


from shadowgrouping.truncations import truncate_binary_search
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler
from shadowgrouping.measurement_schemes import Measurement_scheme, Shadow_Grouping
from shadowgrouping.hamiltonian import load_pauli_list
from shadowgrouping.weight_functions import Bernstein_bound
from shadowgrouping.guarantees import (get_epsilon_Hoeffding_scalar, get_epsilon_Chebyshev_scalar,
                                       get_epsilon_Bernstein_scalar, get_epsilon_Bernstein,
                                       get_epsilon_Bernstein_no_restricted_validity)
from shadowgrouping.guarantees import (get_epsilon_Hoeffding_scalar_tighter, get_epsilon_Chebyshev_scalar_tighter,
                                       get_epsilon_Bernstein_scalar_tighter, get_epsilon_Bernstein_tighter,
                                       get_epsilon_Bernstein_tighter_no_restricted_validity)

@pytest.fixture(params=[[get_epsilon_Hoeffding_scalar, False],
                        [get_epsilon_Chebyshev_scalar, False],
                        [get_epsilon_Bernstein_scalar, False],
                        [get_epsilon_Bernstein, False],
                        [get_epsilon_Bernstein_no_restricted_validity, False],
                        [get_epsilon_Hoeffding_scalar_tighter, True],
                        [get_epsilon_Chebyshev_scalar_tighter, True],
                        [get_epsilon_Bernstein_scalar_tighter, True],
                        [get_epsilon_Bernstein_tighter, True],
                        [get_epsilon_Bernstein_tighter_no_restricted_validity, True]])
def get_epsilon_functions_picker(request):
    return request.param

def test_truncate_binary_search_never_worse_all(get_epsilon_functions_picker):
    w = np.array([10.0, 1.6, 0.7, 0.005, 0.002])
    obs = np.array([
        [3, 3, 3, 3],
        [1, 1, 0, 0],
        [1, 1, 2, 2],
        [1, 2, 3, 3],
        [3, 1, 3, 1]
    ])
    N_hits = np.array([500, 70, 70, 5, 2])
    settings_dict = {'ZZZZ': 500, 'XXYY': 70, 'XYZZ': 5, 'ZXZX': 2}
    scheme = Measurement_scheme(obs, w, epsilon=0.1)
    scheme.N_hits = N_hits
    state = np.random.rand(16,)
    state = 1/np.linalg.norm(state)*state
    state = StateSampler(state)
    estimator = Energy_estimator(scheme, state)
    estimator.settings_dict = settings_dict
    
    delta = 0.05 
    
    if get_epsilon_functions_picker[1]:
        baseline_error = get_epsilon_functions_picker[0](delta, estimator.measurement_scheme.N_hits,
                                                         estimator.measurement_scheme.w,
                                                         estimator.settings_dict,
                                                         estimator.measurement_scheme.obs)
    else:
        baseline_error = get_epsilon_functions_picker[0](delta, estimator.measurement_scheme.N_hits,
                                                         estimator.measurement_scheme.w)
    
    eps_stat, eps_syst, num_trun_obs = truncate_binary_search(
        estimator, delta, get_epsilon_functions_picker[0], 
        tightened=get_epsilon_functions_picker[1], in_place=True, verbose=False
    )

    assert num_trun_obs >= 0
    assert num_trun_obs <= obs.shape[0]
    assert eps_syst + eps_stat <= baseline_error
    assert estimator.measurement_scheme.num_obs <= 5
    assert estimator.measurement_scheme.num_obs >= 0

def test_truncate_binary_search_expected_result_loose():
    molecule_name = "LiH" # choose one out of the molecules ['H2', 'H2_6-31g', 'LiH', 'BeH2', 'H2O', 'NH3']
    mapping_name = "JW" # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g" # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    observables, w, offset, E_GS, state = load_pauli_list(folder_Hamiltonians,molecule_name,basis_set,mapping_name,diagonalize=False)
    state = np.random.rand(2**12,)
    state = 1/np.linalg.norm(state)*state
    alpha = np.max(np.abs(w))/np.min(np.abs(w)) + np.min(np.abs(w))
    method = Shadow_Grouping(observables,w,0.1,Bernstein_bound(alpha=alpha)())
    estimator = Energy_estimator(method,StateSampler(state))
    estimator.reset()
    estimator.propose_next_settings(num_steps=200)
    
    delta = 0.33
    eps_stat, eps_syst, num_trun_obs = truncate_binary_search(
        estimator, delta, get_epsilon_Hoeffding_scalar, 
        tightened=False, in_place=True, verbose=False
    )

    assert num_trun_obs == 360
    assert np.isclose(eps_stat+eps_syst, 3.7585587949843555)
    assert np.isclose(eps_syst, 1.5541734794898352)
    assert estimator.measurement_scheme.num_obs == 630-360
    
def test_truncate_binary_search_expected_result_tight():
    molecule_name = "NH3" # choose one out of the molecules ['H2', 'H2_6-31g', 'LiH', 'BeH2', 'H2O', 'NH3']
    mapping_name = "JW" # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g" # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    observables, w, offset, E_GS, state = load_pauli_list(folder_Hamiltonians,molecule_name,basis_set,mapping_name,diagonalize=False)
    state = np.random.rand(2**16,)
    state = 1/np.linalg.norm(state)*state
    alpha = np.max(np.abs(w))/np.min(np.abs(w)) + np.min(np.abs(w))
    method = Shadow_Grouping(observables,w,0.1,Bernstein_bound(alpha=alpha)())
    estimator = Energy_estimator(method,StateSampler(state))
    estimator.reset()
    estimator.propose_next_settings(num_steps=700)
    
    delta = 0.33
    eps_stat, eps_syst, num_trun_obs = truncate_binary_search(
        estimator, delta, get_epsilon_Hoeffding_scalar_tighter, 
        tightened=True, in_place=True, verbose=False
    )

    assert num_trun_obs == 371
    assert np.isclose(eps_stat+eps_syst, 4.9741657168671045)
    assert np.isclose(eps_syst, 0.002508587610967572)
    assert estimator.measurement_scheme.num_obs == 3056-371

def test_truncate_analytical_nothing_truncated(capfd):
    molecule_name = "H2" # choose one out of the molecules ['H2', 'H2_6-31g', 'LiH', 'BeH2', 'H2O', 'NH3']
    mapping_name = "JW" # choose one out of ["JW","BK","Parity"]
    basis_set = "sto3g" # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
    observables, w, offset, E_GS, state = load_pauli_list(folder_Hamiltonians,molecule_name,basis_set,mapping_name,diagonalize=False)
    state = np.random.rand(2**4,)
    state = 1/np.linalg.norm(state)*state
    alpha = np.max(np.abs(w))/np.min(np.abs(w)) + np.min(np.abs(w))
    method = Shadow_Grouping(observables,w,0.1,Bernstein_bound(alpha=alpha)())
    estimator = Energy_estimator(method,StateSampler(state))
    estimator.reset()
    estimator.propose_next_settings(num_steps=1000)
    
    delta = 0.33
    eps_stat, eps_syst, num_trun_obs = truncate_binary_search(
        estimator, delta, get_epsilon_Hoeffding_scalar, 
        tightened=False, in_place=True, verbose=False
    )

    assert num_trun_obs == 0
    assert eps_syst == 0.0
    assert np.isclose(eps_stat, get_epsilon_Hoeffding_scalar(delta, estimator.measurement_scheme.N_hits, 
                                                             estimator.measurement_scheme.w))
    assert estimator.measurement_scheme.num_obs == 14
    
    out, _ = capfd.readouterr()
    assert "Nothing had to be truncated." in out