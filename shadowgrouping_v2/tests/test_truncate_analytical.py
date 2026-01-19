import numpy as np, sys, pytest

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.truncations import truncate_analytical
from shadowgrouping.measurement_schemes import Measurement_scheme
from shadowgrouping.guarantees import (N_delta_Hoeffding_scalar, N_delta_Chebyshev_scalar, 
                                       N_delta_Bernstein_scalar, N_delta_Bernstein,
                                       N_delta_Bernstein_no_restricted_validity)
from shadowgrouping.guarantees import (get_epsilon_Hoeffding_scalar, get_epsilon_Chebyshev_scalar,
                                       get_epsilon_Bernstein_scalar, get_epsilon_Bernstein,
                                       get_epsilon_Bernstein_no_restricted_validity)
from shadowgrouping.guarantees import (get_epsilon_Hoeffding_scalar_tighter, get_epsilon_Chebyshev_scalar_tighter,
                                       get_epsilon_Bernstein_scalar_tighter, get_epsilon_Bernstein_tighter,
                                       get_epsilon_Bernstein_tighter_no_restricted_validity)

@pytest.fixture(params=[[get_epsilon_Hoeffding_scalar, N_delta_Hoeffding_scalar],
                        [get_epsilon_Chebyshev_scalar, N_delta_Chebyshev_scalar],
                        [get_epsilon_Bernstein_scalar, N_delta_Bernstein_scalar],
                        [get_epsilon_Bernstein, N_delta_Bernstein],
                        [get_epsilon_Bernstein_no_restricted_validity, N_delta_Bernstein_no_restricted_validity]])
def get_epsilon_and_N_delta_functions_picker(request):
    return request.param

@pytest.fixture(params=[[get_epsilon_Hoeffding_scalar_tighter, N_delta_Hoeffding_scalar],
                        [get_epsilon_Chebyshev_scalar_tighter, N_delta_Chebyshev_scalar],
                        [get_epsilon_Bernstein_scalar_tighter, N_delta_Bernstein_scalar],
                        [get_epsilon_Bernstein_tighter, N_delta_Bernstein],
                        [get_epsilon_Bernstein_tighter_no_restricted_validity, N_delta_Bernstein_no_restricted_validity]])
def get_epsilon_tighter_and_N_delta_functions_picker(request):
    return request.param

def test_truncate_analytical_some_truncated(get_epsilon_and_N_delta_functions_picker):
    w = np.array([0.5, 0.3, 0.2, 0.005, 0.002])
    obs = np.array([
        [0, 1, 2, 3],
        [1, 1, 0, 0],
        [3, 0, 0, 0],
        [1, 2, 3, 3],
        [3, 1, 3, 1]
    ])
    N_hits = np.array([200, 100, 300, 5, 2])
    scheme = Measurement_scheme(obs, w, epsilon=0.1)
    scheme.N_hits = N_hits
    
    delta = 0.05 # Yields 7.37, 20.0, 14.8, 79.6, 62.5 for
                 # N_delta_Hoeffding_scalar, N_delta_Chebyshev_scalar,
                 # N_delta_Bernstein_scalar, N_delta_Bernstein,
                 # N_delta_Bernstein_no_restricted_validity
    eps_stat, eps_syst, num_trun_obs = truncate_analytical(
        scheme, delta, get_epsilon_and_N_delta_functions_picker[1], 
        get_epsilon_and_N_delta_functions_picker[0], in_place=True
    )

    assert num_trun_obs == 2
    assert np.isclose(eps_syst, 0.007)
    assert scheme.num_obs == 3
    assert np.all(scheme.N_hits >= 100)
    assert scheme.obs.shape == (3, 4)

def test_truncate_analytical_all_truncated(get_epsilon_and_N_delta_functions_picker, capfd):
    w = np.array([0.1, 0.2])
    obs = np.array([
        [0, 0, 1, 1],
        [2, 2, 3, 3]
    ])
    N_hits = np.array([5, 6])  # all below threshold
    scheme = Measurement_scheme(obs, w, epsilon=0.1)
    scheme.N_hits = N_hits
    
    delta = 0.05 # Yields 7.37, 20.0, 14.8, 79.6, 62.5 for
                 # N_delta_Hoeffding_scalar, N_delta_Chebyshev_scalar,
                 # N_delta_Bernstein_scalar, N_delta_Bernstein,
                 # N_delta_Bernstein_no_restricted_validity
    eps_stat, eps_syst, num_trun_obs = truncate_analytical(
        scheme, delta, get_epsilon_and_N_delta_functions_picker[1], 
        get_epsilon_and_N_delta_functions_picker[0], in_place=True
    )

    assert num_trun_obs == 2
    assert np.isclose(eps_stat, 0.0)
    assert np.isclose(eps_syst, 0.3)
    assert scheme.num_obs == 2  # Unchanged due to early return
    assert np.array_equal(scheme.N_hits, N_hits)
    
    out, _ = capfd.readouterr()
    assert "No observable reached the threshold" in out
    assert "Scheme unaltered." in out

def test_truncate_analytical_nothing_truncated(get_epsilon_and_N_delta_functions_picker, capfd):
    w = np.array([0.4, 0.6])
    obs = np.array([
        [1, 0, 3, 2],
        [3, 2, 1, 0]
    ])
    N_hits = np.array([200, 400])  # all above threshold
    scheme = Measurement_scheme(obs, w, epsilon=0.1)
    scheme.N_hits = N_hits

    delta = 0.05 # Yields 7.37, 20.0, 14.8, 79.6, 62.5 for
                 # N_delta_Hoeffding_scalar, N_delta_Chebyshev_scalar,
                 # N_delta_Bernstein_scalar, N_delta_Bernstein,
                 # N_delta_Bernstein_no_restricted_validity
    eps_stat, eps_syst, num_trun_obs = truncate_analytical(
        scheme, delta, get_epsilon_and_N_delta_functions_picker[1], 
        get_epsilon_and_N_delta_functions_picker[0], in_place=True
    )

    assert num_trun_obs == 0
    assert eps_syst == 0.0
    assert np.isclose(eps_stat, get_epsilon_and_N_delta_functions_picker[0](delta, N_hits, w))
    assert scheme.num_obs == 2
    
    out, _ = capfd.readouterr()
    assert "Nothing had to be truncated." in out
    
def test_truncate_analytical_rejects_tight_functions(get_epsilon_tighter_and_N_delta_functions_picker):
    w = np.array([0.5, 0.2])
    obs = np.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0]
    ])
    N_hits = np.array([100, 100])
    scheme = Measurement_scheme(obs, w, epsilon=0.1)
    scheme.N_hits = N_hits

    delta = 0.05 # Yields 7.37, 20.0, 14.8, 79.6, 62.5 for
                 # N_delta_Hoeffding_scalar, N_delta_Chebyshev_scalar,
                 # N_delta_Bernstein_scalar, N_delta_Bernstein,
                 # N_delta_Bernstein_no_restricted_validity

    with pytest.raises(ValueError, match="Tightened guarantee functions are not compatible"):
        truncate_analytical(
            scheme, delta,
            get_epsilon_tighter_and_N_delta_functions_picker[1],
            get_epsilon_tighter_and_N_delta_functions_picker[0],
            in_place=True
        )