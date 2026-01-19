import numpy as np, sys, pytest

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\ShadowGrouping Code\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.truncations import pre_truncate
from shadowgrouping.measurement_schemes import Measurement_scheme

def test_pre_truncate_removes_correct_observables():
    # Setup
    w = np.array([0.1, 0.05, 0.5, 0.02, 0.3])
    obs = np.array([
        [0, 1, 2, 3],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [3, 0, 0, 0],
        [3, 1, 0, 2]
    ])
    N_hits = np.array([10, 10, 10, 10, 10])
    scheme = Measurement_scheme(obs, w, epsilon=0.1)
    scheme.N_hits = N_hits

    # Pre-truncate with tau = 0.1, which should remove obs with weights 0.02 and 0.05
    tau = 0.1
    eps_sys = pre_truncate(scheme, tau)

    assert np.isclose(eps_sys, 0.07)
    assert scheme.num_obs == 3
    assert np.all(scheme.w >= 0.1)
    assert scheme.obs.shape == (3, 4)
    assert scheme.N_hits.shape == (3,)
    
def test_pre_truncate_discards_all():
    w = np.array([0.01, 0.01])
    obs = np.array([
        [0, 1, 1, 0],
        [2, 2, 0, 3]
    ])
    N_hits = np.array([5, 5])
    scheme = Measurement_scheme(obs, w, epsilon=0.1)
    scheme.N_hits = N_hits
    
    tau = 1.0  # Large enough to remove all
    eps_sys = pre_truncate(scheme, tau)

    assert eps_sys == 0
    assert scheme.num_obs == 2
    assert scheme.obs.shape == (2, 4)  # Unchanged
    assert np.array_equal(scheme.obs, obs)

def test_pre_truncate_discards_nothing():
    w = np.array([0.3, 0.4])
    obs = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ])
    N_hits = np.array([20, 30])
    scheme = Measurement_scheme(obs, w, epsilon=0.1)
    scheme.N_hits = N_hits
    
    tau = 0.01  # Too small to discard anything
    eps_sys = pre_truncate(scheme, tau)

    assert eps_sys == 0
    assert scheme.num_obs == 2
    assert scheme.obs.shape == (2, 4)
    assert np.array_equal(scheme.obs, obs)