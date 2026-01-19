import pytest, sys
import numpy as np

# adding ShadowGrouping package to the system path
# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

from shadowgrouping.guarantees import N_delta_Bernstein

@pytest.fixture(params=[0, 1, -0.01, 1.01])
def invalid_delta(request):
    return request.param

def test_very_small_delta(delta=1e-10):
    assert np.isfinite(N_delta_Bernstein(delta))
    
def test_invalid_deltas_raise_error(invalid_delta):
    with pytest.raises(ValueError, match="delta must be in the interval \\(0,1\\)"):
        N_delta_Bernstein(invalid_delta)

