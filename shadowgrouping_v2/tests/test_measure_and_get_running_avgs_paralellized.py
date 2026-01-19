import pytest, numpy as np, time, sys

log_file_path = "parallelization_benchmark_log.txt"
log_file = open(log_file_path, "w")
original_stdout = sys.stdout
sys.stdout = log_file

# SG_package_path = r"C:\\Users\\bpmur\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
SG_package_path = r"C:\\Users\\Bruno Murta\\OneDrive\\Microsoft Teams Chat Files\\Documents\\Work\\Research\\Numerics\\shadowgrouping_bpmurta_edit\\"
sys.path.insert(0, SG_package_path)

# defining paths where Hamiltonians and measurement schemes are stored
folder_Hamiltonians = SG_package_path + "Hamiltonians\\"
folder_schemes = SG_package_path + "Saved_Schemes\\"

from shadowgrouping.measurement_schemes import Shadow_Grouping
from shadowgrouping.weight_functions import Bernstein_bound
from shadowgrouping.energy_estimator import Energy_estimator, StateSampler
from shadowgrouping.hamiltonian import load_pauli_list

# Parameters to test
molecule_names = ["H2", "LiH", "BeH2", "H2O", "NH3"]
Nrounds_list = [1000, 10000, 100000]

# Initialization of other relevant variables
mapping_name = "JW" # choose one out of ["JW","BK","Parity"]
basis_set = "sto3g" # choose one out of ["sto3g","6-31g"] - the latter only for H2 molecule
num_qubits = {'H2': 4, 'LiH': 12, 'BeH2': 14, 'H2O': 14, "NH3": 16} 
N_reps_exp = 100

@pytest.mark.parametrize("molecule_name", molecule_names)
@pytest.mark.parametrize("Nrounds", Nrounds_list)
def test_parallel_vs_sequential(molecule_name, Nrounds):
    # Loading Hamiltonian
    observables, w, offset, E_GS, state = load_pauli_list(folder_Hamiltonians,molecule_name,basis_set,mapping_name,diagonalize=False)
    
    # loading exact ground state
    path_to_GS = folder_Hamiltonians + molecule_name + "_" + basis_set + "_" + str(num_qubits[molecule_name]) + "qubits\\exact_gs.txt"
    with open(path_to_GS, 'r') as f:
        data_string = f.read()
    state = data_string.split('\n')
    state = state[:len(state)-1]
    for i in range(len(state)):
        state[i] = complex(state[i])
    state = np.array(state)
    
    # Loading exact ground state energy
    path_to_GS_energy = folder_Hamiltonians + molecule_name + "_" + basis_set + "_" + str(num_qubits[molecule_name]) + "qubits\\exact_gs_energy.txt"
    with open(path_to_GS_energy, 'r') as f:
        data_string = f.read()
    E_GS = float(data_string)
    
    # Loading saved measurement scheme
    filename = folder_schemes + molecule_name + '_scheme_SGv1'
    npzfile = np.load(filename+'.npz')
    all_different_settings_saved = list(npzfile['ads'])
    settings_for_Nrounds = all_different_settings_saved[:Nrounds]
    
    # Defining state sampler
    state_sampler = StateSampler(state)

    # Construct shadow grouping method and estimator
    alpha = np.max(np.abs(w))/np.min(np.abs(w)) + np.min(np.abs(w))
    method = Shadow_Grouping(observables,w,0.1,Bernstein_bound(alpha=alpha)())
    estimator = Energy_estimator(method,state_sampler,offset=offset,N_reps_exp=N_reps_exp)
    
    # Adding measurement scheme to estimator
    estimator.reset()
    estimator.load_saved_scheme(settings_for_Nrounds)

    # Sequential execution
    t0 = time.perf_counter()
    estimator.clear_outcomes()
    estimator.measure()
    estimator.get_running_avgs()
    energy_seq = estimator.get_energy()
    t_seq = time.perf_counter() - t0
    temp = abs(np.array(energy_seq) - E_GS)
    RMSE_value_seq = np.mean(temp)
    RMSE_std_seq = np.std(temp)

    # Parallel execution
    t0 = time.perf_counter()
    estimator.clear_outcomes()
    estimator.measure_parallelized()
    estimator.get_running_avgs_parallelized()
    energy_par = estimator.get_energy()
    t_par = time.perf_counter() - t0
    temp = abs(np.array(energy_par) - E_GS)
    RMSE_value_par = np.mean(temp)
    RMSE_std_par = np.std(temp)

    # Print benchmark results
    print(f"\nBenchmark for {molecule_name} | N_reps_exp = {N_reps_exp}, Nrounds = {Nrounds}")
    print(f"Sequential   Time: {t_seq:.3f}s | RMSE: {RMSE_value_seq:.6f} \u00B1 {RMSE_std_seq:.6f}")
    print(f"Parallelized Time: {t_par:.3f}s | RMSE: {RMSE_value_par:.6f} \u00B1 {RMSE_std_par:.6f}")

def teardown_module(module):
    sys.stdout = original_stdout
    log_file.close()
