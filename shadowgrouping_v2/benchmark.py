from shadowgrouping_v2.measurement_schemes import N_delta, sample_obs_from_setting
from shadowgrouping_v2.hamiltonian import char_to_int
import numpy as np
from copy import deepcopy
import json

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)

def save_to_json(diction,filename):
        with open(filename+".json", 'w') as f: 
            json.dump(diction, f, cls=NumpyEncoder)
        return
    
def load_dict(filename):
    outdict = {}
    with open(filename, "r") as f:
        f.readline()
        for line in f.readlines():
            vals = line.strip().split()
            key = vals[0]
            if key in outdict.keys():
                key += "+"
            outdict[key] = (float(vals[1]),float(vals[2]))
    return outdict
    
def load_settings(filename,estimator,N=None):
    estimator.reset()
    with open(filename, 'r') as f:
        data = json.load(f)
    if N is None:
        N = np.max(np.array(list(data.keys()),dtype=int))
        data = data[str(N)]
    else:
        temp = data.get(str(N),None)
        if temp is None:
            print("Warning! N={} not available as key in {}. Aborted.".format(N,data.keys()))
            return estimator
        data = temp
    estimator.num_settings = N
    estimator.settings_dict = data
    estimator.settings_buffer = estimator.settings_dict.copy()
    assert len(list(data.keys())[0]) == estimator.measurement_scheme.num_qubits, "Loaded settings correspond to a different qubit number."
    for setting,reps in data.items():
        setting = [char_to_int[c] for c in setting]
        is_hit = [sample_obs_from_setting(o,setting) for o in estimator.measurement_scheme.obs]
        estimator.measurement_scheme.N_hits[is_hit] += reps
    return estimator

def track_method_epsilon(estimator,E_GS,delta,benchmark_params={"Nshots":1000, "Nreps": 100, "Nsteps": 10, "truncate": False, "Nstart": None, "settings_filename": None,"load_at": None}):
    assert isinstance(benchmark_params,dict) or benchmark_params is None, "benchmark_params have to be either None or a dictionary."
    if benchmark_params is None:
        benchmark_params = {}
    # preprocessing of benchmark params
    Nshots        = max(1,benchmark_params.get("Nshots",10000))
    Nreps         = max(1,benchmark_params.get("Nreps",100))
    Nsteps        = max(2,benchmark_params.get("Nsteps",40))
    Nstart        = max(N_delta(delta),benchmark_params.get("Nstart",10))
    filename      = benchmark_params.get("settings_filename",None)
    assert isinstance(filename,str) or filename is None, "settings_filename has to be either None or str."
    load_at_N     = benchmark_params.get("load_at", None) if filename is not None else None
    if load_at_N is not None:
        assert load_at_N < Nstart, "Loading point has been set equal to or after starting point of tracking."
    save_dicts    = filename is not None
    truncate      = benchmark_params.get("truncate", False)
    use_naive     = benchmark_params.get("use_naive", False)
    N_steps       = np.unique(np.round(np.logspace(np.log10(Nstart),np.log10(Nshots),Nsteps),0)).astype(int)
    
    N_current = 0
    eps_provable = np.zeros(len(N_steps))
    eps_empirical = np.zeros_like(eps_provable)
    eps_std = np.zeros_like(eps_empirical)
    
    energies = np.zeros((Nreps,len(N_steps)))
    saved_settings = {}
    if estimator.measurement_scheme.is_sampling:
        # allocate and measure in one go, then repeat this process
        for j in range(Nreps):
            estimator.reset()
            N_current = 0
            if estimator.is_adaptive:
                threshold_ind = 0
            for i,Nval in enumerate(N_steps):
                # if estimator.method is adaptive, then take outcomes into account
                if estimator.is_adaptive:
                    while Nval > np.append(estimator.update_steps,Nshots)[threshold_ind]:
                        # sample settings until all applicable threshold values have been reached or the measurement budget is full anyways
                        # method itself takes care of taking the outcomes into account
                        Nshots_batch = estimator.update_steps[threshold_ind] - N_current
                        estimator.propose_next_settings(Nshots_batch)
                        estimator.measure() # outcome parsing to method is performed internally here
                        N_current += Nshots_batch
                        threshold_ind += 1
                        assert N_current in estimator.update_steps
                if Nval > N_current:
                    Nshots_batch = Nval - N_current
                    estimator.propose_next_settings(Nshots_batch)
                    estimator.measure()
                    N_current += Nshots_batch
                    assert N_current in N_steps
                if use_naive:
                    temp = deepcopy(estimator.measurement_scheme)
                    temp.update_variance_estimate(Nval != N_steps[i]) # update covariance matrix only if no further settings have to be alloted
                    energy = temp.get_energy()
                else:
                    energy = estimator.get_energy()
                energies[j,i] = energy
                eps_provable[i] += sum(estimator.measurement_scheme.get_epsilon_sys_stat(delta))
                if save_dicts and j==0:
                    saved_settings[str(Nval)] = estimator.settings_dict
                    save_to_json(saved_settings,filename)
        temp = abs(energies - E_GS)
        eps_empirical = np.mean(temp,axis=0)
        eps_std = np.std(temp,axis=0)
        assert len(eps_empirical) == len(eps_provable) and len(eps_empirical) == len(eps_std), "Lengths of arrays have changed during is_adaptive-part."
        eps_provable /= Nreps
    else:
        # find measurement settings once and sample based on these
        estimator.reset()
        if load_at_N is not None:
            estimator = load_settings(filename,estimator,N=load_at_N)
            assert estimator.num_settings == load_at_N, "Loading settings from file did not work."
            N_current = load_at_N
            filename += "_long"
        for i,Nval in enumerate(N_steps):
            estimator.propose_next_settings(Nval-N_current)
            for j in range(Nreps):
                estimator.clear_outcomes()
                estimator.measure()
                energies[j,i] = estimator.get_energy()
            temp = abs(energies[:,i] - E_GS)
            eps_empirical[i] = np.mean(temp)
            eps_std[i] = np.std(temp)
            N_current = Nval
            eps_provable[i] = sum(estimator.measurement_scheme.get_epsilon_sys_stat(delta))
            if save_dicts:
                saved_settings[str(Nval)] = estimator.settings_dict
                save_to_json(saved_settings,filename)
        if truncate:
            saved_settings_trunc = {}
            N_current = 0
            energies_truncated = np.zeros_like(energies)
            eps_prov_truncated = np.zeros_like(eps_provable)
            eps_emp_truncated  = np.zeros_like(eps_empirical)
            eps_std_truncated  = np.zeros_like(eps_std)
            eps_truncation = estimator.measurement_scheme.truncate(delta)
            estimator.reset()
            for i,Nval in enumerate(N_steps):
                estimator.propose_next_settings(Nval-N_current)
                for j in range(Nreps):
                    estimator.clear_outcomes()
                    estimator.measure()
                    energies_truncated[j,i] = estimator.get_energy()
                temp = abs(energies_truncated[:,i] - E_GS)
                eps_emp_truncated[i] = np.mean(temp)
                eps_std_truncated[i] = np.std(temp)
                N_current = Nval
                eps_prov_truncated[i] = sum(estimator.measurement_scheme.get_epsilon_sys_stat(delta)) + eps_truncation
                if save_dicts:
                    saved_settings_trunc[str(Nval)] = estimator.settings_dict
                    save_to_json(saved_settings_trunc,filename+"_trunc")
            
            return N_steps, eps_provable, eps_empirical, eps_std, energies, eps_prov_truncated, eps_emp_truncated, eps_std_truncated, energies_truncated
    return N_steps, eps_provable, eps_empirical, eps_std, energies
