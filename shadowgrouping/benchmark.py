from shadowgrouping.energy_estimator import Energy_estimator, StateSampler
from shadowgrouping.measurement_schemes import N_delta
import numpy as np
from copy import deepcopy

def benchmark_empirical(method,offset,state,E_GS,benchmark_params={"Nshots":1000, "Nreps": 100, "truncate_delta": None}):
    """ Benchmark 1 of the manuscript. Uses the method to allocate <Nshots> many measurement settings to measure from <state>.
        Afterwards, the energy estimate E' is reconstructed and benchmarked against <E_GS> via RMSE.
        This is averaged over <Nreps> independent measurement runs.
        If truncate_delta is set to a value above zero, truncates the observable list after first allocation and allocates again.
        Returns RMSE and its std deviation as well as the individual estimates.
    """
    assert isinstance(state,StateSampler), "State-instance has to be wrapped in StateSampler class."
    assert isinstance(benchmark_params,dict) or benchmark_params is None, "benchmark_params have to be either None or a dictionary."
    if benchmark_params is None:
        benchmark_params = {}
    # preprocessing of benchmark params
    Nshots         = benchmark_params.get("Nshots",1000)
    Nreps          = benchmark_params.get("Nreps",100)
    truncate_delta = benchmark_params.get("truncate_delta", None)
    use_naive      = benchmark_params.get("use_naive", False)
    if truncate_delta is not None:
        assert truncate_delta > 0, "Delta value for truncation value has to be positive, but was {}.".format(truncate_delta)
    
    # generate settings
    if method.is_sampling:
        estimates = []
        for r in range(Nreps):
            estimator = Energy_estimator(method,state,offset)
            estimator.reset()
            # if estimator.method is adaptive, then take outcomes into account
            if estimator.is_adaptive:
                last_threshold = 0
                for threshold in np.append(estimator.update_steps,Nshots):
                    # sample settings until the next threshold value is reached or the measurement budget is full
                    # method itself takes care of taking the outcomes into account
                    Nshots_batch = threshold - last_threshold
                    estimator.propose_next_settings(Nshots_batch)
                    estimator.measure() # outcome parsing to method is performed internally here
                    last_threshold = threshold
            else:
                estimator.propose_next_settings(Nshots)
                estimator.measure()
            if use_naive:
                estimator.measurement_scheme.update_variance_estimate()
                energy = estimator.measurement_scheme.get_energy()
            else:
                energy = estimator.get_energy()[0]
            estimates.append(energy)
        estimates = np.array(estimates)
    else:
        # next settings have to be allocated only once
        if truncate_delta is not None:
            for _ in range(Nshots):
                method.find_setting()
            method.truncate(truncate_delta)
            method.reset()
        estimator = Energy_estimator(method,state,offset,repeats=Nreps)
        estimator.propose_next_settings(Nshots)
        estimator.measure()
        estimates = estimator.get_energy()
        
    # get statistics
    diffs = (estimates - E_GS)**2
    RMSE  = np.sqrt(np.mean(diffs))
    STD   = np.sqrt(np.std(diffs)/Nreps)
    
    return RMSE, STD, estimates

def benchmark_provable(method,delta,benchmark_params={"Nshots":1000, "Nreps": 100, "Nsteps": 10, "truncate": False}):
    """ Benchmark 2 of the manuscript. Uses the method to allocate <Nshots> many measurement settings.
        Afterwards, the provable error epsilon is reconstructed between N_delta(<delta>) and <Nshots>, with <Nsteps>+2 log-steps.
        This is averaged over <Nreps> independent measurement runs in case the method samples the settings.
        Returns epsilon and (epsilon,epsilon_truncated) in case the method is also to be truncated.
    """
    assert isinstance(benchmark_params,dict) or benchmark_params is None, "benchmark_params have to be either None or a dictionary."
    if benchmark_params is None:
        benchmark_params = {}
    # preprocessing of benchmark params
    Nshots   = benchmark_params.get("Nshots",1000)
    Nreps    = benchmark_params.get("Nreps",100)
    Nsteps   = benchmark_params.get("Nsteps",10)
    truncate = benchmark_params.get("truncate", False)
    Nvals    = np.unique(np.round(np.logspace(np.log10(N_delta(delta)),np.log10(Nshots),Nsteps+2),0)).astype(int)
    
    # generate settings
    epsilon = np.zeros(len(Nvals),dtype=float)
    if method.is_sampling:
        for _ in range(Nreps):
            method.reset()
            Nnext = 0
            for i,Nval in enumerate(Nvals):
                for _ in range(Nval-Nnext):
                    method.find_setting()
                epsilon[i] += sum(method.get_epsilon_sys_stat(delta))
                Nnext = Nval
        epsilon /= Nreps
    else:
        Nnext = 0
        for i,Nval in enumerate(Nvals):
            for _ in range(Nval-Nnext):
                method.find_setting()
            epsilon[i] = sum(method.get_epsilon_sys_stat(delta))
            Nnext = Nval
        if truncate:
            # after first run through:
            # truncate the observable list and reallocate everything again on the truncated list
            epsilon_truncated = np.zeros_like(epsilon)
            epsilon_syst = method.truncate(delta)
            method.reset()
            Nnext = 0
            for i,Nval in enumerate(Nvals):
                for _ in range(Nval-Nnext):
                    method.find_setting()
                epsilon_truncated[i] = sum(method.get_epsilon_sys_stat(delta)) + epsilon_syst
                Nnext = Nval
            return Nvals, epsilon, epsilon_truncated
    return Nvals, epsilon

def save_dict(filename,savedict,append=False):
    mode = "a" if append else "w"
    with open(filename, mode) as f:
        if not append:
            f.write("Method\tRMSE\tSTD\n")
        for label,val in savedict.items():
            line = label + "\t{}\t{}\n".format(val[0],val[1])
            f.write(line)
    return

def save_dict_provable(filename,savedict,Nsteps,append=False):
    mode = "a" if append else "w"
    with open(filename, mode) as f:
        if not append:
            f.write("Method")
            for N in Nsteps:
                f.write("\t{}".format(N))
            f.write("\n")
        for label,vals in savedict.items():
            f.write(label)
            for val in vals:
                f.write("\t{}".format(val))
            f.write("\n")
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

def load_dict_provable(filename):
    outdict = {}
    with open(filename, "r") as f:
        # first row contains the Nsteps values
        Nsteps = np.array(f.readline().strip().split()[1:],dtype=int)
        # other lines contain the corresponding values
        for line in f.readlines():
            vals = line.strip().split()
            key = vals[0]
            if key in outdict.keys():
                key += "+"
            outdict[key] = np.array(vals[1:],dtype=float)
    return Nsteps, outdict
