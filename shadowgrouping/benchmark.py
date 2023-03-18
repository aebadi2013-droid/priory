from energy_estimator import Energy_estimator, StateSampler
from measurement_schemes import N_delta
import numpy as np

def benchmark_empirical(method,offset,state,E_GS,benchmark_params={"Nshots":1000, "Nreps": 100}):
    """ Benchmark 1 of the manuscript. Uses the method to allocate <Nshots> many measurement settings to measure from <state>.
        Afterwards, the energy estimate E' is reconstructed and benchmarked against <E_GS> via RMSE.
        This is averaged over <Nreps> independent measurement runs.
        Returns RMSE and its std deviation.
    """
    assert isinstance(state,StateSampler), "State-instance has to be wrapped in StateSampler class."
    assert isinstance(benchmark_params,dict) or benchmark_params is None, "benchmark_params have to be either None or a dictionary."
    # preprocessing of benchmark params
    Nshots = benchmark_params.get("Nshots",1000)
    Nreps  = benchmark_params.get("Nreps",100)
    
    # generate settings
    if method.is_adaptive:
        estimates = []
        for r in range(Nreps):
            estimator = Energy_estimator(method,state,offset)
            estimator.propose_next_settings(Nshots)
            estimator.measure()
            estimates.append(estimator.get_energy())
    else:
        # next settings have to be allocated only once
        estimator = Energy_estimator(method,state,offset,repeats=Nreps)
        estimator.propose_next_settings(Nshots)
        estimator.measure()
        estimates = estimator.get_energy()
        
    # get statistics
    diffs = (estimates - E_GS)**2
    RMSE  = np.sqrt(np.mean(diffs))
    STD   = np.sqrt(np.std(diffs)/Nreps)
    
    return RMSE, STD

def benchmark_provable(method,delta,benchmark_params={"Nshots":1000, "Nreps": 100, "Nsteps": 10, "truncate": False}):
    """ Benchmark 2 of the manuscript. Uses the method to allocate <Nshots> many measurement settings.
        Afterwards, the provable error epsilon is reconstructed between N_delta(<delta>) and <Nshots>, with <Nsteps>+2 log-steps.
        This is averaged over <Nreps> independent measurement runs in case the method samples the settings.
        Returns epsilon and (epsilon,epsilon_truncated) in case the method is also to be truncated.
    """
    assert isinstance(benchmark_params,dict) or benchmark_params is None, "benchmark_params have to be either None or a dictionary."
    # preprocessing of benchmark params
    Nshots   = benchmark_params.get("Nshots",1000)
    Nreps    = benchmark_params.get("Nreps",100)
    Nsteps   = benchmark_params.get("Nsteps",10)
    truncate = benchmark_params.get("truncate", False)
    Nvals    = np.logspace(np.log10(N_delta(delta)),np.log10(Nshots),Nsteps+2)
    
    # generate settings
    epsilon = np.zeros_like(Nvals)
    if method.is_adaptive:
        for _ in range(Nreps):
            method.reset()
            Nnext = 0
            for i,Nval in enumerate(Nvals):
                for _ in range(Nval-Nnext):
                    method.find_setting()
                epsilon[i] += method.get_epsilon(delta)
                Nnext = Nval
        epsilon /= Nreps
    else:
        Nnext = 0
        for i,Nval in enumerate(Nvals):
            for _ in range(Nval-Nnext):
                method.find_setting()
            epsilon[i] = method.get_epsilon(delta)
            Nnext = Nval
        if truncate:
            savestate = method.copy()
            epsilon_truncated = np.zeros_like(epsilon)
            epsilon_syst = method.truncate(delta)
            method.reset()
            Nnext = 0
            for i,Nval in enumerate(Nvals):
                for _ in range(Nval-Nnext):
                    method.find_setting()
                epsilon_truncated[i] = method.get_epsilon(delta)
            method = savestate
            return epsilon, epsilon_truncated+epsilon_syst
    return epsilon