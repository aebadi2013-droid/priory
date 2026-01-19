import numpy as np
from qibo import models, gates
from shadowgrouping_v2.helper_functions import ( 
    char_to_int, settings_to_dict, index_to_string, setting_to_obs_form,
    sample_obs_batch_from_setting_numba, sample_obs_batch_from_setting_batch_numba)
from shadowgrouping_v2.noise_models import (
    apply_local_depolarizing_noise_end_of_circuit, apply_global_depolarizing_noise_end_of_circuit)
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_setting(setting, reps, statevector, N_reps_exp, measurement_scheme, order, is_adaptive):
    sampler = StateSampler(statevector)
    reps_eff = N_reps_exp * reps
    samples = sampler.sample(meas_basis=setting, nshots=reps_eff)

    setting_int = np.array([char_to_int[c] for c in setting])
    compatibility_mask = sample_obs_batch_from_setting_numba(measurement_scheme.obs, setting_int)

    obs_samples_data = {}
    for i in np.where(compatibility_mask)[0]:
        o = measurement_scheme.obs[i]
        temp = samples.copy()
        temp[:, o == 0] = 1
        obs_samples = np.prod(temp, axis=1)
        obs_samples_data[i] = obs_samples

    outcome_row = order[setting] if is_adaptive else None
    return setting, samples, obs_samples_data, outcome_row

def process_obs_samples(i, obs_dict, N_reps_exp):
    totals_i = np.zeros(N_reps_exp)
    counts_i = np.zeros(N_reps_exp, dtype=int)
    for samples in obs_dict.values():
        split_chunks = np.array_split(samples, N_reps_exp)
        for k, chunk in enumerate(split_chunks):
            totals_i[k] += np.sum(chunk)
            counts_i[k] += len(chunk)
    return i, totals_i, counts_i

class StateSampler():
    """ Convenience class that holds a fixed state of length 2**num_qubits. The latter number is inferred automatically.
        Provides a sampling method that obtains samples from the state in the chosen basis.
        
        Input:
        - state, numpy array of size 2**N with N = num_qubits. Coefficients are to be specified in the computational basis.
        - density_matrix, bool (defaults to False). Whether <state> has to be provided as density matrix (2^N,2^N)
        or state vector (2^N,)
    """
    def __init__(self,state,density_matrix=False):
        self.state = np.array(state)
        self.num_qubits = int(np.round(np.log2(len(state)),0))
        assert len(self.state) == 2**self.num_qubits, "State size has to be of size 2**N for some integer N."
        self.circuit = models.Circuit(self.num_qubits,density_matrix=density_matrix)
        self.X = [gates.H,gates.I] # use Hadamard gate to switch to +/- basis
        S_dagger = lambda i: gates.U1(i,-np.pi/2)
        self.Y = [S_dagger,gates.H] # use Hadamard + phase gate to switch to +/-i basis
        self.Z = [gates.I,gates.I] # no need to change if already in comp. basis or not measuring at all
        self.I = self.Z
        return
        
    def sample(self,meas_basis=None,nshots=1):
        """ Draws <nshots> samples from the state.
            If no measurement basis is defined, samples are drawn in the computational basis.
            
            Inputs:
            - meas_basis as a Pauli string, i.e. a str of length num_qubits containing only X,Y,Z,I.
            - nshots (int) specifying how many samples to draw.
            
            Returns:
            - samples, numpy array of shape (nshots x  num_qubits).
        """
        c = self.circuit.copy(deep=True)
        if meas_basis is not None:
            assert len(meas_basis)==self.num_qubits, "Measurement basis has to be specified for each qubit."
            c.add([getattr(self,s)[0](i) for i,s in enumerate(meas_basis)])
            c.add([getattr(self,s)[1](i) for i,s in enumerate(meas_basis)])
        c.add(gates.M(*range(self.num_qubits)))
        out = c(initial_state=self.state.copy(),nshots=nshots)
        out = out.samples()
        # mask-out non-measured qubits in <meas-basis>
        out = out * np.array([s!="I" for s in meas_basis],dtype=int)[np.newaxis,:]
        return -2*out + 1 # map {0,1} to {1,-1} outcomes
    
class Sign_estimator():
    
    def __init__(self,measurement_scheme,state,offset):
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state        = state
        self.offset       = offset
        self.setting_inds = []
        self.outcomes     = []
        self.num_settings = 0
        self.num_outcomes = 0
        #self.settings_dict = {} 1111111
    
    def reset(self):
        self.setting_inds = []
        self.outcomes     = []
        self.num_settings, self.num_outcomes = 0, 0
        #self.settings_dict = {} 1111111
        self.measurement_scheme.reset()
        
    def clear_outcomes(self):
        self.outcomes = []
        self.num_outcomes = 0
        
    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        inds = self.measurement_scheme.find_setting(num_steps)
        self.setting_inds = np.append(self.setting_inds,inds) if len(self.setting_inds)>0 else inds
        self.num_settings += num_steps
        for ind in inds:
            self.measurement_scheme.settings_dict[ind] = 1 #1111111
        return
    
    def measure(self):
        """ If there are proposed settings in self.settings that have not been measured, do so.
            The internal state of the VQE does not alter by doing so.
        """
        num_meas = self.num_settings - self.num_outcomes
        if num_meas > 0:
            # run all the last prepared measurement settings
            # from the settings list, fetch the unique settings and their respective counts
            recent_settings = self.setting_inds[-num_meas:]            
            outcomes = np.zeros(num_meas,dtype=int)
            for unique,nshots in zip(*np.unique(recent_settings,return_counts=True)):
                setting = self.measurement_scheme.obs[unique]
                samples = self.state.sample(meas_basis=index_to_string(setting),nshots=nshots)
                outcomes[recent_settings==unique] = np.prod(samples,axis=-1)
            self.outcomes = np.append(self.outcomes,outcomes)
            self.num_outcomes += num_meas
        else:
            print("No more measurements required at the moment. Please propose new setting(s) first.")
        return
    
    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energy. """
        if self.num_outcomes == 0:
            # if no measurements have been done yet, just return the offset value
            return self.offset
        w = self.measurement_scheme.w
        sgn = np.sign(w)
        norm = np.sum(np.abs(w))
        energy = np.mean(self.outcomes*sgn[self.setting_inds])*norm
        return energy + self.offset

class Energy_estimator():
    """ Convenience class that holds both a measurement scheme and a StateSampler instance.
        The main workflow consists of proposing the next (few) measurement settings and measuring them in the respective bases.
        Furthermore, it tracks all measurement settings and their respective outcomes (of value +/-1 per qubit).
        Based on these values, the current energy estimate can be calculated.
        
        Inputs:
        - measurement_scheme, see class Measurement_Scheme and subclasses for information.
        - state, see class StateSampler.
        - Energy offset (defaults to 0) for the energy estimation.
          This consists of the identity term in the corresponding Hamiltonian decomposition.
    """
    def __init__(self,measurement_scheme,state,offset=0,spin_corr=None,N_reps_exp=1):
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state        = state
        self.offset       = offset
        # convenience counters to keep track of measurements settings and respective outcomes
        #self.settings_dict = {} 1111111
        self.settings_buffer = {}
        self.raw_samples_dict = {}
        self.obs_samples_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.num_settings = 0
        self.num_outcomes = 0
        self.running_N = np.zeros(self.measurement_scheme.num_obs, dtype=int)
        self._N_reps_exp = None
        self.N_reps_exp = N_reps_exp
        self.measurement_scheme.reset()
        self.is_adaptive  = measurement_scheme.is_adaptive
        if self.is_adaptive:
            self.update_steps = np.array(measurement_scheme.update_steps)
            self.order = {}
            assert hasattr(measurement_scheme,"receive_outcome"), "The given method does not have the function receive_outcome()."
        else:
            self.order = None
        
    def reset(self):
        #self.settings_dict = {} 1111111
        self.settings_buffer = {}
        self.samples_dict = {}
        self.obs_samples_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.num_settings, self.num_outcomes = 0, 0
        self.running_N = np.zeros(self.measurement_scheme.num_obs, dtype=int)
        self.running_avgs = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs))
        self.measurement_scheme.N_hits = np.zeros_like(self.measurement_scheme.N_hits)
        self.measurement_scheme.all_settings_list = []
        self.measurement_scheme.num_diff_settings_list = []
        self.measurement_scheme.diff_settings_counter = 0
        if hasattr(self,"outcome_dict"):
            self.outcome_dict = {}
        return
    
    def clear_outcomes(self):
        self.settings_buffer = self.measurement_scheme.settings_dict.copy() #1111111
        #self.measurement_scheme.settings_buffer = self.settings_buffer
        self.samples_dict = {}
        self.running_N = np.zeros(self.measurement_scheme.num_obs, dtype=int)
        self.running_avgs = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs))
        self.obs_samples_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.num_outcomes = 0
        if hasattr(self,"outcome_dict"):
            self.outcome_dict = {}
        return
    
    @property
    def N_reps_exp(self):
        return self._N_reps_exp
    
    @N_reps_exp.setter
    def N_reps_exp(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("N_reps_exp must be a positive integer.")
        self._N_reps_exp = value
        self.running_avgs = np.zeros((value, self.measurement_scheme.num_obs))
        self.clear_outcomes()
        print(f"Updated N_reps_exp to {value}. Cleared outcomes using clear_outcomes().")
        
    def load_saved_scheme(self, settings_for_Nrounds, num_diff_settings_for_Nrounds=None):
        self.num_settings = len(settings_for_Nrounds)
        settings_to_dict(settings_for_Nrounds, self.measurement_scheme.settings_dict, self.settings_buffer) #1111111
        distinct_settings, distinct_settings_reps = zip(*self.measurement_scheme.settings_dict.items()) #1111111
        converted_distinct_settings = np.array([setting_to_obs_form(s) for s in distinct_settings])
        distinct_settings_reps = np.array(distinct_settings_reps)
        compatibility_matrix = sample_obs_batch_from_setting_batch_numba(self.measurement_scheme.obs, converted_distinct_settings)
        self.measurement_scheme.N_hits = (compatibility_matrix.astype(int) @ distinct_settings_reps)
    
        if self.measurement_scheme.save_scheme:
            self.measurement_scheme.all_settings_list = settings_for_Nrounds
            if num_diff_settings_for_Nrounds is not None:
                self.measurement_scheme.num_diff_settings_list = num_diff_settings_for_Nrounds
                self.measurement_scheme.diff_settings_counter = num_diff_settings_for_Nrounds[len(num_diff_settings_for_Nrounds)-1]
            else:
                raise ValueError("The argument 'num_diff_settings_for_Nrounds' must be provided (i.e., cannot be None) when 'self.save_scheme' is set to True.")
        return
    
    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        if self.is_adaptive:
            # check that num_steps does not exceed the next threshold for updating the measurement_scheme internally
            # if so, limit num_steps accordingly
            thresholds = self.update_steps - self.num_settings
            thresholds = thresholds[thresholds>0] # throw away all reached thresholds but the last one (threshold is ordered)
            if len(thresholds)>0:
                max_steps_allowed = thresholds[0]
                if num_steps > max_steps_allowed:
                    print("Warning! Trying to allocate more settings than allowed before updating the measurement scheme with outcomes.")
                    print("Num_steps = {0} reduced to {1}. Allocating num_steps={1} instead.".format(num_steps,max_steps_allowed))
                    num_steps = max_steps_allowed
        settings = []
        for i in range(num_steps):
            p, _ = self.measurement_scheme.find_setting()
            settings.append(p)
            if self.is_adaptive:
                settings_to_dict(settings, self.measurement_scheme.settings_dict, self.settings_buffer, 
                             self.is_adaptive, self.order) #1111111
            else:
                settings_to_dict(settings, self.measurement_scheme.settings_dict, self.settings_buffer)
        settings = np.array(settings)
        self.num_settings += num_steps
        #if self.is_adaptive:
        #    settings_to_dict(settings, self.measurement_scheme.settings_dict, self.settings_buffer, 
        #                     self.is_adaptive, self.order) #1111111
        #else:
        #    settings_to_dict(settings, self.measurement_scheme.settings_dict, self.settings_buffer) #1111111
        return
    
    def measure_and_get_running_avgs(self,p_GDN=None,p_array_LDN=None):
        num_meas = self.num_settings - self.num_outcomes
        if num_meas == 0:
            print("Trying to measure more measurement settings than allocated. Please allocate measurements first by calling propose_next_settings() first.")
            return
        if self.is_adaptive:
            outcomes = np.zeros((num_meas, self.measurement_scheme.num_qubits), dtype=int)
        # Preallocate accumulators for all repetitions and observables
        totals = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs))
        counts = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs), dtype=int)
        for setting, reps in self.settings_buffer.items():
            reps_eff = self.N_reps_exp * reps
            samples = self.state.sample(meas_basis=setting, nshots=reps_eff)
            # Apply local depolarizing noise model if p_array_LDN is provided
            if p_array_LDN is not None:
                samples = apply_local_depolarizing_noise_end_of_circuit(samples,setting,p_array_LDN)
            # Apply global depolarizing noise model if p_GDN is provided
            if p_GDN is not None:
                samples = apply_global_depolarizing_noise_end_of_circuit(samples, p_GDN)
            # Update raw_samples_dict
            if setting not in self.raw_samples_dict:
                self.raw_samples_dict[setting] = samples
            else:
                self.raw_samples_dict[setting] = np.vstack((self.raw_samples_dict[setting], samples))
            if hasattr(self, "outcome_dict"):
                self.outcome_dict[setting] = samples
            if self.is_adaptive:
                outcomes[self.order[setting]] = samples
            # Process observable samples
            setting_int = np.array([char_to_int[c] for c in setting])
            compatibility_filter = sample_obs_batch_from_setting_numba(self.measurement_scheme.obs, setting_int)
            for i in np.where(compatibility_filter)[0]:
                o = self.measurement_scheme.obs[i]        
                o_support = (o == 0)
                temp = samples.copy()
                temp[:, o_support] = 1
                obs_samples = np.prod(temp, axis=1)
                # Accumulate running statistics
                for k in range(self.N_reps_exp):
                    chunk = obs_samples[k * reps : (k + 1) * reps]
                    totals[k, i] += np.sum(chunk)
                    counts[k, i] += reps
        # Final update of running averages
        for k in range(self.N_reps_exp):
            for i in range(self.measurement_scheme.num_obs):
                total_reps = self.running_N[i] + counts[k, i]
                if total_reps > 0:
                    self.running_avgs[k, i] = (
                        self.running_avgs[k, i] * self.running_N[i] + totals[k, i]
                    ) / total_reps
        # Update running_N only once
        self.running_N += counts[-1,:]
        if self.is_adaptive:
            for outcome in outcomes:
                self.measurement_scheme.receive_outcome(outcome)
        self.num_outcomes = self.num_settings
        self.settings_buffer = {}
        return

    def measure(self,p_GDN=None,p_array_LDN=None):
        num_meas = self.num_settings - self.num_outcomes
        if num_meas == 0:
            print("Trying to measure more measurement settings than allocated. Please allocate measurements first by calling propose_next_settings() first.")
            return
        if self.is_adaptive:
            outcomes = np.zeros((num_meas, self.measurement_scheme.num_qubits), dtype=int)    
        for setting, reps in self.settings_buffer.items():
            reps_eff = self.N_reps_exp * reps
            samples = self.state.sample(meas_basis=setting, nshots=reps_eff)
            # Apply local depolarizing noise model if p_array_LDN is provided
            if p_array_LDN is not None:
                samples = apply_local_depolarizing_noise_end_of_circuit(samples,setting,p_array_LDN)
            # Apply global depolarizing noise model if p_GDN is provided
            if p_GDN is not None:
                samples = apply_global_depolarizing_noise_end_of_circuit(samples, p_GDN)
            if setting in self.raw_samples_dict:
                self.raw_samples_dict[setting] = np.vstack((self.raw_samples_dict[setting], samples))
            else:
                self.raw_samples_dict[setting] = samples
            if hasattr(self, "outcome_dict"):
                self.outcome_dict[setting] = samples
            if self.is_adaptive:
                outcomes[self.order[setting]] = samples
            setting_int = np.array([char_to_int[c] for c in setting])
            compatibility_mask = sample_obs_batch_from_setting_numba(self.measurement_scheme.obs, setting_int)
            # Process only compatible observables
            for i in np.where(compatibility_mask)[0]:
                o = self.measurement_scheme.obs[i]
                temp = samples.copy()
                temp[:, o == 0] = 1
                obs_samples = np.prod(temp, axis=1)    
                if setting in self.obs_samples_dict_list[i]:
                    self.obs_samples_dict_list[i][setting] = np.concatenate(
                        (self.obs_samples_dict_list[i][setting], obs_samples)
                    )
                else:
                    self.obs_samples_dict_list[i][setting] = obs_samples
        if self.is_adaptive:
            for outcome in outcomes:
                self.measurement_scheme.receive_outcome(outcome)
        self.num_outcomes = self.num_settings
        self.settings_buffer = {}
        return
    
    def get_running_avgs(self):
        # Initialize accumulators
        totals = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs))
        counts = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs), dtype=int)
    
        # Loop over observables
        for i, obs_dict in enumerate(self.obs_samples_dict_list):
            for samples in obs_dict.values():
                # Split samples into N_reps_exp chunks
                split_chunks = np.split(samples, self.N_reps_exp)
                for k, chunk in enumerate(split_chunks):
                    totals[k, i] += np.sum(chunk)
                    counts[k, i] += len(chunk)
    
        # Compute updated running averages
        for k in range(self.N_reps_exp):
            for i in range(self.measurement_scheme.num_obs):
                total_reps = self.running_N[i] + counts[k, i]
                if total_reps > 0:
                    self.running_avgs[k, i] = (
                        self.running_avgs[k, i] * self.running_N[i] + totals[k, i]
                    ) / total_reps
    
        # Update running counts (only once, on final rep)
        self.running_N += counts[-1, :]
    
        return
    
    def measure_parallelized(self,p_GDN=None,p_array_LDN=None):
        
        num_meas = self.num_settings - self.num_outcomes
        if num_meas == 0:
            print("Trying to measure more measurement settings than allocated. Please allocate measurements first.")
            return
        if self.is_adaptive:
            outcomes = np.zeros((num_meas, self.measurement_scheme.num_qubits), dtype=int)
    
        # Extract only the picklable statevector
        statevector = self.state.state
    
        futures = []
        with ProcessPoolExecutor() as executor:
            for setting, reps in self.settings_buffer.items():
                futures.append(executor.submit(
                    process_setting,
                    setting, reps,
                    statevector,
                    self.N_reps_exp,
                    self.measurement_scheme,
                    self.order, self.is_adaptive
                ))
    
            for fut in as_completed(futures):
                setting, samples, obs_samples_data, outcome_row = fut.result()
                
                if setting in self.raw_samples_dict:
                    self.raw_samples_dict[setting] = np.vstack((self.raw_samples_dict[setting], samples))
                else:
                    self.raw_samples_dict[setting] = samples
                if hasattr(self, "outcome_dict"):
                    self.outcome_dict[setting] = samples
                if self.is_adaptive and outcome_row is not None:
                    outcomes[outcome_row] = samples
    
                for i, obs_samples in obs_samples_data.items():
                    if setting in self.obs_samples_dict_list[i]:
                        self.obs_samples_dict_list[i][setting] = np.concatenate(
                            (self.obs_samples_dict_list[i][setting], obs_samples)
                        )
                    else:
                        self.obs_samples_dict_list[i][setting] = obs_samples
    
        if self.is_adaptive:
            for outcome in outcomes:
                self.measurement_scheme.receive_outcome(outcome)
    
        self.num_outcomes = self.num_settings
        self.settings_buffer = {}
    
    def get_running_avgs_parallelized(self):
        
        totals = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs))
        counts = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs), dtype=int)
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_obs_samples, i, obs_dict, self.N_reps_exp)
                       for i, obs_dict in enumerate(self.obs_samples_dict_list)]
        
            for fut in as_completed(futures):
                i, totals_i, counts_i = fut.result()
                totals[:, i] = totals_i
                counts[:, i] = counts_i
        
        for k in range(self.N_reps_exp):
            for i in range(self.measurement_scheme.num_obs):
                total_reps = self.running_N[i] + counts[k, i]
                if total_reps > 0:
                    self.running_avgs[k, i] = (
                        self.running_avgs[k, i] * self.running_N[i] + totals[k, i]
                    ) / total_reps
        
        self.running_N += counts[-1, :]
        return

    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energies
            for all the N_reps_exp repetitions of the experiment. """        
        energy_estimates = self.running_avgs.dot(self.measurement_scheme.w) + self.offset
        
        return energy_estimates

    
