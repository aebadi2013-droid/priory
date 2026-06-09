import numpy as np
from qibo import gates

from shadowgrouping_v2.shadowgrouping_my_dev.full_commutativity import (
    diagonalize_and_map, optimize_clifford_decomposition, 
    optimize_clifford_decomposition_global_setting, qibo_circuit_from_gate_list)
from shadowgrouping_v2.shadowgrouping_my_dev.graph_theory_methods import clique_measurement_basis
from shadowgrouping_v2.shadowgrouping_my_dev.helper_functions import (
     int_to_char, settings_to_dict, index_to_string, _pack_token_count_dict, _unpack_token_count_dict,
     token_to_pauli_list, decode_setting_token, commute_blockwise, combine_seed)
from shadowgrouping_v2.shadowgrouping_my_dev.noise_models import (
    apply_local_depolarizing_noise_end_of_circuit, apply_global_depolarizing_noise_end_of_circuit,
    apply_stochastic_bit_flips, apply_assignment_matrix)
from shadowgrouping_v2.shadowgrouping_my_dev.qubit_wise_commutativity import (generate_qwc_basis_transformation_circuit,
                                                            hit_by_batch_numba)
from shadowgrouping_v2.shadowgrouping_my_dev.steady_state_allocator_mixin import SteadyStateAllocatorMixin

S_dagger = lambda i: gates.U1(i,-np.pi/2)

class StateSampler:
    """ Convenience class that holds a fixed state of length 2**num_qubits.
        Provides a sampling method that obtains samples from the state in a chosen
        basis, possibly with added noise (if optional inputs are provided).

        Inputs:
        - state: numpy array of size 2**N (statevector) or (2**N, 2**N) (density matrix),
                 with coefficients given in the computational basis.
        - density_matrix: bool (defaults to False). Whether `state` is a density matrix.
        - p_GDN: parameter of global depolarizing noise model applied at end.
        - p_array_LDN: parameters of local depolarizing noise model, one for
                       each qubit (so list/array of length num_qubits), applied only at end.
        - readout_noise_stochastic: four inputs required to execute
              apply_stochastic_bit_flips (see noise_models.py).
        - readout_noise_A_matrix: four inputs required to execute
              apply_assignment_matrix (see noise_models.py).
    """

    def __init__(self, state, density_matrix=False,
                 p_GDN=None, p_array_LDN=None,
                 readout_noise_stochastic=None,
                 readout_noise_A_matrix=None):
        self.state = np.array(state)
        self.density_matrix = density_matrix

        # Infer number of qubits from leading dimension
        if self.state.ndim == 1:
            dim = self.state.shape[0]
        elif self.state.ndim == 2:
            dim = self.state.shape[0]
            if self.state.shape[0] != self.state.shape[1]:
                raise ValueError("Density matrix must be square: got shape "
                                 f"{self.state.shape}.")
        else:
            raise ValueError("State must be a 1D statevector or 2D density matrix.")

        self.num_qubits = int(round(np.log2(dim)))
        if 2**self.num_qubits != dim:
            raise ValueError("State size has to be 2**N for some integer N; "
                             f"got dim={dim}.")

        # Noise model parameters
        self.p_GDN = p_GDN
        self.p_array_LDN = p_array_LDN
        self.readout_noise_stochastic = readout_noise_stochastic
        self.readout_noise_A_matrix = readout_noise_A_matrix

    def sample(self, basis_transformation_circuit, nshots=1, 
               measured_qubits=None, seed=None):
        """Draw <nshots> samples from the state.

        Parameters
        ----------
        basis_transformation_circuit : qibo.models.Circuit
            Qibo circuit implementing the desired basis transformation. It must be
            constructed with the correct number of qubits and the same `density_matrix`
            flag as this StateSampler's state.
        nshots : int
            Number of samples to draw.
        measured_qubits : sequence of int or None
            List of global qubit indices that are considered "measured" for the
            purpose of noise models. All qubits are actually measured at the end
            of the circuit; this list is only used to select which columns the
            noise models act on. If None, defaults to all qubits.

        Returns
        -------
        samples : np.ndarray
            Array of shape (nshots, num_qubits) with entries in {+1, -1}.
        """
        if not isinstance(nshots, int) or nshots <= 0:
            raise ValueError("nshots must be a positive integer.")

        if measured_qubits is None:
            measured_qubits = list(range(self.num_qubits))
        else:
            measured_qubits = list(measured_qubits)
            for q in measured_qubits:
                if q < 0 or q >= self.num_qubits:
                    raise ValueError(f"Measured qubit index {q} is out of range "
                                     f"for num_qubits={self.num_qubits}.")

        # Work on a copy so the caller's circuit is reusable
        c = basis_transformation_circuit.copy(deep=True)

        # Measure all qubits; some columns may be "junk" but are never used
        c.add(gates.M(*range(self.num_qubits)))
        
        if seed is not None:
            np.random.seed(seed)
        out = c(initial_state=self.state.copy(), nshots=nshots)
        samples = out.samples()          # shape (nshots, num_qubits), entries in {0,1}
        samples = -2 * samples + 1       # now ±1 everywhere

        # Apply local depolarizing noise model if p_array_LDN is provided
        if self.p_array_LDN is not None:
            samples = apply_local_depolarizing_noise_end_of_circuit(
                samples, self.p_array_LDN, measured_qubits=measured_qubits)

        # Apply global depolarizing noise model if p_GDN is provided
        if self.p_GDN is not None:
            samples = apply_global_depolarizing_noise_end_of_circuit(samples,self.p_GDN)

        # Apply readout noise via stochastic bit flips if provided
        if self.readout_noise_stochastic is not None:
            single_qubit_assignment_matrices = self.readout_noise_stochastic[0]
            pairwise_assignment_matrices = self.readout_noise_stochastic[1]
            seed = self.readout_noise_stochastic[2]
            bit_input = self.readout_noise_stochastic[3]
            samples = apply_stochastic_bit_flips(
                samples,
                single_qubit_assignment_matrices,
                pairwise_assignment_matrices,
                measured_qubits=measured_qubits,
                seed=seed,
                bit_input=bit_input)

        # Apply readout noise via assignment matrix if provided
        if self.readout_noise_A_matrix is not None:
            A_row_col = self.readout_noise_A_matrix[0]
            bit_format = self.readout_noise_A_matrix[1]
            lsb_first = self.readout_noise_A_matrix[2]
            seed = self.readout_noise_A_matrix[3]
            samples = apply_assignment_matrix(
                samples,
                A_row_col,
                measured_qubits,
                bit_format=bit_format,
                lsb_first=lsb_first,
                seed=seed)

        return samples
    
class Sign_estimator():
    
    def __init__(self,measurement_scheme,state,offset,N_reps_exp=1):
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state          = state
        self.offset         = offset
        self.setting_inds   = []
        self.num_settings   = 0
        self._N_reps_exp = None
        self.N_reps_exp = N_reps_exp
        self.outcomes_array = [[] for _ in range(self.N_reps_exp)]
        self.num_outcomes   = 0
    
    def reset(self):
        self.setting_inds   = []
        self.outcomes_array = [[] for _ in range(self.N_reps_exp)]
        self.num_settings   = 0 
        self.num_outcomes   = 0
        self.measurement_scheme.settings_dict  = {}
        self.measurement_scheme.reset()
        
    def clear_outcomes(self):
        self.outcomes_array = [[] for _ in range(self.N_reps_exp)]
        self.num_outcomes   = 0

    @property
    def N_reps_exp(self):
        return self._N_reps_exp
    
    @N_reps_exp.setter
    def N_reps_exp(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("N_reps_exp must be a positive integer.")
        self._N_reps_exp = value
        self.outcomes_array = [[] for _ in range(value)]
        self.clear_outcomes()
        print(f"Updated N_reps_exp to {value}. Cleared outcomes using clear_outcomes().")
        
    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        inds = self.measurement_scheme.find_setting(num_steps)
        self.setting_inds = np.append(self.setting_inds,inds) if len(self.setting_inds)>0 else inds
        self.num_settings += num_steps
        for ind in inds:
            self.measurement_scheme.settings_dict[ind] = 1
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
            outcomes_array = []
            for i in range(self.N_reps_exp):
                outcomes_array.append(np.zeros(num_meas,dtype=int))
            for unique,nshots in zip(*np.unique(recent_settings,return_counts=True)):
                setting = self.measurement_scheme.obs[unique]
                measured_qubits = list(np.where(np.array(list(index_to_string(setting))) != 'I')[0])
                samples = self.state.sample(meas_basis=index_to_string(setting),
                                            nshots=nshots*self.N_reps_exp,
                                            measured_qubits=measured_qubits)
                for i in range(self.N_reps_exp):
                    samples_for_this_exp = samples[i*nshots:(i+1)*nshots,:]
                    outcomes_array[i][recent_settings==unique] = np.prod(samples_for_this_exp,axis=-1)
            for i in range(self.N_reps_exp):
                self.outcomes_array[i] = np.append(self.outcomes_array[i], outcomes_array[i])
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
        energies = []
        for i in range(self.N_reps_exp):
            outcomes = self.outcomes_array[i]
            energy = np.mean(outcomes*sgn[self.setting_inds])*norm
            energies.append(energy)
        return np.array(energies) + self.offset

class Energy_estimator(SteadyStateAllocatorMixin):
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
    def __init__(self, measurement_scheme, state, offset=0, N_reps_exp=1,
                 compat_type='qwc', update_steps=None, qubit_connectivity=None,
                 store_raw_samples=False):
        assert measurement_scheme.num_qubits == state.num_qubits, \
            "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state = state
        self.offset = offset
    
        self.store_raw_samples = bool(store_raw_samples)
        self._raw_samples_dtype = np.int8  # compact storage dtype when enabled
    
        self.raw_samples_dict = {}
        self.obs_samples_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.obs_samples_seen_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.num_settings = 0
        self.num_outcomes = 0
        self.running_avgs_pairs = np.zeros((self.measurement_scheme.num_obs, self.measurement_scheme.num_obs))
        self.running_N = np.zeros(self.measurement_scheme.num_obs, dtype=int)
        self.running_N_pairs = np.zeros((self.measurement_scheme.num_obs,
                                         self.measurement_scheme.num_obs), dtype=int)
        self._basis_circuit_cache = {}
        self._N_reps_exp = None
        self.N_reps_exp = N_reps_exp
    
        self.measurement_scheme.reset()
        self.is_adaptive = measurement_scheme.is_adaptive
        if self.is_adaptive:
            assert update_steps is not None, "update_steps must be provided for adaptive method"
            self.update_steps = np.array(update_steps)
    
        self.compat_type = compat_type
        self.qubit_connectivity = qubit_connectivity
    
        ms_type = getattr(self.measurement_scheme, "commutativity_type", None)
        if ms_type is not None and ms_type != self.compat_type:
            raise ValueError(
                f"Energy_estimator.compat_type='{self.compat_type}' does not match "
                f"measurement_scheme.commutativity_type='{ms_type}'.")
    
        self._steady_reset_state()
    
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
        
    def reset(self):
        self.measurement_scheme.settings_dict = {}
        self.measurement_scheme.settings_buffer = {}
        self.raw_samples_dict = {}
        self.obs_samples_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.obs_samples_seen_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.num_settings, self.num_outcomes = 0, 0
        self.running_N = np.zeros(self.measurement_scheme.num_obs, dtype=int)
        self.running_N_pairs = np.zeros((self.measurement_scheme.num_obs,
                                         self.measurement_scheme.num_obs), dtype=int)
        self.running_avgs = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs))
        self.running_avgs_pairs = np.zeros((self.measurement_scheme.num_obs, self.measurement_scheme.num_obs))
        self.measurement_scheme.N_hits = np.zeros_like(self.measurement_scheme.N_hits)

        """if self.measurement_scheme.save_scheme:
            self.measurement_scheme.all_settings_list = []
            self.measurement_scheme.num_diff_settings_list = []
            self.measurement_scheme.diff_settings_counter = 0"""

        if hasattr(self.measurement_scheme, "N_hits_pairs"):
            self.measurement_scheme.N_hits_pairs = np.zeros((self.measurement_scheme.num_obs,
                                                             self.measurement_scheme.num_obs), dtype=int)

        if hasattr(self.measurement_scheme, "V") and self.measurement_scheme.is_adaptive:
            self.measurement_scheme.V = np.zeros_like(self.measurement_scheme.V)
            self.measurement_scheme.cov_real = np.zeros_like(self.measurement_scheme.V)
            self.measurement_scheme.cov_initialized = False

        if hasattr(self, "outcome_dict"):
            self.outcome_dict = {}

        if hasattr(self.measurement_scheme, "fc_blocks_dict"):
            self.measurement_scheme.fc_blocks_dict = {}

        self._basis_circuit_cache.clear()

        self._steady_reset_state()
        return
    
    def clear_outcomes(self):
        self.measurement_scheme.settings_buffer = self.measurement_scheme.settings_dict.copy()
        self.raw_samples_dict = {}
        self.running_N = np.zeros(self.measurement_scheme.num_obs, dtype=int)
        self.running_N_pairs = np.zeros((self.measurement_scheme.num_obs,
                                         self.measurement_scheme.num_obs), dtype=int)
        self.running_avgs = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs))
        self.running_avgs_pairs = np.zeros((self.measurement_scheme.num_obs, self.measurement_scheme.num_obs))
        self.obs_samples_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.obs_samples_seen_dict_list = [{} for _ in range(self.measurement_scheme.num_obs)]
        self.num_outcomes = 0

        if hasattr(self.measurement_scheme, "V") and self.measurement_scheme.is_adaptive:
            self.measurement_scheme.V = np.zeros((self.measurement_scheme.num_obs,
                                                  self.measurement_scheme.num_obs))
            self.measurement_scheme.cov_real = np.zeros((self.measurement_scheme.num_obs,
                                                         self.measurement_scheme.num_obs))
            self.measurement_scheme.cov_initialized = False

        if hasattr(self, "outcome_dict"):
            self.outcome_dict = {}

        self._basis_circuit_cache.clear()
        return
    
    def propose_next_settings(self, num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        if self.is_adaptive:
            # check that num_steps does not exceed the next threshold 
            # for updating the measurement_scheme internally.
            # If so, limit num_steps accordingly
            thresholds = self.update_steps - self.num_settings
            thresholds = thresholds[thresholds > 0]
            if len(thresholds) > 0:
                max_steps_allowed = thresholds[0]
                if num_steps > max_steps_allowed:
                    print("Warning! Trying to allocate more settings than allowed before updating "
                          "the measurement scheme with outcomes.")
                    print("Num_steps = {0} reduced to {1}. Allocating num_steps={1} instead."
                          .format(num_steps, max_steps_allowed))
                    num_steps = max_steps_allowed
    
        # Collect the settings for these rounds
        settings_rounds = []
        for i in range(num_steps):
            p , _ = self.measurement_scheme.find_setting()
            # p is a sorted array of observable indices measured in this round.
            # The physical basis/circuit is reconstructed later from p and compat_type.
            settings_rounds.append(np.asarray(p, dtype=np.int32))
    
        # Update counter for total number of proposed rounds
        self.num_settings += num_steps
    
        # Update the dict(s) of distinct settings and how often they occur
        order_attr = getattr(self.measurement_scheme, "order", None)
        settings_to_dict(
            settings_rounds,
            self.measurement_scheme.settings_dict,
            self.measurement_scheme.settings_buffer,
            order=order_attr
        )
        return

    def propose_next_settings_measure_and_get_running_avgs_with_empirical_info(self, num_steps=1):
        """Find the <num_steps> next setting(s), taking care of any sampling required 
           to update the empirical covariance matrix.
        """
    
        def measure_and_update(reached_checkpoint):
            self.measure()
            if reached_checkpoint:
                self.get_running_avgs_with_empirical_covariance_update()
                self.measurement_scheme.cov_initialized = True
            else:
                self.get_running_avgs()
    
        while num_steps > 0:
            thresholds = self.update_steps - self.num_settings
            future_thresholds = thresholds[thresholds > 0]
    
            if len(future_thresholds) == 0:
                # No more checkpoints left, finish all remaining steps
                self.propose_next_settings(num_steps)
                measure_and_update(reached_checkpoint=False)
                break
    
            steps_to_next_checkpoint = future_thresholds[0]
    
            if num_steps > steps_to_next_checkpoint:
                # Go up to the next checkpoint
                self.propose_next_settings(steps_to_next_checkpoint)
                measure_and_update(reached_checkpoint=True)
                num_steps -= steps_to_next_checkpoint
            else:
                # Do the remaining steps, possibly ending at a checkpoint
                self.propose_next_settings(num_steps)
                at_checkpoint = (num_steps == steps_to_next_checkpoint)
                measure_and_update(at_checkpoint)
                break
            
    def _append_raw_samples(self, setting_token: bytes, samples) -> None:
        # Efficient storage of raw samples if self.store_raw_samples is True
        if not self.store_raw_samples:
            return
    
        # Compact representation (works for ±1 outcomes and also 0/1 outcomes)
        s = np.asarray(samples, dtype=self._raw_samples_dtype)
    
        if setting_token not in self.raw_samples_dict:
            self.raw_samples_dict[setting_token] = s
        else:
            # Preserves previous behavior (single array per token)
            self.raw_samples_dict[setting_token] = np.vstack((self.raw_samples_dict[setting_token], s))
    
    def _get_basis_cache_entry(self, setting_token: bytes, ms):
        entry = self._basis_circuit_cache.get(setting_token)
        if entry is not None:
            return entry
    
        n = ms.num_qubits
        obs_ids = decode_setting_token(setting_token).astype(int)
    
        if self.compat_type == "qwc":
            setting_int = np.zeros(n, dtype=np.int8)
            for oid in obs_ids:
                o = ms.obs[oid]
                non_id = (o != 0)
                fill = non_id & (setting_int == 0)
                setting_int[fill] = o[fill]
            # Sanity check: Are all observables in setting QWC?
            selected_obs = ms.obs[obs_ids]
            ok = hit_by_batch_numba(selected_obs, setting_int).astype(bool)
            #print("checking compatibility", ok)
            if not np.all(ok):
                raise RuntimeError(
                    "Decoded QWC token contains observables that are not all hit by "
                    "the reconstructed product basis.")
            circuit, _ = generate_qwc_basis_transformation_circuit(
                setting_int, density_matrix=self.state.density_matrix)
            entry = {"type": "qwc", "circuit": circuit, "obs_ids": obs_ids}
    
        elif self.compat_type == "fc":
            paulis = token_to_pauli_list(setting_token, "fc", ms.obs)
            diag = diagonalize_and_map(paulis)
            diag_opt = optimize_clifford_decomposition(
                diagonalization_result=diag,
                n_qubits=n,
                qubit_connectivity=self.qubit_connectivity
            )
            circuit = qibo_circuit_from_gate_list(n, diag_opt["gates"], self.state.density_matrix)
            pauli_to_local = {p: i for i, p in enumerate(paulis)}
            entry = {
                "type": "fc",
                "circuit": circuit,
                "obs_ids": obs_ids,
                "diag_opt": diag_opt,
                "pauli_to_local": pauli_to_local}
    
        elif self.compat_type == "kc":
            FC_blocks = getattr(ms, "fc_blocks_dict", {}).get(setting_token, None)
            if FC_blocks is None:
                raise KeyError(
                    "kC sampling: missing fc_blocks_dict[setting_token]. "
                    "Ensure measurement_scheme.find_setting() stores the partition under the token.")
    
            diag_blocks = []        # list of diagonalization_result dicts (unoptimized)
            diag_blocks_opt = []    # only filled in the per-block optimization branch (connectivity=None)
    
            for block in FC_blocks:
                block = list(map(int, block))
                frag_strings = []
                seen = set()
    
                for obs_idx in obs_ids:
                    row_int = ms.obs[obs_idx]
                    frag_chars = ["I"] * n
                    any_non_id = False
                    for q in block:
                        val = int(row_int[q])
                        if val != 0:
                            any_non_id = True
                            frag_chars[q] = int_to_char[val]
                    if not any_non_id:
                        continue
                    frag_str = "".join(frag_chars)
                    if frag_str in seen:
                        continue
                    seen.add(frag_str)
                    frag_strings.append(frag_str)
    
                if not frag_strings:
                    continue
    
                ok, witness = commute_blockwise(frag_strings)
                if not ok:
                    raise RuntimeError(
                        f"Noncommuting block fragments: {witness} in frag_strings={frag_strings}"
                    )
    
                diag_block = diagonalize_and_map(frag_strings)
                diag_blocks.append(diag_block)
    
            if self.qubit_connectivity is None:
                glist_kc = []
                for diag_block in diag_blocks:
                    diag_block_opt = optimize_clifford_decomposition(
                        diagonalization_result=diag_block,
                        n_qubits=n,
                        qubit_connectivity=None
                    )
                    diag_blocks_opt.append(diag_block_opt)
                    glist_kc.extend(diag_block_opt["gates"])
    
                circuit = qibo_circuit_from_gate_list(n, glist_kc, self.state.density_matrix)
    
                # Build frag_str -> (sign, z_qubits) map from per-block optimized mappings
                block_pauli_map = {}
                for diag_block_opt in diag_blocks_opt:
                    for m in diag_block_opt["mappings"]:
                        pauli_str = m["pauli"]
                        sign = float(m["sign"])
                        z_qubits = np.asarray(m["z_qubits_to_parity"], dtype=np.int32)
    
                        if pauli_str in block_pauli_map:
                            prev_sign, prev_z = block_pauli_map[pauli_str]
                            if prev_sign != sign or prev_z.shape != z_qubits.shape or np.any(prev_z != z_qubits):
                                raise RuntimeError("Inconsistent duplicate mapping for same frag_str.")
                        else:
                            block_pauli_map[pauli_str] = (sign, z_qubits)
    
            else:
                diag_opt_global = optimize_clifford_decomposition_global_setting(
                    diag_results=diag_blocks,
                    n_qubits=n,
                    qubit_connectivity=self.qubit_connectivity,
                    transpilation_trials=40,      # you can parameterize this if desired
                    method_connectivity="greedy", # or "steiner" if you prefer
                    swap_cost=3,
                    seed_base=0)
    
                circuit = qibo_circuit_from_gate_list(n, diag_opt_global["gates"], self.state.density_matrix)
    
                # Build frag_str -> (sign, z_qubits) map from globally optimized mappings
                block_pauli_map = {}
                for m in diag_opt_global["mappings"]:
                    pauli_str = m["pauli"]
                    sign = float(m["sign"])
                    z_qubits = np.asarray(m["z_qubits_to_parity"], dtype=np.int32)
    
                    if pauli_str in block_pauli_map:
                        prev_sign, prev_z = block_pauli_map[pauli_str]
                        if prev_sign != sign or prev_z.shape != z_qubits.shape or np.any(prev_z != z_qubits):
                            raise RuntimeError("Inconsistent duplicate mapping for same frag_str (global routing).")
                    else:
                        block_pauli_map[pauli_str] = (sign, z_qubits)
    
            # Precompute per-observable evaluation data (sign + z_qubits indices)
            kc_obs_signs = np.ones(len(obs_ids), dtype=np.float64)
            kc_obs_zqubits = []  # list of np.ndarray[int32], variable length
    
            for j, global_idx in enumerate(obs_ids):
                row_int = ms.obs[int(global_idx)]
                sgn = 1.0
                z_list = []
    
                for block in FC_blocks:
                    block = list(map(int, block))
                    frag_chars = ["I"] * n
                    any_non_id = False
                    for q in block:
                        val = int(row_int[q])
                        if val != 0:
                            any_non_id = True
                            frag_chars[q] = int_to_char[val]
                    if not any_non_id:
                        continue
    
                    frag_str = "".join(frag_chars)
                    if frag_str not in block_pauli_map:
                        raise RuntimeError(
                            "kC cache build: missing frag_str in block_pauli_map. "
                            "This indicates circuit/mapping mismatch.")
    
                    sign_block, zq = block_pauli_map[frag_str]
                    sgn *= sign_block
                    if zq.size:
                        z_list.append(zq)
    
                kc_obs_signs[j] = sgn
                kc_obs_zqubits.append(np.concatenate(z_list).astype(np.int32, copy=False))
    
            entry = {
                "type": "kc",
                "circuit": circuit,
                "obs_ids": obs_ids,
                "FC_blocks": FC_blocks,
                "block_pauli_map": block_pauli_map,
                "kc_obs_signs": kc_obs_signs,
                "kc_obs_zqubits": kc_obs_zqubits}
    
        else:
            raise ValueError(f"Unsupported compat_type '{self.compat_type}'.")
    
        self._basis_circuit_cache[setting_token] = entry
        return entry
        
    def measure_and_get_running_avgs(self, seed=None):
        num_meas = sum(self.measurement_scheme.settings_buffer.values())
        if num_meas == 0:
            print("Trying to measure more measurement settings than allocated.")
            return
    
        ms = self.measurement_scheme
    
        if self.is_adaptive:
            outcomes = np.zeros((num_meas, ms.num_qubits), dtype=int)
    
        # Accumulators for running averages
        totals = np.zeros((self.N_reps_exp, ms.num_obs))
        counts = np.zeros((self.N_reps_exp, ms.num_obs), dtype=int)
    
        for setting_token, reps in ms.settings_buffer.items():
            reps_eff = self.N_reps_exp * reps
            seed_eff = combine_seed(seed, setting_token) if seed is not None else None
    
            entry = self._get_basis_cache_entry(setting_token, ms)
            obs_ids = entry["obs_ids"]
            circuit = entry["circuit"]


            samples = self.state.sample(
                basis_transformation_circuit=circuit,
                nshots=reps_eff,
                seed=seed_eff,
                measured_qubits=None # Assumed all qubits are measured/noisy
            )
    
            # Optional storage of raw samples
            self._append_raw_samples(setting_token, samples)
    
            if hasattr(self, "outcome_dict"):
                self.outcome_dict[setting_token] = samples
    
            if self.is_adaptive and hasattr(ms, "order"):
                outcomes[ms.order[setting_token]] = samples
        
            if self.compat_type == "qwc":
                for i in obs_ids:
                    o = ms.obs[i]
                    measured_mask = (o != 0)
                    obs_samples = np.prod(samples[:, measured_mask], axis=1)
                    for kk in range(self.N_reps_exp):
                        chunk = obs_samples[kk * reps: (kk + 1) * reps]
                        totals[kk, i] += np.sum(chunk)
                        counts[kk, i] += reps
    
            elif self.compat_type == "fc":
                diag_opt = entry["diag_opt"]
                pauli_to_local = entry["pauli_to_local"]
    
                for m in diag_opt["mappings"]:
                    sign = float(m["sign"])
                    z_qubits = np.asarray(m["z_qubits_to_parity"], dtype=int)
    
                    obs_samples = sign * np.prod(samples[:, z_qubits], axis=1)
    
                    local_idx = pauli_to_local[m["pauli"]]
                    global_idx = obs_ids[local_idx]
    
                    for kk in range(self.N_reps_exp):
                        chunk = obs_samples[kk * reps: (kk + 1) * reps]
                        totals[kk, global_idx] += np.sum(chunk)
                        counts[kk, global_idx] += reps
    
            elif self.compat_type == "kc":
                kc_obs_signs = entry["kc_obs_signs"]
                kc_obs_zqubits = entry["kc_obs_zqubits"]
            
                for j, global_idx in enumerate(obs_ids):
                    zq = kc_obs_zqubits[j]
                    if zq.size == 0:
                        continue  # should not happen unless identity slipped in
                    sign = float(kc_obs_signs[j])
            
                    obs_samples = sign * np.prod(samples[:, zq], axis=1)
            
                    for kk in range(self.N_reps_exp):
                        chunk = obs_samples[kk * reps: (kk + 1) * reps]
                        totals[kk, global_idx] += np.sum(chunk)
                        counts[kk, global_idx] += reps
            else:
                raise ValueError(f"Unsupported compat_type '{self.compat_type}'.")
    
        # Update running averages
        for kk in range(self.N_reps_exp):
            for i in range(ms.num_obs):
                total_reps = self.running_N[i] + counts[kk, i]
                if total_reps > 0:
                    self.running_avgs[kk, i] = (
                        self.running_avgs[kk, i] * self.running_N[i] + totals[kk, i]
                    ) / total_reps
    
        self.running_N += counts[-1, :]
    
        if self.is_adaptive and hasattr(ms, "receive_outcome"):
            for outcome in outcomes:
                ms.receive_outcome(outcome)
    
        self.num_outcomes = self.num_settings
        self.measurement_scheme.settings_buffer = {}
        return
    
    # measure needs to be updated according to measure_and_get_running_avgs,
    # especially because of k-commutativity
    
    def measure(self):
        num_meas = sum(self.measurement_scheme.settings_buffer.values())
        if num_meas == 0:
            print("Trying to measure more measurement settings than allocated.")
            return
    
        ms = self.measurement_scheme
    
        if self.is_adaptive:
            outcomes = np.zeros((num_meas, ms.num_qubits), dtype=int)
    
        for setting_token, reps in ms.settings_buffer.items():
            reps_eff = self.N_reps_exp * reps
    
            if self.compat_type == 'qwc':
                obs_ids = decode_setting_token(setting_token).astype(int)    
                setting_int = np.zeros(ms.num_qubits, dtype=np.int8)
                for oid in obs_ids:
                    o = ms.obs[oid]
                    non_id = (o != 0)
                    fill = non_id & (setting_int == 0)
                    setting_int[fill] = o[fill]
                circuit, _ = generate_qwc_basis_transformation_circuit(
                    setting_int, density_matrix=self.state.density_matrix)    
                measured_qubits = list(range(ms.num_qubits))
                samples = self.state.sample(
                    basis_transformation_circuit=circuit,
                    measured_qubits=measured_qubits,
                    nshots=reps_eff)
    
            elif self.compat_type == 'fc':
                obs_ids = decode_setting_token(setting_token).astype(int)
                paulis = token_to_pauli_list(setting_token, 'fc', ms.obs)
    
                diag = diagonalize_and_map(paulis)
                diag_opt = optimize_clifford_decomposition(
                    diagonalization_result=diag,
                    n_qubits=ms.num_qubits,
                    qubit_connectivity=self.qubit_connectivity)
    
                circuit = qibo_circuit_from_gate_list(
                    ms.num_qubits, diag_opt["gates"],
                    density_matrix=self.state.density_matrix)
                measured_qubits = diag_opt["measured"]
    
                samples = self.state.sample(
                    basis_transformation_circuit=circuit,
                    measured_qubits=measured_qubits,
                    nshots=reps_eff)
    
            elif self.compat_type == 'kc':
                obs_ids = decode_setting_token(setting_token).astype(int)
                FC_blocks = ms.fc_blocks_dict.get(setting_token, [])
    
                glist_kc = []
                fc_pivots = []
                fc_qubits_set = set()
                diag_blocks_opt = []
    
                for block in FC_blocks:
                    block = list(block)
                    for q in block:
                        fc_qubits_set.add(q)
    
                    frag_strings = []
                    seen = set()
                    for obs_idx in obs_ids:
                        row_int = ms.obs[obs_idx]
                        frag_int = np.zeros(ms.num_qubits, dtype=np.int8)
                        for q in block:
                            frag_int[q] = row_int[q]
                        if not np.any(frag_int):
                            continue
                        frag_str = "".join(int_to_char[int(x)] for x in frag_int)
                        if frag_str in seen:
                            continue
                        seen.add(frag_str)
                        frag_strings.append(frag_str)
    
                    if not frag_strings:
                        continue
                    ok, witness = commute_blockwise(frag_strings)
                    if not ok:
                        raise RuntimeError(f"Noncommuting block fragments: {witness} in frag_strings={frag_strings}")
                    diag_block = diagonalize_and_map(frag_strings)
                    diag_block_opt = optimize_clifford_decomposition(
                        diagonalization_result=diag_block,
                        n_qubits=ms.num_qubits,
                        qubit_connectivity=self.qubit_connectivity)
                    diag_blocks_opt.append(diag_block_opt)
    
                    glist_kc.extend(diag_block_opt["gates"])
                    fc_pivots.extend(diag_block_opt["measured"])
    
                qwc_measured_qubits = []
                basis_int = None
    
                if len(obs_ids) > 0:
                    basis_int = clique_measurement_basis(
                        ms.obs,
                        list(obs_ids),
                        which_format="integer",
                        complete_basis=False)
                    basis_int = np.asarray(basis_int, dtype=np.int8)
    
                    for q in fc_qubits_set:
                        basis_int[q] = 0
    
                    for q in range(ms.num_qubits):
                        b = int(basis_int[q])
                        if b == 0:
                            continue
                        if b == 1:
                            glist_kc.append(gates.H(q))
                            qwc_measured_qubits.append(q)
                        elif b == 2:
                            glist_kc.append(S_dagger(q))
                            glist_kc.append(gates.H(q))
                            qwc_measured_qubits.append(q)
                        elif b == 3:
                            qwc_measured_qubits.append(q)
    
                measured_qubits = sorted(set(fc_pivots).union(qwc_measured_qubits))
    
                circuit = qibo_circuit_from_gate_list(
                    ms.num_qubits, glist_kc,
                    density_matrix=self.state.density_matrix)
    
                samples = self.state.sample(
                    basis_transformation_circuit=circuit,
                    measured_qubits=measured_qubits,
                    nshots=reps_eff)
    
            else:
                raise ValueError(
                    f"Unsupported compat_type '{self.compat_type}', "
                    "expected 'qwc', 'fc', or 'kc'.")
    
            # Store raw samples
            if setting_token not in self.raw_samples_dict:
                self.raw_samples_dict[setting_token] = samples
            else:
                self.raw_samples_dict[setting_token] = np.vstack(
                    (self.raw_samples_dict[setting_token], samples))
    
            if hasattr(self, "outcome_dict"):
                self.outcome_dict[setting_token] = samples
    
            if self.is_adaptive and hasattr(ms, "order"):
                outcomes[ms.order[setting_token]] = samples
    
            if self.compat_type == 'qwc':
                for i in obs_ids:
                    o = ms.obs[i]
                    measured_mask = (o != 0)
                    obs_samples = np.prod(samples[:, measured_mask], axis=1)
                    if setting_token in self.obs_samples_dict_list[i]:
                        self.obs_samples_dict_list[i][setting_token] = np.concatenate(
                            (self.obs_samples_dict_list[i][setting_token], obs_samples))
                    else:
                        self.obs_samples_dict_list[i][setting_token] = obs_samples
    
            elif self.compat_type == 'fc':
                maps = diag_opt["mappings"]
                for m in maps:
                    z_qubits = m["z_qubits_to_parity"]
                    sign = m["sign"]
                    obs_samples = float(sign) * np.prod(samples[:, z_qubits], axis=1)
    
                    local_idx = paulis.index(m["pauli"])
                    global_idx = obs_ids[local_idx]
    
                    if setting_token in self.obs_samples_dict_list[global_idx]:
                        self.obs_samples_dict_list[global_idx][setting_token] = np.concatenate(
                            (self.obs_samples_dict_list[global_idx][setting_token], obs_samples))
                    else:
                        self.obs_samples_dict_list[global_idx][setting_token] = obs_samples
    
            elif self.compat_type == 'kc':
                num_cols = samples.shape[1]
                if num_cols == 0:
                    continue
    
                pivot_to_col = {q: j for j, q in enumerate(measured_qubits)}
                all_obs = ms.obs
                block_pauli_map = {}
    
                for diag_block_opt in diag_blocks_opt:
                    maps_block = diag_block_opt["mappings"]
    
                    for m in maps_block:
                        pauli_str = m["pauli"]
                        sign = float(m["sign"])
    
                        z_qubits = np.asarray(m["z_qubits_to_parity"], dtype=int)
                        global_mask = np.zeros(num_cols, dtype=bool)
                        for q in z_qubits:
                            col = pivot_to_col[q]
                            global_mask[col] = True
    
                        if pauli_str in block_pauli_map:
                            prev_sign, prev_mask = block_pauli_map[pauli_str]
                            block_pauli_map[pauli_str] = (
                                prev_sign * sign,
                                np.logical_or(prev_mask, global_mask),
                            )
                        else:
                            block_pauli_map[pauli_str] = (sign, global_mask)
    
                fc_qubits_set = set(fc_qubits_set)
    
                for global_idx in obs_ids:
                    o = all_obs[global_idx]
    
                    obs_sign = 1.0
                    obs_mask = np.zeros(num_cols, dtype=bool)
    
                    for block in FC_blocks:
                        block = list(block)
                        frag_int = np.zeros(ms.num_qubits, dtype=np.int8)
                        for q in block:
                            frag_int[q] = o[q]
                        if not np.any(frag_int):
                            continue
    
                        frag_str = "".join(int_to_char[int(x)] for x in frag_int)
                        if frag_str in block_pauli_map:
                            sign_block, mask_block = block_pauli_map[frag_str]
                            obs_sign *= sign_block
                            obs_mask |= mask_block
    
                    if basis_int is not None:
                        for col_idx, q in enumerate(measured_qubits):
                            if q in fc_qubits_set:
                                continue
                            if o[q] != 0:
                                obs_mask[col_idx] = True
    
                    if not np.any(obs_mask):
                        continue
    
                    obs_samples = obs_sign * np.prod(samples[:, obs_mask], axis=1)
    
                    obs_dict = self.obs_samples_dict_list[global_idx]
                    if setting_token in obs_dict:
                        obs_dict[setting_token] = np.concatenate(
                            (obs_dict[setting_token], obs_samples)
                        )
                    else:
                        obs_dict[setting_token] = obs_samples
    
            else:
                raise ValueError(
                    f"Unsupported compat_type '{self.compat_type}', "
                    "expected 'qwc', 'fc', or 'kc'."
                )
    
        if self.is_adaptive and hasattr(ms, "receive_outcome"):
            for outcome in outcomes:
                ms.receive_outcome(outcome)
    
        self.num_outcomes = self.num_settings
        self.measurement_scheme.settings_buffer = {}
        
        return
    
    def get_running_avgs(self):
        # Initialize accumulators
        totals = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs))
        counts = np.zeros((self.N_reps_exp, self.measurement_scheme.num_obs), dtype=int)
    
        # Loop over observables
        for i, obs_dict in enumerate(self.obs_samples_dict_list):
            seen_dict = self.obs_samples_seen_dict_list[i]
            for setting, samples in obs_dict.items():
                seen = seen_dict.get(setting, 0)
                new_samples = samples[seen:]  # Only process new samples
                if len(new_samples) == 0:
                    continue
                split_chunks = np.split(new_samples, self.N_reps_exp)
                for k, chunk in enumerate(split_chunks):
                    totals[k, i] += np.sum(chunk)
                    counts[k, i] += len(chunk)
                # Update number of samples seen
                seen_dict[setting] = seen + len(new_samples)
    
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
    
    def get_running_avgs_with_empirical_covariance_update(self):
        """
        Update running averages and empirical covariance matrix from new samples.
        Assumes a single experimental repetition (N_reps_exp == 1).
        """
        # Initialize accumulators
        assert self.N_reps_exp == 1, ("getting_running_avgs_with_empirical_covariance_update "
                                      "is only meant to be used with N_reps_exp = 1.")
        totals = np.zeros((self.measurement_scheme.num_obs))
        counts = np.zeros((self.measurement_scheme.num_obs), dtype=int)
        
        co_totals = np.zeros((self.measurement_scheme.num_obs, self.measurement_scheme.num_obs))
        co_counts = np.zeros((self.measurement_scheme.num_obs, self.measurement_scheme.num_obs), dtype=int)
    
        # Loop over observables
        for i, obs_dict_i in enumerate(self.obs_samples_dict_list):
            seen_dict_i = self.obs_samples_seen_dict_list[i]
            for setting, samples_i in obs_dict_i.items():
                # Empirical mean part
                seen_i = seen_dict_i.get(setting, 0)
                new_samples_i = samples_i[seen_i:]  # Only process new samples
                if len(new_samples_i) == 0:
                    continue
                totals[i] += np.sum(new_samples_i)
                counts[i] += len(new_samples_i)
                # Empirical covariances part
                for j in range(i, self.measurement_scheme.num_obs):
                    obs_dict_j = self.obs_samples_dict_list[j]
                    seen_dict_j = self.obs_samples_seen_dict_list[j]
                    if setting in obs_dict_j:
                        seen_j = seen_dict_j.get(setting, 0)
                        samples_j = obs_dict_j[setting]
                        new_samples_j = samples_j[seen_j:]
                        if len(new_samples_j) == 0:
                            continue
                        co_totals[i,j] += np.dot(new_samples_i, new_samples_j)
                        co_totals[j,i] = co_totals[i,j]
                        co_counts[i,j] += len(new_samples_i)
                        co_counts[j,i] = co_counts[i,j]
                                
                # Update number of samples seen
                # NOTE: We only update seen_dict_i here.
                # seen_dict_j is not updated during this loop because observable j
                # will eventually be processed as i in the outer loop,
                # ensuring that each setting's new samples are processed exactly once.
                seen_dict_i[setting] = seen_i + len(new_samples_i)
                    
        # Compute updated running averages
        for i in range(self.measurement_scheme.num_obs):
            total_reps = self.running_N[i] + counts[i]
            if total_reps > 0:
                self.running_avgs[0,i] = (
                    self.running_avgs[0,i] * self.running_N[i] + totals[i]
                ) / total_reps
                
        # Update running counts here because we need updated version to
        # recompute the empirical covariance matrix
        self.running_N += counts

        for i in range(self.measurement_scheme.num_obs):
            for j in range(i, self.measurement_scheme.num_obs):
                total_co_reps = self.running_N_pairs[i,j] + co_counts[i,j]
                if total_co_reps > 0:
                    self.running_avgs_pairs[i,j] = (
                        self.running_avgs_pairs[i,j] * self.running_N_pairs[i,j] + co_totals[i,j]
                        ) / total_co_reps
                    self.running_avgs_pairs[j,i] = self.running_avgs_pairs[i,j]
                    self.measurement_scheme.V[i,j] = (self.running_avgs_pairs[i,j] * total_co_reps 
                                                      - 1/self.num_settings 
                                                      * (self.running_avgs[0,i] * self.running_N[i]) 
                                                      * (self.running_avgs[0,j] * self.running_N[j]) )
                    self.measurement_scheme.V[j,i] = self.measurement_scheme.V[i,j]
                    
        self.measurement_scheme.cov_real = np.ascontiguousarray(((self.measurement_scheme.V + self.measurement_scheme.V.conj().T).real) * 0.5, dtype=np.float64)

        # Update running pairs counts
        self.running_N_pairs += co_counts
    
        return
    
    def get_checkpoint_state(self, *, include_is_hit_array: bool = True, include_N_hits_pairs: bool = False):
        """
        Return a plain dict containing the estimator state needed to resume later.
    
        This is designed for QWC/FC workflows and intentionally excludes raw samples
        and basis-circuit caches to keep checkpoints small.
    
        Parameters
        ----------
        include_is_hit_array : bool
            If True, store measurement_scheme.is_hit_array for faster resume of scheme stats.
            If False, it can be reconstructed from settings_dict on load (slower, smaller checkpoint).
    
        include_N_hits_pairs : bool
            If True, also store measurement_scheme.N_hits_pairs.
            Only use this if you truly need pair-hit guarantees after resume; this can be very large.
        """
        ms = self.measurement_scheme
    
        # Pack token-count dicts in a pickle-free way
        sd_tokens_hex, sd_counts = _pack_token_count_dict(ms.settings_dict)
        sb_tokens_hex, sb_counts = _pack_token_count_dict(ms.settings_buffer)
    
        state = {
            # -------- metadata / safety checks --------
            "ckpt_version": np.array(1, dtype=np.int64),
            "compat_type": np.array(str(self.compat_type)),
            "num_obs": np.array(ms.num_obs, dtype=np.int64),
            "num_qubits": np.array(ms.num_qubits, dtype=np.int64),
            "N_reps_exp": np.array(self.N_reps_exp, dtype=np.int64),
            "offset": np.array(float(self.offset), dtype=np.float64),
    
            # -------- estimator counters --------
            "num_settings": np.array(int(self.num_settings), dtype=np.int64),
            "num_outcomes": np.array(int(self.num_outcomes), dtype=np.int64),
    
            # -------- running averages (critical) --------
            "running_avgs": np.asarray(self.running_avgs, dtype=np.float64),
            "running_N": np.asarray(self.running_N, dtype=np.int64),
    
            # -------- scheme state (critical) --------
            "ms_N_hits": np.asarray(ms.N_hits, dtype=np.int64),
    
            # token-count representation of settings_dict / settings_buffer
            "ms_settings_dict_tokens_hex": sd_tokens_hex,
            "ms_settings_dict_counts": sd_counts,
            "ms_settings_buffer_tokens_hex": sb_tokens_hex,
            "ms_settings_buffer_counts": sb_counts,
    
            # Whether pair hits were being tracked (for safe restore logic)
            "ms_has_N_hits_pairs": np.array(int(hasattr(ms, "N_hits_pairs")), dtype=np.int64),
        }
    
        if include_is_hit_array:
            # Store only the filled rows
            state["ms_is_hit_array"] = np.asarray(ms.is_hit_array, dtype=bool)
    
        if include_N_hits_pairs and hasattr(ms, "N_hits_pairs"):
            state["ms_N_hits_pairs"] = np.asarray(ms.N_hits_pairs, dtype=np.int64)
    
        # Optional scheme history if save_scheme=True
        """if getattr(ms, "save_scheme", False):
            ads_obj = np.empty(len(ms.all_settings_list), dtype=object)
            for i, s in enumerate(ms.all_settings_list):
                arr = np.asarray(s, dtype=np.int32).ravel()
                if arr.size:
                    arr = np.unique(arr)
                    arr.sort()
                ads_obj[i] = arr
    
            state["ms_save_scheme_enabled"] = np.array(1, dtype=np.int64)
            state["ms_all_settings_list"] = ads_obj
            state["ms_num_diff_settings_list"] = np.asarray(ms.num_diff_settings_list, dtype=np.int64)
            state["ms_diff_settings_counter"] = np.array(int(ms.diff_settings_counter), dtype=np.int64)
        else:
            state["ms_save_scheme_enabled"] = np.array(0, dtype=np.int64)"""
            
        # Scheme metadata (helpful for safety checks)
        state["ms_commutativity_type"] = np.array(
            str(getattr(ms, "commutativity_type", "")), dtype=object)
        
        if hasattr(ms, "k"):
            state["ms_k"] = np.array(int(ms.k), dtype=np.int64)
        
        if hasattr(ms, "_kc_top_L"):
            state["ms_kc_top_L"] = np.array(int(ms._kc_top_L), dtype=np.int64)
            
        if getattr(self, "compat_type", None) == "kc" and hasattr(ms, "fc_blocks_dict"):
            # store as hex-keyed dict to avoid bytes issues
            fc_blocks_hex = {
                tok.hex(): [[int(q) for q in block] for block in blocks]
                for tok, blocks in ms.fc_blocks_dict.items()
            }
            state["ms_fc_blocks_dict"] = np.array(fc_blocks_hex, dtype=object)
    
        return state
    
    def load_checkpoint_state(self, state: dict, *, strict: bool = True):
        """
        Restore estimator state from a checkpoint dict produced by get_checkpoint_state().
    
        QWC/FC only (non-adaptive workflows).
        """
        ms = self.measurement_scheme
    
        # ----------------------------
        # 0) Validate metadata
        # ----------------------------
        ckpt_compat = str(np.asarray(state["compat_type"]).item())
        ckpt_num_obs = int(np.asarray(state["num_obs"]).item())
        ckpt_num_qubits = int(np.asarray(state["num_qubits"]).item())
        ckpt_nreps = int(np.asarray(state["N_reps_exp"]).item())
    
        if ckpt_compat != str(self.compat_type):
            raise ValueError(f"Checkpoint compat_type={ckpt_compat} does not match estimator compat_type={self.compat_type}.")
        if ckpt_num_obs != ms.num_obs:
            raise ValueError(f"Checkpoint num_obs={ckpt_num_obs} does not match current num_obs={ms.num_obs}.")
        if ckpt_num_qubits != ms.num_qubits:
            raise ValueError(f"Checkpoint num_qubits={ckpt_num_qubits} does not match current num_qubits={ms.num_qubits}.")
        if ckpt_nreps != self.N_reps_exp:
            raise ValueError(f"Checkpoint N_reps_exp={ckpt_nreps} does not match current N_reps_exp={self.N_reps_exp}.")
    
        # Validate scheme commutativity type if stored
        if "ms_commutativity_type" in state:
            ckpt_ms_type = str(np.asarray(state["ms_commutativity_type"]).item())
            cur_ms_type = str(getattr(ms, "commutativity_type", ""))
            if ckpt_ms_type != cur_ms_type:
                raise ValueError(
                    f"Checkpoint measurement_scheme.commutativity_type={ckpt_ms_type} "
                    f"does not match current={cur_ms_type}.")
        
        # Validate k for kC
        if getattr(self, "compat_type", None) == "kc" and "ms_k" in state and hasattr(ms, "k"):
            ckpt_k = int(np.asarray(state["ms_k"]).item())
            if ckpt_k != int(ms.k):
                raise ValueError(f"Checkpoint k={ckpt_k} does not match current k={ms.k}.")
    
        # 1) Start from a clean state
        self.reset()    
        self.raw_samples_dict = {}
    
        # 2) Restore settings_dict / settings_buffer
        ms.settings_dict = _unpack_token_count_dict(
            np.asarray(state["ms_settings_dict_tokens_hex"]),
            np.asarray(state["ms_settings_dict_counts"])
        )
        ms.settings_buffer = _unpack_token_count_dict(
            np.asarray(state["ms_settings_buffer_tokens_hex"]),
            np.asarray(state["ms_settings_buffer_counts"])
        )
    
        # 3) Restore N_hits (or rebuild)
        if "ms_N_hits" in state:
            ms.N_hits = np.asarray(state["ms_N_hits"], dtype=np.int64).copy()
        else:
            # Fallback: rebuild from settings_dict (slower)
            ms.N_hits = np.zeros(ms.num_obs, dtype=np.int64)
            for tok, reps in ms.settings_dict.items():
                idx = np.asarray(decode_setting_token(tok), dtype=np.int32).ravel()
                if idx.size:
                    idx = np.unique(idx)
                    idx.sort()
                    ms.N_hits[idx] += int(reps)
    
        # 4) Restore N_hits_pairs only if present
        if "ms_N_hits_pairs" in state and hasattr(ms, "N_hits_pairs"):
            ms.N_hits_pairs = np.asarray(state["ms_N_hits_pairs"], dtype=np.int64).copy()
        elif hasattr(ms, "N_hits_pairs"):
            # Keep zeroed unless you want to rebuild (can be expensive)
            ms.N_hits_pairs = np.zeros((ms.num_obs, ms.num_obs), dtype=np.int64)
    
        # 5) Restore running-average accumulators (critical)
        self.running_avgs = np.asarray(state["running_avgs"], dtype=np.float64).copy()
        self.running_N = np.asarray(state["running_N"], dtype=np.int64).copy()
    
        if self.running_avgs.shape != (self.N_reps_exp, ms.num_obs):
            raise ValueError(
                f"running_avgs shape mismatch: got {self.running_avgs.shape}, "
                f"expected {(self.N_reps_exp, ms.num_obs)}."
            )
        if self.running_N.shape != (ms.num_obs,):
            raise ValueError(
                f"running_N shape mismatch: got {self.running_N.shape}, "
                f"expected {(ms.num_obs,)}."
            )
    
        # Pair-running accumulators are not used in current QWC/FC energy estimation path.
        # Keep them zero to save memory and checkpoint size.
        self.running_avgs_pairs = np.zeros((ms.num_obs, ms.num_obs), dtype=np.float64)
        self.running_N_pairs = np.zeros((ms.num_obs, ms.num_obs), dtype=np.int64)
    
        # 6) Restore counters
        self.num_settings = int(np.asarray(state["num_settings"]).item())
        self.num_outcomes = int(np.asarray(state["num_outcomes"]).item())
    
        # Basic consistency checks
        total_in_dict = int(sum(ms.settings_dict.values()))
        total_in_buffer = int(sum(ms.settings_buffer.values()))
    
        if strict and total_in_dict != self.num_settings:
            raise ValueError(
                f"Inconsistent checkpoint: sum(settings_dict.values())={total_in_dict} "
                f"but num_settings={self.num_settings}."
            )
    
        if strict and not (0 <= self.num_outcomes <= self.num_settings):
            raise ValueError(
                f"Inconsistent checkpoint: num_outcomes={self.num_outcomes}, num_settings={self.num_settings}."
            )
    
        if strict and total_in_buffer > total_in_dict:
            raise ValueError(
                f"Inconsistent checkpoint: settings_buffer sum={total_in_buffer} exceeds settings_dict sum={total_in_dict}."
            )
    
        # 7) Restore unique-setting bookkeeping
        # seen_settings can always be reconstructed from settings_dict keys
        ms.seen_settings = set(ms.settings_dict.keys())
    
        if "ms_is_hit_array" in state:
            is_hit = np.asarray(state["ms_is_hit_array"], dtype=bool)
            # Rebuild backing buffer
            n_rows = int(is_hit.shape[0])
            ms._is_hit_cap = max(16, n_rows)
            ms._is_hit_buf = np.empty((ms._is_hit_cap, ms.num_obs), dtype=bool)
            if n_rows > 0:
                if is_hit.shape[1] != ms.num_obs:
                    raise ValueError(
                        f"is_hit_array shape mismatch: got {is_hit.shape}, expected (*, {ms.num_obs})."
                    )
                ms._is_hit_buf[:n_rows] = is_hit
            ms._is_hit_rows_used = n_rows
        else:
            # Fallback: reconstruct is_hit rows from unique tokens (slower)
            unique_tokens = list(ms.settings_dict.keys())
            n_rows = len(unique_tokens)
            ms._is_hit_cap = max(16, n_rows)
            ms._is_hit_buf = np.empty((ms._is_hit_cap, ms.num_obs), dtype=bool)
            ms._is_hit_rows_used = 0
            for tok in unique_tokens:
                idx = np.asarray(decode_setting_token(tok), dtype=np.int32).ravel()
                row = np.zeros(ms.num_obs, dtype=bool)
                if idx.size:
                    idx = np.unique(idx)
                    idx.sort()
                    row[idx] = True
                ms._append_is_hit_row(row)
    
        # Reset/clear transient caches (rebuild lazily)
        ms._hit_outer_cache = {}
        if hasattr(ms, "_fc_row_cache"):
            ms._fc_row_cache = {}
        self._basis_circuit_cache.clear()
        
        # 8) Restore fc_blocks_dict if kC is used
        if getattr(self, "compat_type", None) == "kc":
            if not hasattr(ms, "fc_blocks_dict"):
                ms.fc_blocks_dict = {}
        
            if "ms_fc_blocks_dict" in state:
                fc_blocks_hex = np.asarray(state["ms_fc_blocks_dict"], dtype=object).item()
                ms.fc_blocks_dict = {
                    bytes.fromhex(h): [[int(q) for q in block] for block in blocks]
                    for h, blocks in fc_blocks_hex.items()
                }
            else:
                # Important: don't fail hard if checkpoint was made without this field,
                # but warn because sampling may fail if settings_buffer is non-empty.
                import warnings
                warnings.warn(
                    "kC checkpoint missing 'ms_fc_blocks_dict'. "
                    "Resume may fail if there are buffered settings to measure.",
                    RuntimeWarning,
                )
                ms.fc_blocks_dict = {}
    
        # 9) Restore save_scheme history (optional)
        if getattr(ms, "save_scheme", False) and int(np.asarray(state.get("ms_save_scheme_enabled", 0)).item()) == 1:
            ms.all_settings_list = []
            for arr in np.asarray(state["ms_all_settings_list"], dtype=object):
                a = np.asarray(arr, dtype=np.int32).ravel()
                if a.size:
                    a = np.unique(a)
                    a.sort()
                ms.all_settings_list.append([int(x) for x in a])
    
            ms.num_diff_settings_list = [int(x) for x in np.asarray(state["ms_num_diff_settings_list"], dtype=np.int64)]
            ms.diff_settings_counter = int(np.asarray(state["ms_diff_settings_counter"]).item())
        else:
            if getattr(ms, "save_scheme", False):
                ms.all_settings_list = []
                ms.num_diff_settings_list = []
                ms.diff_settings_counter = 0
    
        # Raw sample dicts are intentionally not restored (memory-frugal checkpointing)
        self.raw_samples_dict = {}
        self.obs_samples_dict_list = [{} for _ in range(ms.num_obs)]
        self.obs_samples_seen_dict_list = [{} for _ in range(ms.num_obs)]
        if hasattr(self, "outcome_dict"):
            self.outcome_dict = {}
    
        return

    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energies
            for all the N_reps_exp repetitions of the experiment. """        
        energy_estimates = self.running_avgs.dot(self.measurement_scheme.w) + self.offset
        
        return energy_estimates 


