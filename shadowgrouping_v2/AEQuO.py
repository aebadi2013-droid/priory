import numpy as np, itertools
from shadowgrouping.poolgenerators import Shadow_Grouping_FC
from shadowgrouping_v2.shadowgrouping_my_dev.full_commutativity import (
    _GF2LinearBasis,
    _export_basis_compact,
    in_span_batch_numba,
)
from shadowgrouping_v2.shadowgrouping_my_dev.qubit_wise_commutativity import sample_obs_batch_from_setting_numba
from shadowgrouping_v2.shadowgrouping_my_dev.helper_functions import encode_setting_token

int_to_char = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

# Source code has been sourced from https://github.com/AndrewJena/VQE_measurement_optimization
# The following class has been modified (and a few hidden member functions added by promoting helper functions) to be compatible with the rest of this API
# Compatability with the original source code has been verified
# No helper function below has been modified

class AEQuO(Shadow_Grouping_FC):
    """
    Greedy AEQuO bucket-filling algorithm with QWC or FC measurement groups.

    The Bayesian allocation logic follows the original AEQuO implementation.
    The commutativity model affects only the compatibility graph, physical
    setting description, hit harvesting, and outcome decoding.

    Parameters
    ----------
    observables, weights, offset
        Hamiltonian Pauli strings, coefficients, and identity offset.

    adaptiveness_L : int, optional
        AEQuO adaptiveness parameter.

    interval_skewness_l : float, optional
        AEQuO update-schedule skewness parameter.

    budget : int, optional
        Predefined number of measurement rounds.

    commutativity_type : {"qwc", "fc"}, optional
        Compatibility relation used to construct the AEQuO LDF clique
        partition. QWC is the backward-compatible default.

    save_scheme : bool, optional
        Retain the round-by-round ShadowGrouping-style setting history.

    compute_N_hits_pairs : bool, optional
        Update the full harvested-observable pair-count matrix each round.

    Notes
    -----
    ``self.cliques`` retains AEQuO's identity-inclusive, non-overlapping LDF
    partition and continues to drive ``S`` and ``V``. Joint +/-1 outcome
    counts are stored in dense integer tensors rather than an object array of
    Python dictionaries.
    ``self.measurement_settings_pool`` stores the unique corresponding physical
    QWC or FC setting records. ``self.cliques_pool`` is aligned one-to-one with
    those records and stores only their sorted, zero-based harvested-observable
    index arrays, matching the allocation-facing pool convention used by the
    other measurement protocols. These harvested hit sets may overlap.

    ``receive_outcomes`` accepts proposal-ordered feedback in batches. In FC
    mode, the optimized path accepts decoded observable-value matrices together
    with their setting tokens, observable indices, and proposal positions.
    ``receive_outcome`` remains available as a backward-compatible one-outcome
    wrapper. Bayesian formulas and checkpoint rules are unchanged.
    """

    def __init__(self, observables, weights, offset,
                 adaptiveness_L=0, interval_skewness_l=0, budget=0,
                 commutativity_type="qwc", save_scheme=False,
                 compute_N_hits_pairs=True):
        super().__init__(
            observables,
            weights,
            weight_function=None,
            save_scheme=save_scheme,
            handle_ties=True,
            compute_N_hits_pairs=compute_N_hits_pairs,
            commutativity_type=commutativity_type,
            initial_ordering_strategy="coefficient",
        )

        # AEQuO's identity-inclusive LDF cliques form a partition, but the
        # physical settings exposed through cliques_pool harvest additional
        # compatible observables and can therefore overlap.
        self.is_overlapping = True

        self.V = np.zeros((self.num_obs + 1, self.num_obs + 1), dtype=float)
        self.offset = offset
        self.__observables_to_AEQuO_list()

        assert isinstance(adaptiveness_L, int), \
            "adaptiveness_L-value has to be integer."
        assert adaptiveness_L >= 0, \
            f"adaptiveness_L-value has to non-negative, but was {adaptiveness_L}."
        assert interval_skewness_l >= 0, \
            f"interval_skewness_l-value has to non-negative, but was {interval_skewness_l}."
        assert isinstance(budget, int), "budget-value has to be integer."
        assert budget >= 0, \
            f"budget-value has to be non-negative, but was {budget}."

        self.shots = budget
        self.L = adaptiveness_L + 1 if self.shots > 0 else 1
        self.l = interval_skewness_l
        self.update_steps = Ll_updates(self.L, self.l, self.shots)

        self.is_adaptive = self.L > 1
        self.is_sampling = self.is_adaptive

        # Exact sufficient statistics for the four joint +/-1 outcomes of
        # every ordered AEQuO-vertex pair. Axis values 0 and 1 represent +1
        # and -1, respectively. The pending tensor contains feedback received
        # since the latest Bayesian checkpoint.
        self.outcome_counts = self._empty_outcome_counts()
        self.pending_outcome_counts = self._empty_outcome_counts()
        self.supports_batched_feedback = True

        if self.commutativity_type == "qwc":
            self.compatibility_graph = qubitwise_commutation_graph(self.paulis)
        else:
            self.compatibility_graph = full_commutation_graph(self.paulis)

        self.cliques = LDF(self.compatibility_graph)
        self.measurement_settings_pool = []
        self.cliques_pool = []
        self.clique_setting_records = {}
        self._build_measurement_settings_pool()
        self.current_setting_record = None

        self.sampled_cliques = []
        self.sampled_cliques_since_update = []
        self.outcomes_since_update = []
        self.update_variance_estimate()

        cliques1, cliques2 = itertools.tee(self.cliques, 2)

        if (not self.update_steps & set(range(1, self.shots))) and not any(
            set(aa1) & set(aa2)
            for aa1, aa2 in itertools.product(cliques1, cliques2)
            if aa1 != aa2
        ):
            self.setting_function = self.non_overlapping_bayes_min_var
            self.clique_counts = [0] * len(self.cliques)
            self.clique_stds = [
                np.sqrt(self.V[clique][:, clique].sum())
                for clique in self.cliques
            ]
        else:
            self.setting_function = self.overlapping_bayes_min_var

        self.S = np.zeros((self.num_obs + 1, self.num_obs + 1), dtype=int)
        self.index_set = set(range(self.num_obs + 1))
        self.update_steps = np.sort(list(self.update_steps))[1:]

        # AEQuO-specific order tracking used by Energy_estimator to restore the
        # chronological order after identical setting tokens are aggregated.
        self.order = {}

        # Retain this historical AEQuO assignment even though the parent class
        # already owns the same attribute.
        self.settings_dict = {}

    def _empty_outcome_counts(self):
        """Return the zeroed identity-inclusive joint-outcome count tensor."""
        p = self.num_obs + 1
        return np.zeros((p, p, 2, 2), dtype=np.int64)

    @property
    def outcome_dict(self):
        """
        Materialize a backward-compatible snapshot of the legacy counters.

        The adaptive implementation itself never constructs this object array.
        Mutating the returned dictionaries does not alter ``outcome_counts``.
        """
        return outcome_counts_to_legacy_dict_array(self.outcome_counts)

    def reset(self):
        """
        Reset execution and Bayesian state while retaining the internal clique
        metadata and the complete public measurement-setting pools.
        """
        super().reset()

        self.outcome_counts = self._empty_outcome_counts()
        self.pending_outcome_counts = self._empty_outcome_counts()
        self.V = np.zeros((self.num_obs + 1, self.num_obs + 1), dtype=float)
        self.S = np.zeros((self.num_obs + 1, self.num_obs + 1), dtype=int)
        self.index_set = set(range(self.num_obs + 1))

        self.sampled_cliques = []
        self.sampled_cliques_since_update = []
        self.outcomes_since_update = []
        self.current_setting_record = None
        self.order = {}
        self.update_variance_estimate()

        if self.setting_function == self.non_overlapping_bayes_min_var:
            self.clique_counts = [0] * len(self.cliques)
            self.clique_stds = [
                np.sqrt(self.V[clique][:, clique].sum())
                for clique in self.cliques
            ]

    def find_setting(self):
        """Select one AEQuO clique, register its QWC/FC setting, and return its hit set."""
        if len(self.sampled_cliques) == self.shots:
            print(
                "Warning! Starting to inquire more samples than predefined "
                f"measurement budget of {self.shots} for this class."
            )
            print(
                "Further measurement settings can be accessed, however, no "
                "update step for the variance estimates is performed."
            )

        if (
            len(self.sampled_cliques) in self.update_steps
            and len(self.sampled_cliques) > 0
        ):
            self.update_variance_estimate()
            self.sampled_cliques_since_update = []
            self.outcomes_since_update = []
            # ``update_variance_estimate`` normally commits and clears this
            # tensor. Clearing again also preserves the historical behavior of
            # discarding incomplete checkpoint feedback after a skipped update.
            self.pending_outcome_counts.fill(0)

        # setting_function retains its original responsibility for AEQuO's
        # Bayesian sample-count state and sampled-clique histories.
        clique = self.setting_function()
        key = tuple(map(int, clique))

        if key not in self.clique_setting_records:
            raise KeyError(f"No measurement-setting record found for clique {clique}.")

        record = self.clique_setting_records[key]
        setting_indices = self._register_setting(
            record["setting_indices"],
            selected_mask=None,
            setting_token=record["setting_token"],
        )

        self.last_generator_indices = record["generator_indices"].copy()
        qwc_setting = record["qwc_setting"]
        self.last_qwc_setting = (
            None if qwc_setting is None else qwc_setting.copy()
        )
        self.current_setting_record = record

        return setting_indices
        
    def overlapping_bayes_min_var(self):
        # The standard version of the setting_function, i.e., the function that 
        # samples a new setting (a.k.a. clique) from a fixed set
        S1 = self.S + 1 # Adding one sample to every single pair of observables
        s = 1/(self.S.diagonal()|(self.S.diagonal()==0))
        s1 = 1/S1.diagonal()
        factor = self.num_obs+1-np.count_nonzero(self.S.diagonal())
        S1[range(self.num_obs+1),range(self.num_obs+1)] = [a if a != 1 else -factor for a in S1.diagonal()]
        V1 = self.V*(self.S*s*s[:,None] - S1*s1*s1[:,None]) # Variances
        V2 = 2*self.V*(self.S*s*s[:,None] - self.S*s*s1[:,None]) # Co-variances
        cliques1 = iter(self.cliques) # this is an iterator of self.cliques
        # The next line is where AEQuO "prioritizes cliques that are 
        # statistically likely to have bigger contributions to the error" (p. 5 of paper)
        clique = sorted(max(cliques1,key=lambda xx : V1[xx][:,xx].sum()+V2[xx][:,list(self.index_set.difference(xx))].sum()))
        self.sampled_cliques.append(clique)
        self.sampled_cliques_since_update.append(clique)
        self.S[np.ix_(clique,clique)] += 1
        
        return clique
    
    def non_overlapping_bayes_min_var(self):
        # The alternative version of the setting_function, i.e., the function that samples a new setting (a.k.a. clique) from a fixed set
        """ If the partition has no overlapping sets, we can speed up the allocation of measurements. """
        # Since there are no overlaps, there are no correlations, so we do not have to worry about covariances.
        # Just pick clique with largest total standard deviation
        # Again, this is where AEQuO "prioritizes cliques that are statistically likely to have bigger contributions to the error" (p. 5 of paper)
        max_index = np.argmax(self.clique_stds)
        clique = self.cliques[max_index]
        self.clique_counts[max_index] += 1
        self.clique_stds[max_index] *= ((self.clique_counts[max_index]-1) or not (self.clique_counts[max_index]-1))/(self.clique_counts[max_index]+1)
        self.S[np.ix_(clique,clique)] += 1#self.Ones[len(clique)]
        self.sampled_cliques.append(clique)
        self.sampled_cliques_since_update.append(clique)
        
        return clique
    
    def update_variance_estimate(self, update_V=True):
        """
        Commit pending sufficient statistics and update the Bayesian graph.

        The integer counts are exactly those accumulated by the former nested
        dictionary loop. Only their storage and evaluation are vectorized.
        """
        num_allocated = len(self.sampled_cliques_since_update)
        num_received = len(self.outcomes_since_update)

        if num_allocated != num_received:
            print("Warning at step {}!".format(len(self.sampled_cliques)))
            print(
                "Not every allocated clique (there are {} allocations since "
                "last update and {} outcomes right now) received an outcome."
                .format(num_allocated, num_received)
            )
            print("Skipping the update.")
            return

        self.outcome_counts += self.pending_outcome_counts
        self.pending_outcome_counts.fill(0)

        if update_V:
            self.V = bayes_variance_graph(
                self.outcome_counts,
                self.coeffs,
            ).adj

        return

    @staticmethod
    def _validate_clique_outcome_matrix(clique, values):
        """Validate and return an ``(num_rounds, clique_size)`` int8 array."""
        clique = np.asarray(clique, dtype=np.int32).reshape(-1)
        values = np.asarray(values)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        if values.ndim != 2 or values.shape[1] != clique.size:
            raise ValueError(
                "Clique outcomes must have shape (num_rounds, clique_size); "
                f"expected second dimension {clique.size}, got "
                f"{values.shape}."
            )

        if not np.all((values == 1) | (values == -1)):
            raise ValueError("AEQuO clique outcomes must contain only +/-1.")

        return values.astype(np.int8, copy=False)

    @staticmethod
    def _accumulate_clique_outcome_counts(target, clique, values):
        """Add one clique's batched four-outcome counts to ``target``."""
        clique = np.asarray(clique, dtype=np.intp).reshape(-1)
        values = AEQuO._validate_clique_outcome_matrix(clique, values)

        positive = (values == 1).astype(np.int64)
        negative = 1 - positive

        counts_pp = positive.T @ positive
        counts_pm = positive.T @ negative
        counts_mp = negative.T @ positive
        counts_mm = negative.T @ negative

        rows, cols = np.ix_(clique, clique)
        target[rows, cols, 0, 0] += counts_pp
        target[rows, cols, 0, 1] += counts_pm
        target[rows, cols, 1, 0] += counts_mp
        target[rows, cols, 1, 1] += counts_mm

    def _process_qwc_outcomes_batch(self, clique, raw_outcomes):
        """Decode many raw QWC samples for one AEQuO clique at once."""
        raw_outcomes = np.asarray(raw_outcomes)
        if raw_outcomes.ndim == 1:
            raw_outcomes = raw_outcomes.reshape(1, -1)

        if raw_outcomes.ndim != 2 or raw_outcomes.shape[1] != self.num_qubits:
            raise ValueError(
                "Batched QWC outcomes must have shape "
                f"(num_rounds, {self.num_qubits}), got "
                f"{raw_outcomes.shape}."
            )
        if not np.all((raw_outcomes == 1) | (raw_outcomes == -1)):
            raise ValueError("QWC AEQuO outcomes must contain only +/-1.")

        includes_identity, clique_obs = self._translate_aequo_clique(clique)
        num_rounds = raw_outcomes.shape[0]
        num_columns = clique_obs.size + int(includes_identity)
        values = np.empty((num_rounds, num_columns), dtype=np.int8)

        column = 0
        if includes_identity:
            values[:, 0] = 1
            column = 1

        for obs_idx in clique_obs:
            support = self.obs[int(obs_idx)] != 0
            values[:, column] = np.prod(
                raw_outcomes[:, support],
                axis=1,
            ).astype(np.int8, copy=False)
            column += 1

        return values

    def _process_fc_outcomes_batch(self, clique, setting_token, obs_ids,
                                   decoded_values):
        """Extract one clique's columns from a decoded FC sample matrix."""
        key = tuple(map(int, clique))
        if key not in self.clique_setting_records:
            raise KeyError(f"No setting record found for AEQuO clique {clique}.")

        record = self.clique_setting_records[key]
        if setting_token != record["setting_token"]:
            raise ValueError(
                "FC batch token does not match the setting generated for "
                f"AEQuO clique {clique}."
            )

        obs_ids = np.asarray(obs_ids, dtype=np.int32).reshape(-1)
        decoded_values = np.asarray(decoded_values)
        if decoded_values.ndim == 1:
            decoded_values = decoded_values.reshape(1, -1)

        if decoded_values.ndim != 2 or decoded_values.shape[1] != obs_ids.size:
            raise ValueError(
                "FC decoded batch must have shape "
                "(num_rounds, len(obs_ids))."
            )
        if np.unique(obs_ids).size != obs_ids.size:
            raise ValueError("FC outcome payload contains duplicate obs_ids.")
        if np.any(obs_ids < 0) or np.any(obs_ids >= self.num_obs):
            raise IndexError(
                "FC outcome payload contains an observable index outside "
                f"[0, {self.num_obs})."
            )
        if not np.all((decoded_values == 1) | (decoded_values == -1)):
            raise ValueError("FC decoded observable values must contain only +/-1.")

        column_by_obs = {
            int(obs_idx): column
            for column, obs_idx in enumerate(obs_ids)
        }
        clique_arr = np.asarray(clique, dtype=np.int32).reshape(-1)
        values = np.empty(
            (decoded_values.shape[0], clique_arr.size),
            dtype=np.int8,
        )

        for column, vertex in enumerate(clique_arr):
            vertex = int(vertex)
            if vertex == 0:
                values[:, column] = 1
                continue

            obs_idx = vertex - 1
            if obs_idx not in column_by_obs:
                raise KeyError(
                    "FC outcome batch is missing observable index "
                    f"{obs_idx}, required by AEQuO clique {clique}."
                )
            values[:, column] = decoded_values[:, column_by_obs[obs_idx]]

        return values

    def _commit_normalized_feedback(self, cliques, normalized_outcomes):
        """Atomically append normalized outcomes and their tensor counts."""
        delta = self._empty_outcome_counts()
        grouped = {}

        for position, (clique, values) in enumerate(
            zip(cliques, normalized_outcomes)
        ):
            if values is None:
                raise RuntimeError(
                    "At least one batched adaptive outcome was not decoded."
                )
            key = tuple(map(int, clique))
            grouped.setdefault(key, []).append(position)

        validated = [None] * len(normalized_outcomes)
        for key, positions in grouped.items():
            clique = np.asarray(key, dtype=np.int32)
            matrix = self._validate_clique_outcome_matrix(
                clique,
                np.asarray([normalized_outcomes[p] for p in positions]),
            )
            self._accumulate_clique_outcome_counts(delta, clique, matrix)
            for row, position in enumerate(positions):
                validated[position] = matrix[row].astype(int).tolist()

        self.pending_outcome_counts += delta
        self.outcomes_since_update.extend(validated)

    def _receive_legacy_outcomes(self, outcomes):
        """Batch legacy raw-QWC or per-round FC feedback in proposal order."""
        if self.commutativity_type == "qwc":
            raw = np.asarray(outcomes)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            num_new = raw.shape[0]
        else:
            outcomes = list(outcomes)
            num_new = len(outcomes)

        start = len(self.outcomes_since_update)
        stop = start + num_new
        if stop > len(self.sampled_cliques_since_update):
            print(
                "Warning at step {}! Trying to feed outcomes for which no "
                "cliques have been allocated yet."
                .format(len(self.sampled_cliques))
            )
            print("Given outcomes have not been incorporated into scheme.")
            return

        cliques = self.sampled_cliques_since_update[start:stop]
        normalized = [None] * num_new
        groups = {}
        for position, clique in enumerate(cliques):
            groups.setdefault(tuple(map(int, clique)), []).append(position)

        for key, positions in groups.items():
            clique = np.asarray(key, dtype=np.int32)
            if self.commutativity_type == "qwc":
                matrix = self._process_qwc_outcomes_batch(
                    clique,
                    raw[positions],
                )
                for row, position in enumerate(positions):
                    normalized[position] = matrix[row]
            else:
                for position in positions:
                    normalized[position] = self._process_fc_outcome(
                        clique,
                        outcomes[position],
                    )

        self._commit_normalized_feedback(cliques, normalized)

    def _receive_setting_batches(self, payload):
        """Receive QWC or FC setting batches with proposal-order positions."""
        if payload.get("format") != "setting_batches":
            raise ValueError("Unsupported batched AEQuO feedback format.")

        num_new = int(payload.get("num_outcomes", -1))
        if num_new < 0:
            raise ValueError("Batched feedback requires nonnegative num_outcomes.")

        start = len(self.outcomes_since_update)
        stop = start + num_new
        if stop > len(self.sampled_cliques_since_update):
            print(
                "Warning at step {}! Trying to feed outcomes for which no "
                "cliques have been allocated yet."
                .format(len(self.sampled_cliques))
            )
            print("Given outcomes have not been incorporated into scheme.")
            return

        cliques = self.sampled_cliques_since_update[start:stop]
        normalized = [None] * num_new
        seen_positions = np.zeros(num_new, dtype=bool)

        for batch in payload.get("batches", []):
            positions = np.asarray(
                batch.get("positions", []),
                dtype=np.int64,
            ).reshape(-1)
            values = np.asarray(batch.get("values", []))

            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.ndim != 2 or values.shape[0] != positions.size:
                raise ValueError(
                    "Every feedback batch must provide one values row per "
                    "proposal-order position."
                )
            if np.any(positions < 0) or np.any(positions >= num_new):
                raise IndexError(
                    "Batched feedback contains an out-of-range proposal "
                    "position."
                )
            if np.unique(positions).size != positions.size:
                raise ValueError(
                    "A feedback batch contains duplicate proposal positions."
                )
            if np.any(seen_positions[positions]):
                raise ValueError(
                    "A proposal position appears in more than one feedback "
                    "batch."
                )

            groups = {}
            for row, position in enumerate(positions):
                key = tuple(map(int, cliques[int(position)]))
                groups.setdefault(key, []).append((row, int(position)))

            batch_format = batch.get("format")
            for key, row_positions in groups.items():
                clique = np.asarray(key, dtype=np.int32)
                rows = [item[0] for item in row_positions]

                if batch_format == "raw_qubit_values_batch":
                    if self.commutativity_type != "qwc":
                        raise ValueError(
                            "Raw-qubit feedback batches are valid only for "
                            "QWC AEQuO."
                        )
                    record = self.clique_setting_records.get(key)
                    if record is None:
                        raise KeyError(
                            f"No setting record found for AEQuO clique {clique}."
                        )
                    if batch.get("setting_token") != record["setting_token"]:
                        raise ValueError(
                            "QWC batch token does not match the setting "
                            f"generated for AEQuO clique {clique}."
                        )
                    matrix = self._process_qwc_outcomes_batch(
                        clique,
                        values[rows],
                    )
                elif batch_format == "observable_values_batch":
                    if self.commutativity_type != "fc":
                        raise ValueError(
                            "Decoded-observable feedback batches are valid "
                            "only for FC AEQuO."
                        )
                    matrix = self._process_fc_outcomes_batch(
                        clique,
                        batch.get("setting_token"),
                        batch.get("obs_ids", []),
                        values[rows],
                    )
                else:
                    raise ValueError(
                        f"Unsupported feedback batch format {batch_format!r}."
                    )

                for local_row, (_, position) in enumerate(row_positions):
                    normalized[position] = matrix[local_row]

            seen_positions[positions] = True

        if num_new and not np.all(seen_positions):
            missing = np.flatnonzero(~seen_positions)
            raise RuntimeError(
                "Batched feedback is missing proposal positions "
                f"{missing.tolist()}."
            )

        self._commit_normalized_feedback(cliques, normalized)

    def receive_outcomes(self, outcomes):
        """
        Receive several adaptive outcomes without changing their chronology.

        ``outcomes`` may be a legacy proposal-ordered sequence or a
        ``format='setting_batches'`` payload produced by ``Energy_estimator``.
        """
        if isinstance(outcomes, dict) and outcomes.get("format") == "setting_batches":
            self._receive_setting_batches(outcomes)
        else:
            self._receive_legacy_outcomes(outcomes)
        return

    def receive_outcome(self, outcome):
        """Backward-compatible wrapper for one adaptive outcome."""
        self.receive_outcomes([outcome])
        return
        
    def __observables_to_AEQuO_list(self):
        """ Helper function that turns the format to the one digestible by AEQuO. """
        pauli_strings = ["I"*self.obs.shape[1]]
        for obs in self.obs:
            string = ""
            for o in obs:
                string += int_to_char[o]
            pauli_strings.append(string)
        self.paulis = string_to_pauli(pauli_strings)
        self.coeffs = np.hstack(([self.offset],self.w))
        return

    @staticmethod
    def _translate_aequo_clique(clique):
        """
        Remove AEQuO's artificial identity vertex and translate the remaining
        vertices to zero-based indices into ``self.obs``.
        """
        clique_arr = np.asarray(clique, dtype=np.int64).reshape(-1)
        includes_identity = bool(np.any(clique_arr == 0))
        obs_indices = clique_arr[clique_arr != 0] - 1
        return includes_identity, obs_indices.astype(np.int32, copy=False)

    def _make_setting_record(self, clique, setting_indices, qwc_setting,
                             generator_indices):
        """Construct one immutable-by-convention AEQuO/ShadowGrouping record."""
        clique_arr = np.asarray(clique, dtype=np.int32).reshape(-1).copy()
        includes_identity, clique_obs = self._translate_aequo_clique(clique_arr)

        setting_indices = np.asarray(
            setting_indices,
            dtype=np.int32,
        ).reshape(-1)
        setting_indices = np.unique(setting_indices)
        setting_indices.sort()

        if setting_indices.size == 0:
            raise RuntimeError(
                f"AEQuO clique {clique_arr.tolist()} generated an empty setting."
            )

        if (
            np.any(setting_indices < 0)
            or np.any(setting_indices >= self.num_obs)
        ):
            raise IndexError(
                "A generated AEQuO setting contains an observable index "
                f"outside [0, {self.num_obs})."
            )

        if not np.all(np.isin(clique_obs, setting_indices)):
            raise RuntimeError(
                "A generated setting does not contain every nonidentity member "
                f"of AEQuO clique {clique_arr.tolist()}."
            )

        generator_indices = np.asarray(
            generator_indices,
            dtype=np.int32,
        ).reshape(-1).copy()

        if (
            np.any(generator_indices < 0)
            or np.any(generator_indices >= self.num_obs)
        ):
            raise IndexError(
                "A generated AEQuO setting contains a generator index outside "
                f"[0, {self.num_obs})."
            )

        if qwc_setting is not None:
            qwc_setting = np.asarray(
                qwc_setting,
                dtype=np.int8,
            ).reshape(-1).copy()
            if qwc_setting.shape != (self.num_qubits,):
                raise ValueError(
                    "qwc_setting must have shape "
                    f"({self.num_qubits},), got {qwc_setting.shape}."
                )

        return {
            "aequo_clique": clique_arr,
            "includes_identity": includes_identity,
            "clique_observable_indices": clique_obs.copy(),
            "setting_indices": setting_indices,
            "setting_token": encode_setting_token(setting_indices),
            "qwc_setting": qwc_setting,
            "generator_indices": generator_indices,
        }

    def _build_qwc_setting_record(self, clique):
        """Build a QWC product setting and harvest every target observable it hits."""
        _, clique_obs = self._translate_aequo_clique(clique)
        if clique_obs.size == 0:
            raise RuntimeError(
                "An identity-only AEQuO clique cannot define a physical QWC "
                "measurement setting."
            )

        setting = np.zeros(self.num_qubits, dtype=np.int8)
        generator_indices = []

        for obs_idx in clique_obs:
            idx = int(obs_idx)
            o = self.obs[idx]
            compatible = np.all((o == 0) | (setting == 0) | (o == setting))
            if not compatible:
                raise RuntimeError(
                    f"AEQuO clique {list(clique)} is not QWC-compatible."
                )

            non_id = o != 0
            sets_new = non_id & (setting == 0)
            if np.any(sets_new):
                generator_indices.append(idx)
            setting[non_id] = o[non_id]

        selected_mask = sample_obs_batch_from_setting_numba(
            self.obs,
            setting,
        ).astype(bool, copy=False)
        setting_indices = np.flatnonzero(selected_mask).astype(np.int32)

        return self._make_setting_record(
            clique=clique,
            setting_indices=setting_indices,
            qwc_setting=setting,
            generator_indices=generator_indices,
        )

    def _build_fc_setting_record(self, clique):
        """Build independent FC generators and harvest their complete target span."""
        _, clique_obs = self._translate_aequo_clique(clique)
        if clique_obs.size == 0:
            raise RuntimeError(
                "An identity-only AEQuO clique cannot define a physical FC "
                "measurement setting."
            )

        # Validate the graph-to-observable index translation explicitly.
        for i, j in itertools.combinations(clique_obs, 2):
            if not self._get_fc_compat_row(int(i))[int(j)]:
                raise RuntimeError(
                    f"AEQuO clique {list(clique)} contains anticommuting "
                    "Pauli strings."
                )

        basis = _GF2LinearBasis(max_bits=2 * self.num_qubits)
        generator_indices = []

        for obs_idx in clique_obs:
            idx = int(obs_idx)
            if basis.add(self._packed[idx]):
                generator_indices.append(idx)

        if not generator_indices:
            raise RuntimeError(
                f"AEQuO clique {list(clique)} produced no nonzero FC generator."
            )

        basis_rows_u64, pivot_bits_u8 = _export_basis_compact(basis)
        selected_mask = in_span_batch_numba(
            self._packed_u64,
            basis_rows_u64,
            pivot_bits_u8,
        ).astype(bool, copy=False)
        setting_indices = np.flatnonzero(selected_mask).astype(np.int32)

        return self._make_setting_record(
            clique=clique,
            setting_indices=setting_indices,
            qwc_setting=None,
            generator_indices=generator_indices,
        )

    def _canonicalize_public_measurement_settings_pool(self, records):
        """
        Build the unique public physical-setting pool from per-clique records.

        ``clique_setting_records`` remains one-to-one with AEQuO's internal
        identity-inclusive cliques. The public ``measurement_settings_pool``
        instead contains one record per distinct harvested hit set, identified
        by its canonical setting token. ``cliques_pool`` stores independent
        copies of those hit sets in the generic allocation-facing format.
        """
        public_records = []
        seen_tokens = set()

        for record in records:
            setting_indices = np.asarray(
                record["setting_indices"],
                dtype=np.int32,
            ).reshape(-1)
            setting_indices = np.unique(setting_indices)
            setting_indices.sort()

            if setting_indices.size == 0:
                raise RuntimeError(
                    "AEQuO generated an empty public measurement setting."
                )

            if (
                np.any(setting_indices < 0)
                or np.any(setting_indices >= self.num_obs)
            ):
                raise IndexError(
                    "An AEQuO public measurement setting contains an "
                    "observable index outside "
                    f"[0, {self.num_obs})."
                )

            token = encode_setting_token(setting_indices)
            stored_token = record.get("setting_token", token)
            if stored_token != token:
                raise RuntimeError(
                    "An AEQuO measurement-setting record has a setting token "
                    "that does not match its canonical observable-index set."
                )

            if token in seen_tokens:
                continue

            seen_tokens.add(token)

            # Keep the rich public record independent of the record used by
            # AEQuO's internal clique-to-setting lookup.
            public_record = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in record.items()
            }
            public_record["setting_indices"] = setting_indices.copy()
            public_record["setting_token"] = token
            public_records.append(public_record)

        if not public_records:
            raise RuntimeError("AEQuO generated an empty measurement-settings pool.")

        covered = np.zeros(self.num_obs, dtype=bool)
        for record in public_records:
            covered[record["setting_indices"]] = True

        if not np.all(covered):
            missing = np.flatnonzero(~covered)
            raise RuntimeError(
                "The final AEQuO measurement-settings pool does not cover every "
                f"observable. Missing indices: {missing.tolist()}."
            )

        self.measurement_settings_pool = public_records
        self.cliques_pool = [
            record["setting_indices"].copy()
            for record in self.measurement_settings_pool
        ]

        if len(self.cliques_pool) != len(self.measurement_settings_pool):
            raise RuntimeError(
                "The AEQuO cliques_pool and measurement_settings_pool are not "
                "aligned."
            )

        for group, record in zip(
            self.cliques_pool,
            self.measurement_settings_pool,
        ):
            if not np.array_equal(group, record["setting_indices"]):
                raise RuntimeError(
                    "An AEQuO cliques_pool entry does not match its aligned "
                    "measurement-setting record."
                )

    def _build_measurement_settings_pool(self):
        """
        Precompute per-clique records and the complete public setting pools.

        The internal record map retains one entry for every fixed AEQuO clique.
        The public pools are canonicalized and deduplicated by harvested hit set.
        """
        records = []
        record_map = {}
        covered_clique_members = np.zeros(self.num_obs, dtype=bool)

        for clique in self.cliques:
            key = tuple(map(int, clique))
            if key in record_map:
                raise RuntimeError(f"Duplicate AEQuO clique encountered: {clique}.")

            if self.commutativity_type == "qwc":
                record = self._build_qwc_setting_record(clique)
            else:
                record = self._build_fc_setting_record(clique)

            records.append(record)
            record_map[key] = record
            covered_clique_members[record["clique_observable_indices"]] = True

        if not records:
            raise RuntimeError("AEQuO generated no per-clique setting records.")

        if not np.all(covered_clique_members):
            missing = np.flatnonzero(~covered_clique_members)
            raise RuntimeError(
                "The AEQuO clique partition does not cover every observable. "
                f"Missing indices: {missing.tolist()}."
            )

        self.clique_setting_records = record_map
        self._canonicalize_public_measurement_settings_pool(records)

    def _clique_to_Pauli_observable(self,clique):
        """
        Backward-compatible QWC helper returning the cached product setting and
        translated nonidentity clique indices.
        """
        if self.commutativity_type != "qwc":
            raise RuntimeError(
                "_clique_to_Pauli_observable is defined only for QWC AEQuO."
            )

        key = tuple(map(int, clique))
        record = self.clique_setting_records[key]
        return (
            record["qwc_setting"].copy(),
            record["clique_observable_indices"].copy(),
        )
    
    def _process_outcome(self,clique,outcome):
        """Dispatch QWC raw-qubit or FC decoded-observable outcome processing."""
        if self.commutativity_type == "qwc":
            return self._process_qwc_outcome(clique, outcome)
        return self._process_fc_outcome(clique, outcome)

    def _process_qwc_outcome(self, clique, outcome):
        """Preserve AEQuO's original QWC per-qubit outcome processing."""
        includes_identity, clique_obs = self._translate_aequo_clique(clique)
        clique_members = self.obs[clique_obs]

        outcome = np.asarray(outcome).reshape(-1)
        if outcome.shape != (self.num_qubits,):
            raise ValueError(
                "A QWC AEQuO outcome must contain one +/-1 value per qubit; "
                f"expected shape ({self.num_qubits},), got {outcome.shape}."
            )
        if not np.all((outcome == 1) | (outcome == -1)):
            raise ValueError("QWC AEQuO outcomes must contain only +/-1 values.")

        tiled = np.repeat(
            outcome.reshape((1, -1)),
            len(clique_members),
            axis=0,
        )
        tiled[clique_members == 0] = 1

        data = [1] if includes_identity else []
        data += list(np.prod(tiled, axis=1).astype(int))
        return data

    def _process_fc_outcome(self, clique, outcome):
        """Extract the selected clique's values from an FC observable payload."""
        if not isinstance(outcome, dict):
            raise TypeError(
                "FC AEQuO requires the decoded observable-value payload "
                "produced by Energy_estimator."
            )

        if outcome.get("format", None) != "observable_values":
            raise ValueError(
                "FC AEQuO outcome payload must have "
                "format='observable_values'."
            )

        key = tuple(map(int, clique))
        if key not in self.clique_setting_records:
            raise KeyError(f"No setting record found for AEQuO clique {clique}.")
        record = self.clique_setting_records[key]

        payload_token = outcome.get("setting_token", None)
        if payload_token != record["setting_token"]:
            raise ValueError(
                "FC outcome payload token does not match the setting generated "
                f"for AEQuO clique {clique}."
            )

        obs_ids = np.asarray(
            outcome.get("obs_ids", []),
            dtype=np.int32,
        ).reshape(-1)
        values = np.asarray(
            outcome.get("values", []),
        ).reshape(-1)

        if obs_ids.shape != values.shape:
            raise ValueError(
                "FC outcome payload obs_ids and values must have equal length."
            )
        if np.unique(obs_ids).size != obs_ids.size:
            raise ValueError("FC outcome payload contains duplicate obs_ids.")
        if np.any(obs_ids < 0) or np.any(obs_ids >= self.num_obs):
            raise IndexError(
                "FC outcome payload contains an observable index outside "
                f"[0, {self.num_obs})."
            )
        if not np.all((values == 1) | (values == -1)):
            raise ValueError("FC decoded observable values must contain only +/-1.")

        value_by_obs = {
            int(obs_idx): int(value)
            for obs_idx, value in zip(obs_ids, values)
        }

        data = []
        for vertex in np.asarray(clique, dtype=np.int32).reshape(-1):
            vertex = int(vertex)
            if vertex == 0:
                data.append(1)
            else:
                obs_idx = vertex - 1
                if obs_idx not in value_by_obs:
                    raise KeyError(
                        "FC outcome payload is missing observable index "
                        f"{obs_idx}, required by AEQuO clique {clique}."
                    )
                data.append(value_by_obs[obs_idx])

        return data
    
    def get_energy(self):
        estim_mean = 0.0
        for i in range(self.paulis.paulis()):
            estim_mean += self.coeffs[i] * naive_Mean(
                self.outcome_counts[i, i]
            )
        return estim_mean

############################################################
############################################################
############ HELPER FUNCTIONS ##############################
############################################################
############################################################

# PAULIS

# a class for storing sets of Pauli operators as pairs of symplectic matrices
class pauli:
    def __init__(self,X,Z):
        # Inputs:
        #     X - (numpy.array) - X-part of Pauli in symplectic form with shape (p,q)
        #     Z - (numpy.array) - Z-part of Pauli in symplectic form with shape (p,q)
        if X.shape != Z.shape:
            raise Exception("X- and Z-parts must have same shape")
        self.X = X
        self.Z = Z

    # check whether self has only X component
    def is_IX(self):
        # Outputs:
        #     (bool) - True if self has only X componenet, False otherwise
        return not np.any(self.Z)

    # check whether self has only Z component 
    def is_IZ(self):
        # Outputs:
        #     (bool) - True if self has only Z componenet, False otherwise
        return not np.any(self.X)

    # check whether the set of Paulis are pairwise commuting on every qubit
    def is_qubitwise_commuting(self):
        # Outputs:
        #     (bool) - True if self is pairwise qubitwise commuting set of Paulis
        p = self.paulis()
        PP = [self.a_pauli(i) for i in range(p)]
        return not any(any((PP[i0].X[0,i2]&PP[i1].Z[0,i2])^(PP[i0].Z[0,i2]&PP[i1].X[0,i2]) for i2 in range(self.qubits())) for i0,i1 in itertools.combinations(range(p),2))

    # pull out the ath Pauli from self
    def a_pauli(self,a):
        # Inputs: 
        #     a - (int) - index of Pauli to be returned
        # Outputs:
        #     (pauli) - the ath Pauli in self
        return pauli(np.array([self.X[a,:]]),np.array([self.Z[a,:]]))

    # count the number of Paulis in self
    def paulis(self):
        # Output: (int)
        return self.X.shape[0]

    # count the number of qubits in self
    def qubits(self):
        # Outputs: (int)
        return self.X.shape[1]

    # delete Paulis indexed by aa
    def delete_paulis_(self,aa):
        # Inputs: 
        #     aa - (list of int)
        if type(aa) is int:
            self.X = np.delete(self.X,aa,axis=0)
            self.Z = np.delete(self.Z,aa,axis=0)
        else:
            for a in sorted(aa,reverse=True):
                self.X = np.delete(self.X,a,axis=0)
                self.Z = np.delete(self.Z,a,axis=0)
        return self

    # return self after deletion of qubits indexed by aa
    def delete_qubits_(self,aa):
        # Inputs: 
        #     aa - (list of int)
        if type(aa) is int:
            self.X = np.delete(self.X,aa,axis=1)
            self.Z = np.delete(self.Z,aa,axis=1)
        else:
            for a in sorted(aa,reverse=True):
                self.X = np.delete(self.X,a,axis=1)
                self.Z = np.delete(self.Z,a,axis=1)

    # return deep copy of self
    def copy(self):
        # Outputs: (pauli)
        X = np.array([[self.X[i0,i1] for i1 in range(self.qubits())] for i0 in range(self.paulis())],dtype=bool)
        Z = np.array([[self.Z[i0,i1] for i1 in range(self.qubits())] for i0 in range(self.paulis())],dtype=bool)
        return pauli(X,Z)

    # print string representation of self
    def print(self):
        sss = pauli_to_string(self)
        if type(sss) is str:
            print(sss)
        else:
            for ss in sss:
                print(ss)

    # print symplectic representation of self
    def print_symplectic(self):
        for i in range(self.paulis()):
            print(''.join(str(int(i1)) for i1 in self.X[i,:]),''.join(str(int(i1)) for i1 in self.Z[i,:]))



# convert a collection of strings (or single string) to a pauli object
def string_to_pauli(sss):
    # Inputs:
    #     sss - (list{str}) or (str) - string representation of Pauli
    # Outputs:
    #     (pauli) - Pauli corresponding to input string(s)
    XDict = {"I":0,"X":1,"Y":1,"Z":0}
    ZDict = {"I":0,"X":0,"Y":1,"Z":1}
    if type(sss) is str:
        X = np.array([[XDict[s] for s in sss]],dtype=bool)
        Z = np.array([[ZDict[s] for s in sss]],dtype=bool)
        return pauli(X,Z)
    else:
        X = np.array([[XDict[s] for s in ss] for ss in sss],dtype=bool)
        Z = np.array([[ZDict[s] for s in ss] for ss in sss],dtype=bool)
        return pauli(X,Z)

# convert a pauli object to a collection of strings (or single string)
def pauli_to_string(P):
    # Inputs:
    #     P - (pauli) - Pauli to be stringified
    # Outputs:
    #     (list{str}) - string representation of Pauli
    X,Z = P.X,P.Z
    ssDict = {(0,0):"I",(0,1):"Z",(1,0):"X",(1,1):"Y"}
    if P.paulis() == 0:
        return ''
    elif P.paulis() == 1:
        return ''.join(ssDict[(X[0,i],Z[0,i])] for i in range(P.qubits()))
    else:
        return [''.join(ssDict[(X[i0,i1],Z[i0,i1])] for i1 in range(P.qubits())) for i0 in range(P.paulis())]

# the symplectic inner product of two pauli objects (each with a single Pauli)
def qubitwise_inner_product(P0,P1):
    # Inputs:
    #     P0 - (pauli) - must have shape (1,q)
    #     P1 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (int) - qubitwise inner product of Paulis modulo 2
    if (P0.paulis() != 1) or (P1.paulis() != 1):
        raise Exception("Qubitwise inner product only works with pair of single Paulis")
    if P0.qubits() != P1.qubits():
        raise Exception("Qubitwise inner product only works if Paulis have same number of qubits")
    return any((P0.X[0,i]&P1.Z[0,i])^(P0.Z[0,i]&P1.X[0,i]) for i in range(P0.qubits()))

# the product of two pauli objects
def pauli_product(P0,P1):
    # Inputs:
    #     P0 - (pauli) - must have shape (1,q)
    #     P1 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (pauli) - product of Paulis
    if P0.paulis() != 1 or P1.paulis() != 1:
        raise Exception("Product can only be calculated for single Paulis")
    return pauli(np.logical_xor(P0.X,P1.X),np.logical_xor(P0.Z,P1.Z))


# GRAPHS

# a class for storing graphs as adjacency matrices
#     since we are dealing with covariance matrices with both vertex and edge weights,
#     this is a suitable format to capture that complexity
class graph:
    # Inputs:
    #     adj_mat - (numpy.array) - (weighted) adjacency matrix of graph
    #     dtype   - (numpy.dtype) - data type of graph weights
    def __init__(self,adj_mat=np.array([]),dtype=float):
        self.adj = adj_mat.astype(dtype)

    # adds a vertex to self
    def add_vertex_(self,c=1):
        # Inputs:
        #     c - (float) - vertex weight
        if len(self.adj) == 0:
            self.adj = np.array([c])
        else:
            m0 = np.zeros((len(self.adj),1))
            m1 = np.zeros((1,len(self.adj)))
            m2 = np.array([[c]])
            self.adj = np.block([[self.adj,m0],[m1,m2]])

    # weight a vertex
    def lade_vertex_(self,a,c):
        # Inputs:
        #     a - (int)   - vertex to be weighted
        #     c - (float) - vertex weight
        self.adj[a,a] = c

    # weight an edge
    def lade_edge_(self,a0,a1,c):
        # Inputs:
        #     a0 - (int)   - first vertex
        #     a1 - (int)   - second vertex
        #     c  - (float) - vertex weight
        self.adj[a0,a1] = c
        self.adj[a1,a0] = c

    # returns a set of the neighbors of a given vertex
    def neighbors(self,a):
        # Inputs:
        #     a - (int) - vertex for which neighbors should be returned
        # Outputs:
        #     (list{int}) - set of neighbors of vertex a
        aa1 = set([])
        for i in range(self.ord()):
            if (a != i) and (self.adj[a,i] != 0):
                aa1.add(i)
        return aa1

    # returns list of all edges in self
    def edges(self):
        # Outputs:
        #     (list{list{int}}) - list of edges in self
        aaa = []
        for i0,i1 in itertools.combinations(range(self.ord()),2):
            if i1 in self.neighbors(i0):
                aaa.append([i0,i1])
        return aaa

    # check whether a collection of vertices is a clique in self
    def clique(self,aa):
        # Inputs:
        #     aa - (list{int}) - list of vertices to be checked for clique
        # Outputs:
        #     (bool) - True if aa is a clique in self; False otherwise
        for i0,i1 in itertools.combinations(aa,2):
            if self.adj[i0,i1] == 0:
                return False
        return True

    # returns the degree of a given vertex
    def degree(self,a):
        # Inputs:
        #     a - (int) - vertex for which degree should be returned
        # Outputs:
        #     (int) - degree of vertex a
        return np.count_nonzero(self.adj[a,:])

    # returns the number of vertices in self
    def ord(self):
        # Outputs:
        #     (int) - number of vertices in self
        return self.adj.shape[0]

    # print adjacency matrix representation of self
    def print(self):
        for i0 in range(self.ord()):
            print('[',end=' ')
            for i1 in range(self.ord()):
                s = self.adj[i0,i1]
                if str(s)[0] == '-':
                    print(f'{self.adj[i0,i1]:.2f}',end=" ")
                else:
                    print(' '+f'{self.adj[i0,i1]:.2f}',end=" ")
            print(']')

    # print self as a list of vertices together with their neighbors
    def print_neighbors(self):
        for i0 in range(self.ord()):
            print(i0,end=": ")
            for i1 in self.neighbors(i0):
                print(i1,end=" ")
            print()

    # return a deep copy of self
    def copy(self):
        # Outputs:
        #     (graph) - deep copy of self
        return graph(np.array([[self.adj[i0,i1] for i1 in range(self.ord())] for i0 in range(self.ord)]))

# returns all non-empty cliques in a graph
def nonempty_cliques(A):
    # Inputs:
    #     A - (graph) - graph for which all cliques should be found
    # Outputs:
    #     (list{list{int}}) - a list containing all non-empty cliques in A
    p = A.ord()
    aaa = set([frozenset([])])
    for i in range(p):
        iset = set([i])
        inter = A.neighbors(i)
        aaa |= set([frozenset(iset|(inter&aa)) for aa in aaa])
    aaa.remove(frozenset([]))
    return list([list(aa) for aa in aaa])

# reduces a clique covering of a graph by removing cliques with lowest weight
def post_process_cliques(A,aaa,k=1):
    # Inputs:
    #     A   - (graph)           - varaince graph from which weights of cliques can be obtained
    #     aaa - (list{list{int}}) - a clique covering of the Hamiltonian
    #     k   - (int)             - number of times each vertex must be covered
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which cover A
    p = A.ord()
    V = A.adj
    s = np.array([sum([i in aa for aa in aaa]) for i in range(p)])
    D = {}
    for aa in aaa:
        D[str(aa)] = V[aa][:,aa].sum()
    aaa1 = aaa.copy()
    aaa1 = list(filter(lambda x : all(a>=(k+1) for a in s[aa]),aaa1))
    while aaa1:
        aa = min(aaa1,key=lambda x : D[str(x)])
        aaa.remove(aa)
        aaa1.remove(aa)
        s -= np.array([int(i in aa) for i in range(p)])
        aaa1 = list(filter(lambda x : all(a>=(k+1) for a in s[aa]),aaa1))
    return aaa

# returns a largest-degree-first clique partition of a graph
def LDF(A):
    # Inputs:
    #     A - (graph) - graph for which partition should be found
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which partition A
    p = A.ord()
    remaining = set(range(p))
    N = {}
    for i in range(p):
        N[i] = A.neighbors(i)
    aaa = []
    while remaining:
        a = max(remaining,key=lambda x : len(N[x]&remaining))
        aa0 = set([a])
        aa1 = N[a]&remaining
        while aa1:
            a2 = max(aa1,key=lambda x : len(N[x]&aa1))
            aa0.add(a2)
            aa1 &= N[a2]
        aaa.append(aa0)
        remaining -= aa0
    return [sorted(list(aa)) for aa in aaa]

# returns the qubitwise commutation graph of a given Pauli
def qubitwise_commutation_graph(P):
    # Inputs:
    #     P - (pauli) - Pauli to check for qubitwise commutation relations
    # Outputs:
    #     (graph) - an edge is weighted 1 if the pair of Paulis qubitwise commute
    p = P.paulis()
    return graph(np.array([[1-qubitwise_inner_product(P.a_pauli(i0),P.a_pauli(i1)) for i1 in range(p)] for i0 in range(p)]))

# returns the full-commutation graph of a given Pauli collection
def full_commutation_graph(P):
    # Inputs:
    #     P - (pauli) - Pauli collection in binary symplectic form
    # Outputs:
    #     (graph) - an edge is weighted 1 iff the pair of Paulis commute fully
    #
    # For rows (x_i,z_i) and (x_j,z_j), full commutativity is equivalent to
    #     x_i . z_j + z_i . x_j = 0 (mod 2).
    X = np.asarray(P.X, dtype=np.uint8)
    Z = np.asarray(P.Z, dtype=np.uint8)

    if X.shape != Z.shape or X.ndim != 2:
        raise ValueError(
            "P.X and P.Z must be two-dimensional arrays with equal shape."
        )

    symplectic_products = (
        X.astype(np.int64) @ Z.astype(np.int64).T
        + Z.astype(np.int64) @ X.astype(np.int64).T
    ) % 2
    adjacency = 1 - symplectic_products

    return graph(adjacency)

# ESTIMATED PHYSICS FUNCTIONS

def outcome_counts_to_legacy_dict_array(outcome_counts):
    """Materialize the historical object-array representation for inspection."""
    counts = np.asarray(outcome_counts)
    if counts.ndim != 4 or counts.shape[2:] != (2, 2):
        raise ValueError(
            "outcome_counts must have shape (p, p, 2, 2)."
        )
    if counts.shape[0] != counts.shape[1]:
        raise ValueError("The first two outcome-count axes must be square.")

    p = counts.shape[0]
    return np.array([
        [{
            (1, 1): int(counts[i, j, 0, 0]),
            (1, -1): int(counts[i, j, 0, 1]),
            (-1, 1): int(counts[i, j, 1, 0]),
            (-1, -1): int(counts[i, j, 1, 1]),
        } for j in range(p)]
        for i in range(p)
    ], dtype=object)


def _single_counts(x):
    """Return (+,+) and (-,-) counts from legacy or numeric storage."""
    if isinstance(x, dict):
        return x[(1, 1)], x[(-1, -1)]

    x = np.asarray(x)
    if x.shape != (2, 2):
        raise ValueError("A numeric single-observable counter must be 2x2.")
    return x[0, 0], x[1, 1]


def _pair_counts(x):
    """Return (++,+-,-+,--) counts from legacy or numeric storage."""
    if isinstance(x, dict):
        return x[(1, 1)], x[(1, -1)], x[(-1, 1)], x[(-1, -1)]

    x = np.asarray(x)
    if x.shape != (2, 2):
        raise ValueError("A numeric pair counter must be 2x2.")
    return x[0, 0], x[0, 1], x[1, 0], x[1, 1]

def naive_Mean(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    x0, x1 = _single_counts(xDict)
    if (x0+x1) == 0:
        return 0
    return (x0-x1)/(x0+x1)

# Bayesian estimation of mean from samples
def bayes_Mean(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    x0, x1 = _single_counts(xDict)
    return (x0-x1)/(x0+x1+2)

# Bayesian estimation of variance from samples
def bayes_Var(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian variance of mean
    x0, x1 = _single_counts(xDict)
    return 4*((x0+1)*(x1+1))/((x0+x1+2)*(x0+x1+3))

# Bayesian estimation of covariance from samples
def bayes_Cov(xyDict,xDict,yDict):
    # Inputs:
    #     xyDict - (Dict) - number of ++/+-/-+/-- outcomes for pair of Paulis
    #     xDict  - (Dict) - number of ++/+-/-+/-- outcomes for first Pauli
    #     xDict  - (Dict) - number of ++/+-/-+/-- outcomes for second Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    xy00, xy01, xy10, xy11 = _pair_counts(xyDict)
    x0, x1 = _single_counts(xDict)
    y0, y1 = _single_counts(yDict)
    p00 = 4*((x0+1)*(y0+1))/((x0+x1+2)*(y0+y1+2))
    p01 = 4*((x0+1)*(y1+1))/((x0+x1+2)*(y0+y1+2))
    p10 = 4*((x1+1)*(y0+1))/((x0+x1+2)*(y0+y1+2))
    p11 = 4*((x1+1)*(y1+1))/((x0+x1+2)*(y0+y1+2))
    return 4*((xy00+p00)*(xy11+p11) - (xy01+p01)*(xy10+p10))/((xy00+xy01+xy10+xy11+4)*(xy00+xy01+xy10+xy11+5))

# approximates the variance graph using Bayesian estimates
def bayes_variance_graph(X,cc):
    # Inputs:
    #     X  - (numpy.array{dict}) - array for tracking measurement outcomes
    #     cc - (list{float})       - coefficients of Hamiltonian
    # Outputs:
    #     (numpy.array{float}) - variance graph calculated with Bayesian estimates
    p = len(cc)
    X_array = np.asarray(X)

    # Backward-compatible legacy path.
    if X_array.ndim != 4:
        return graph(np.array([
            [
                (cc[i0] ** 2) * bayes_Var(X[i0, i0])
                if i0 == i1
                else cc[i0] * cc[i1] * bayes_Cov(
                    X[i0, i1],
                    X[i0, i0],
                    X[i1, i1],
                )
                for i1 in range(p)
            ]
            for i0 in range(p)
        ]))

    if X_array.shape != (p, p, 2, 2):
        raise ValueError(
            "Numeric Bayesian counters must have shape "
            f"({p}, {p}, 2, 2), got {X_array.shape}."
        )

    counts = X_array.astype(np.float64, copy=False)
    xy00 = counts[:, :, 0, 0]
    xy01 = counts[:, :, 0, 1]
    xy10 = counts[:, :, 1, 0]
    xy11 = counts[:, :, 1, 1]

    diagonal = np.arange(p)
    plus = counts[diagonal, diagonal, 0, 0]
    minus = counts[diagonal, diagonal, 1, 1]

    x0 = plus[:, None]
    x1 = minus[:, None]
    y0 = plus[None, :]
    y1 = minus[None, :]

    x_denominator = x0 + x1 + 2.0
    y_denominator = y0 + y1 + 2.0
    marginal_denominator = x_denominator * y_denominator

    p00 = 4.0 * (x0 + 1.0) * (y0 + 1.0) / marginal_denominator
    p01 = 4.0 * (x0 + 1.0) * (y1 + 1.0) / marginal_denominator
    p10 = 4.0 * (x1 + 1.0) * (y0 + 1.0) / marginal_denominator
    p11 = 4.0 * (x1 + 1.0) * (y1 + 1.0) / marginal_denominator

    joint_total = xy00 + xy01 + xy10 + xy11
    covariance = 4.0 * (
        (xy00 + p00) * (xy11 + p11)
        - (xy01 + p01) * (xy10 + p10)
    ) / ((joint_total + 4.0) * (joint_total + 5.0))

    coeffs = np.asarray(cc, dtype=np.float64).reshape(-1)
    adjacency = covariance * coeffs[:, None] * coeffs[None, :]

    variance = 4.0 * (plus + 1.0) * (minus + 1.0) / (
        (plus + minus + 2.0) * (plus + minus + 3.0)
    )
    adjacency[diagonal, diagonal] = coeffs * coeffs * variance

    return graph(adjacency)

# SIMULATION ALGORITHMS

# convert from L,l notation to set of update steps
def Ll_updates(L,l,shots):
    # Inputs:
    #     L     - (int) - number of sections into which shots should be split
    #     l     - (int) - exponential scaling factor for size of sections
    #     shots - (int) - total number of shots required
    # Outputs:
    #     (set{int}) - set containing steps at which algorithm should update
    r0_shots = shots/sum([(1+l)**i for i in range(L)])
    shot_nums = [round(r0_shots*(1+l)**i) for i in range(L-1)]
    shot_nums.append(shots-sum(shot_nums))
    return set([0]+list(itertools.accumulate(shot_nums))[:-1])
