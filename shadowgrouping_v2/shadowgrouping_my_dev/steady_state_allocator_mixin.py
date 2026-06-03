import numpy as np

from .helper_functions import encode_setting_token


class SteadyStateAllocatorMixin:
    """
    Mixin that adds steady-state budget allocation methods to Energy_estimator.

    Expected attributes on `self` (provided by Energy_estimator):
      - self.measurement_scheme
      - self.num_settings
      - self.measurement_scheme.num_obs
      - self.is_adaptive
    """

    def steady_state_budget_allocation(self, num_steps=1, beta=0.01, verbose=False):
        """
        Allocate <num_steps> rounds of settings with the steady-state repetition strategy.

        This version is resumable across multiple calls:
          - before steady state: resumes window generation/checking
          - in steady state: resumes inside the repeated block

        Verbose prints:
          (1) W updates + round index
          (2) Once checking for convergence: W updates + metric vs threshold each window
          (3) End: print find_setting call count and fraction of total rounds

        Notes:
          - Pre-steady: call measurement_scheme.find_setting() one round at a time.
          - In steady-state repeated blocks: do not call find_setting;
                                             update N_hits/N_hits_pairs/settings_dict/buffer.
        """

        if self.is_adaptive:
            raise RuntimeError(
                "steady_state_budget_allocation() is currently intended for non-adaptive schemes "
                "(measurement_scheme.is_adaptive == False)."
            )

        # Configure / reconfigure steady parameters if needed
        self._steady_configure(beta)

        # Track reporting statistics for this call
        find_calls_start = self.steady_find_calls
        rounds_start = self.num_settings

        remaining = int(num_steps)
        if remaining <= 0:
            return

        while remaining > 0:
            # PRE-STEADY REGIME
            if not self.steady_reached:
                self._steady_do_one_round_find_setting(track_window=True, verbose=verbose)
                remaining -= 1

                # Check window boundary condition (depends on phase)
                if self.steady_phase == "FIRST_WINDOW":
                    # End first window once all observables measured >= 1
                    if np.all(self.measurement_scheme.N_hits > 0):
                        self._steady_finalize_window(verbose=verbose)
                else:
                    # End window by size W
                    if self.steady_W > 0 and self.steady_window_pos >= self.steady_W:
                        self._steady_finalize_window(verbose=verbose)

            # STEADY REGIME
            elif self.steady_phase == "STEADY_BLOCK":
                # Consume the NEXT chunk of the current repeated block
                block_remaining = int(self.steady_block_size) - int(self.steady_block_pos)
                if block_remaining < 0:
                    raise RuntimeError("Internal error: steady_block_pos exceeded steady_block_size.")

                # If we are exactly at the end of the block, wrap to the next block
                if block_remaining == 0:
                    self.steady_block_pos = 0
                    block_remaining = int(self.steady_block_size)

                chunk = min(remaining, block_remaining)
                if chunk <= 0:
                    break

                self._steady_apply_block_chunk(chunk)
                remaining -= chunk

                # If block is fully consumed, wrap around to the beginning of the next one
                if self.steady_block_pos == self.steady_block_size:
                    self.steady_block_pos = 0

            else:
                raise RuntimeError(f"Unknown steady_phase: {self.steady_phase}")

        # End-of-call verbose
        if verbose:
            find_calls_end = self.steady_find_calls
            rounds_end = self.num_settings
            called = int(find_calls_end - find_calls_start)
            total = int(rounds_end - rounds_start)
            frac = (called / total) if total > 0 else 0.0
            print(f"[summary] find_setting calls in this allocation: {called} / {total} = {frac:.6f}")

        return

    #  Reset / configure persistent state
    def _steady_reset_state(self):
        # Parameters
        self.steady_beta = None
        self.steady_c_beta = None

        # State machine:
        #   FIRST_WINDOW -> SIZING -> CHECKING -> STEADY_BLOCK
        self.steady_phase = "FIRST_WINDOW"
        self.steady_reached = False

        # Window size and window accumulators
        self.steady_W = 0  # unknown during FIRST_WINDOW
        self.steady_window_pos = 0

        self.steady_win_counts = {}
        self.steady_win_counts_first = None
        self.steady_win_counts_second = None

        self.steady_prev_win_counts = None

        self.steady_S_eff = 1.0
        self.steady_sizing_ok = False  # reserved/diagnostic

        # Convergence debug
        self.steady_last_tv_shared = None
        self.steady_last_tv_thresh = None
        self.steady_last_new_mass = None
        self.steady_last_new_mass_floor = None
        self.steady_last_new_mass_thresh = None

        # Steady block representation (counts per token for ONE block)
        self.steady_block_counts = None
        self.steady_block_size = None
        self.steady_block_updates = None  # optional cache of (idx, count, token)

        # Persistent position inside the current repeated block
        self.steady_block_pos = 0

        # Performance report
        self.steady_find_calls = 0

    def _steady_configure(self, beta: float):
        beta = float(beta)
        if beta <= 0.0 or beta >= 1.0:
            raise ValueError("beta must be in (0,1).")

        # If beta changes mid-run, safest is to reset only allocator state
        if self.steady_beta is None:
            pass
        elif abs(self.steady_beta - beta) > 0:
            self._steady_reset_state()

        self.steady_beta = beta
        self.steady_c_beta = float(np.sqrt(2.0 * np.log(1.0 / beta)))

    #  Tokenization / dict increments
    def _steady_tokenize(self, setting_indices) -> bytes:
        """
        Canonicalize indices (int32, sorted) and encode into bytes token.
        Matches settings_to_dict canonicalization.
        """
        arr = np.asarray(setting_indices, dtype=np.int32).ravel()
        if arr.size > 1:
            arr = np.sort(arr)
        return encode_setting_token(arr)

    def _steady_dict_inc(self, token: bytes, c: int):
        """Increment global settings_dict and settings_buffer by c."""
        sd = self.measurement_scheme.settings_dict
        sb = self.measurement_scheme.settings_buffer
        sd[token] = sd.get(token, 0) + int(c)
        sb[token] = sb.get(token, 0) + int(c)

    #  One-round explicit generation
    def _steady_do_one_round_find_setting(self, track_window=True, verbose=False):
        """
        Do a single round where we CALL find_setting, update:
          - self.num_settings
          - scheme.settings_dict / settings_buffer
          - steady window counts (if track_window=True)

        Note:
          N_hits/N_hits_pairs/is_hit_array/save_scheme updates happen INSIDE find_setting.
        """
        p , _ = self.measurement_scheme.find_setting()
        self.steady_find_calls += 1

        token = self._steady_tokenize(p)
        self._steady_dict_inc(token, 1)

        # Update total rounds counter
        self.num_settings += 1

        if track_window:
            # Window tracking
            self.steady_win_counts[token] = self.steady_win_counts.get(token, 0) + 1

            # Split-window dicts only meaningful once W is known and we are in fixed-size windows
            if self.steady_W > 0:
                half = int(self.steady_W) // 2
                if self.steady_win_counts_first is None:
                    self.steady_win_counts_first = {}
                    self.steady_win_counts_second = {}
                if self.steady_window_pos < half:
                    d = self.steady_win_counts_first
                else:
                    d = self.steady_win_counts_second
                d[token] = d.get(token, 0) + 1

            self.steady_window_pos += 1

        return

    #  Window logic / convergence
    def _steady_compute_S_eff(self, counts: dict, W: int) -> float:
        """
        Renyi-2 effective support:
          S_eff = 1 / sum_t p_t^2 = W^2 / sum_t c_t^2
        """
        if W <= 0 or not counts:
            return 1.0
        s2 = 0.0
        for c in counts.values():
            s2 += float(c) * float(c)
        if s2 <= 0.0:
            return 1.0
        return (float(W) * float(W)) / s2

    def _steady_finalize_window(self, verbose=False):
        """
        Finalize current window:
          - compute S_eff
          - update W (first time, or doubling during sizing)
          - once sizing ok, start checking convergence on next windows
          - once converged, freeze steady block counts
        """
        # window length actually accumulated
        W_cur = int(sum(self.steady_win_counts.values()))
        self.steady_S_eff = self._steady_compute_S_eff(self.steady_win_counts, W_cur)

        # FIRST window: W unknown
        if self.steady_phase == "FIRST_WINDOW":
            M = int(self.measurement_scheme.num_obs)
            W_new = int(max(M, int(np.ceil((self.steady_c_beta ** 2) * self.steady_S_eff))))
            self.steady_W = W_new

            if verbose:
                print(
                    f"[W update] round={self.num_settings}  FIRST_WINDOW done "
                    f"(len={W_cur})  S_eff={self.steady_S_eff:.3f}  -> W={self.steady_W}"
                )

            # Move to sizing fixed-size windows
            self.steady_phase = "SIZING"
            self._steady_reset_window_accumulators()
            return

        # Fixed-size windows
        assert self.steady_W > 0, "Internal error: steady_w must be > 0 outside FIRST_WINDOW."

        required = int(np.ceil((self.steady_c_beta ** 2) * self.steady_S_eff))

        if self.steady_phase == "SIZING":
            # Ensure W large enough for noise floor scaling
            if self.steady_W < required:
                old = int(self.steady_W)
                self.steady_W = int(2 * old)
                if verbose:
                    print(
                        f"[W update] round={self.num_settings}  SIZING "
                        f"S_eff={self.steady_S_eff:.3f}  required={required}  W:{old}->{self.steady_W}"
                    )
                self._steady_reset_window_accumulators()
                return

            # Sizing is OK; start checking from NEXT window
            self.steady_phase = "CHECKING"
            self.steady_prev_win_counts = dict(self.steady_win_counts)
            if verbose:
                print(
                    f"[CHECKING start] round={self.num_settings}  W={self.steady_W}  "
                    f"S_eff={self.steady_S_eff:.3f}  required={required}"
                )
            self._steady_reset_window_accumulators()
            return

        if self.steady_phase == "CHECKING":
            # If complexity increased so much W is too small again, go back to sizing
            if self.steady_W < required:
                old = int(self.steady_W)
                self.steady_W = int(2 * old)
                if verbose:
                    print(
                        f"[W update] round={self.num_settings}  CHECKING->SIZING "
                        f"S_eff={self.steady_S_eff:.3f}  required={required}  W:{old}->{self.steady_W}"
                    )
                self.steady_phase = "SIZING"
                self.steady_prev_win_counts = None
                self._steady_reset_window_accumulators()
                return

            self.steady_win_counts_first = self.steady_win_counts_first or {}
            self.steady_win_counts_second = self.steady_win_counts_second or {}

            curr_counts = dict(self.steady_win_counts)
            prev_counts = dict(self.steady_prev_win_counts or {})

            # Store into attributes expected by _steady_converged()
            self.steady_win_counts = curr_counts
            self.steady_prev_win_counts = prev_counts

            converged = self._steady_converged()

            if verbose:
                tv = float(self.steady_last_tv_shared)
                tvT = float(self.steady_last_tv_thresh)
                nm = float(self.steady_last_new_mass)
                nmT = float(self.steady_last_new_mass_thresh)
                print(
                    f"[CHECK] round={self.num_settings}  W={self.steady_W}  "
                    f"TV_shared={tv:.6g}<= {tvT:.6g}   "
                    f"new_mass={nm:.6g}<= {nmT:.6g}   "
                    f"{'PASS' if converged else 'FAIL'}"
                )

            if converged:
                # Freeze steady block = THIS window’s distribution
                self.steady_reached = True
                self.steady_phase = "STEADY_BLOCK"

                self.steady_block_counts = dict(curr_counts)
                self.steady_block_size = int(sum(curr_counts.values()))
                self._steady_build_block_updates()

                # Persistent progress inside the repeated block
                self.steady_block_pos = 0

                self._steady_reset_window_accumulators()  # no longer used, but harmless

                if verbose:
                    print(
                        f"[STEADY reached] round={self.num_settings}  "
                        f"block_size={self.steady_block_size}  unique={len(self.steady_block_counts)}"
                    )
                return

            # Not converged: shift prev <- curr and continue with next window
            self.steady_prev_win_counts = dict(curr_counts)
            self._steady_reset_window_accumulators()
            return

        raise RuntimeError(f"Unknown steady_phase: {self.steady_phase}")

    def _steady_reset_window_accumulators(self):
        """Start a new window."""
        self.steady_window_pos = 0
        self.steady_win_counts = {}
        self.steady_win_counts_first = {}
        self.steady_win_counts_second = {}

    def _steady_converged(self) -> bool:
        """
        Decide steady-state at end of current window using:
          (1) TV on intersection support (prev vs curr)
          (2) new-setting mass (prev -> curr) compared to split-half noise floor + Hoeffding margin
        """
        prev = getattr(self, "steady_prev_win_counts", None)
        curr = getattr(self, "steady_win_counts", None)
        if prev is None or curr is None:
            return False

        W = int(self.steady_W)
        if W <= 0:
            return False

        # (1) TV on intersection support
        tv_shared = self._steady_tv_on_intersection_support(prev, curr)
        tv_thresh = float(self.steady_c_beta) * np.sqrt(float(self.steady_S_eff) / float(W))
        tv_thresh = min(1.0, tv_thresh)

        # (2) new-setting mass
        new_mass = self._steady_new_setting_mass(prev, curr, W)

        first = getattr(self, "steady_win_counts_first", None)
        second = getattr(self, "steady_win_counts_second", None)
        if first is None or second is None:
            return False

        W2 = int(sum(second.values()))
        if W2 <= 0:
            return False

        new_mass_floor = self._steady_new_mass_between_supports(first, second, W2)

        # Hoeffding margin
        eps = np.sqrt(np.log(2.0 / float(self.steady_beta)) / (2.0 * float(W2)))
        new_mass_thresh = min(1.0, new_mass_floor + eps)

        # Debug storage
        self.steady_last_tv_shared = tv_shared
        self.steady_last_tv_thresh = tv_thresh
        self.steady_last_new_mass = new_mass
        self.steady_last_new_mass_floor = new_mass_floor
        self.steady_last_new_mass_thresh = new_mass_thresh

        return (tv_shared <= tv_thresh) and (new_mass <= new_mass_thresh)

    @staticmethod
    def _steady_tv_on_intersection_support(prev_counts: dict, curr_counts: dict) -> float:
        shared = set(prev_counts.keys()) & set(curr_counts.keys())
        if not shared:
            return 1.0
        prev_tot = sum(prev_counts[t] for t in shared)
        curr_tot = sum(curr_counts[t] for t in shared)
        if prev_tot <= 0 or curr_tot <= 0:
            return 1.0
        tv = 0.0
        for t in shared:
            p = prev_counts[t] / prev_tot
            q = curr_counts[t] / curr_tot
            tv += abs(p - q)
        return 0.5 * tv

    @staticmethod
    def _steady_new_setting_mass(prev_counts: dict, curr_counts: dict, W: int) -> float:
        if W <= 0:
            return 1.0
        shared = set(prev_counts.keys()) & set(curr_counts.keys())
        shared_mass = sum(curr_counts[t] for t in shared) / float(W)
        return max(0.0, 1.0 - shared_mass)

    @staticmethod
    def _steady_new_mass_between_supports(ref_counts: dict, test_counts: dict, test_total: int) -> float:
        if test_total <= 0:
            return 0.0
        ref_support = set(ref_counts.keys())
        shared_mass = 0.0
        for t, c in test_counts.items():
            if t in ref_support:
                shared_mass += c
        shared_mass /= float(test_total)
        return max(0.0, 1.0 - shared_mass)

    #  Steady block helpers
    def _steady_build_block_updates(self):
        """
        Optional cache of per-token index lists:
          steady_block_updates = [(idx_int32_array, count_int, token_bytes), ...]
        Uses measurement_scheme._hit_outer_cache if available; otherwise decodes token.
        """
        assert self.steady_block_counts is not None
        ms = self.measurement_scheme
        updates = []

        # deterministic token order
        for token, c in sorted(self.steady_block_counts.items(), key=lambda kv: kv[0]):
            c = int(c)
            if c <= 0:
                continue

            idx = None
            if hasattr(ms, "_hit_outer_cache"):
                idx = ms._hit_outer_cache.get(token, None)

            if idx is None:
                idx = np.frombuffer(token, dtype=np.int32)

            idx = np.asarray(idx, dtype=np.int32).ravel()
            if idx.size:
                idx = np.unique(idx)
                idx.sort()

            updates.append((idx, c, token))

        self.steady_block_updates = updates

    def _steady_block_prefix_counts(self, L: int) -> dict:
        """
        Deterministic histogram for the first L rounds of the steady block.

        Uses proportional scaling + largest remainders with deterministic tie-breaking.
        """
        L = int(L)
        if L <= 0:
            return {}
        if self.steady_block_counts is None:
            raise RuntimeError("steady_block_counts not initialized.")
        if self.steady_block_size is None or self.steady_block_size <= 0:
            raise RuntimeError("steady_block_size invalid.")
        if L >= self.steady_block_size:
            return dict(self.steady_block_counts)

        return self._steady_scale_counts_to_total(
            self.steady_block_counts,
            target_total=L,
            block_total=self.steady_block_size,
        )

    def _steady_block_interval_counts(self, start: int, stop: int) -> dict:
        """
        Histogram corresponding to the block interval [start, stop),
        computed as prefix(stop) - prefix(start).
        """
        start = int(start)
        stop = int(stop)

        if start < 0 or stop < start or stop > int(self.steady_block_size):
            raise ValueError(
                f"Invalid block interval [{start}, {stop}) for block size {self.steady_block_size}."
            )

        if stop == start:
            return {}

        p_stop = self._steady_block_prefix_counts(stop)
        p_start = self._steady_block_prefix_counts(start)

        out = {}
        keys = set(p_stop.keys()) | set(p_start.keys())
        for token in keys:
            c = int(p_stop.get(token, 0)) - int(p_start.get(token, 0))
            if c > 0:
                out[token] = c
        return out

    def _steady_apply_block_chunk(self, chunk_rounds: int):
        """
        Apply the NEXT chunk_rounds rounds from the current steady block,
        starting at self.steady_block_pos.
        """
        chunk_rounds = int(chunk_rounds)
        if chunk_rounds <= 0:
            return

        start = int(self.steady_block_pos)
        stop = start + chunk_rounds
        if stop > int(self.steady_block_size):
            raise ValueError(
                f"Requested block chunk [{start}, {stop}) exceeds block size {self.steady_block_size}."
            )

        counts = self._steady_block_interval_counts(start, stop)
        self._steady_apply_counts_dict(counts)
        self.steady_block_pos = stop

    def _steady_apply_counts_dict(self, counts: dict):
        """
        Apply a token->count dict once.

        Updates:
          - self.num_settings
          - scheme.settings_dict / settings_buffer
          - scheme.N_hits and scheme.N_hits_pairs (if enabled)
          - save_scheme lists (if enabled)
        """
        ms = self.measurement_scheme
        do_pairs = hasattr(ms, "N_hits_pairs") and getattr(ms, "compute_N_hits_pairs", True)

        total = 0

        # deterministic token order
        for token, c in sorted(counts.items(), key=lambda kv: kv[0]):
            c = int(c)
            if c <= 0:
                continue
            total += c

            self._steady_dict_inc(token, c)

            # decode / retrieve covered observable indices
            idx = None
            if hasattr(ms, "_hit_outer_cache"):
                idx = ms._hit_outer_cache.get(token, None)
            if idx is None:
                idx = np.frombuffer(token, dtype=np.int32)

            idx = np.asarray(idx, dtype=np.int32).ravel()
            if idx.size:
                idx = np.unique(idx)
                idx.sort()
                ms.N_hits[idx] += c
                if do_pairs:
                    ms.N_hits_pairs[np.ix_(idx, idx)] += c

        self.num_settings += total

        if ms.save_scheme and total > 0:
            self._steady_save_scheme_extend_from_counts(counts)

    @staticmethod
    def _steady_scale_counts_to_total(block_counts: dict, target_total: int, block_total: int) -> dict:
        """
        Deterministically scale integer counts to sum to target_total:
          base_t = floor(c_t * target_total / block_total)
          distribute leftover to largest fractional parts
          tie-break deterministically by token bytes

        This determinism is crucial for resumable prefix histograms.
        """
        target_total = int(target_total)
        block_total = int(block_total)

        if target_total <= 0:
            return {}
        if block_total <= 0 or not block_counts:
            return {}

        # deterministic base ordering by token
        items = sorted(block_counts.items(), key=lambda kv: kv[0])

        base = {}
        frac = []

        s = 0
        for token, c in items:
            c = int(c)
            num = c * target_total
            q = num // block_total
            r = num - q * block_total
            q = int(q)
            base[token] = q
            s += q
            frac.append((r, token))

        leftover = target_total - s
        if leftover > 0 and len(frac) > 0:
            # sort by descending fractional remainder, then by token for deterministic tie-breaking
            frac.sort(key=lambda pair: (-pair[0], pair[1]))
            for k in range(leftover):
                token = frac[k % len(frac)][1]
                base[token] += 1

        return {t: c for t, c in base.items() if c > 0}

    def _steady_save_scheme_extend_from_counts(self, counts: dict):
        """
        Extend save_scheme lists from arbitrary token->count dict once (deterministic order).
        """
        ms = self.measurement_scheme
        diff = int(ms.diff_settings_counter)

        for token, c in sorted(counts.items(), key=lambda kv: kv[0]):
            c = int(c)
            if c <= 0:
                continue
            idx = None
            if hasattr(ms, "_hit_outer_cache"):
                idx = ms._hit_outer_cache.get(token, None)
            if idx is None:
                idx = np.frombuffer(token, dtype=np.int32)
            idx_list = list(map(int, np.asarray(idx, dtype=np.int32).tolist()))
            for _r in range(c):
                ms.all_settings_list.append(idx_list)
                ms.num_diff_settings_list.append(diff)

    def _steady_select_gap_anchor(self) -> int:
        raise NotImplementedError("No gap logic in gap-free allocator.")