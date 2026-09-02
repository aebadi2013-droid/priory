import numpy as np, scipy as sp, math
from numba import njit, prange

@njit
def _bernstein_tightest_core(delta, N_hits, N_hits_pairs, w, 
                             is_hit_array, cov_real, tighten_B):
    M = len(w)

    # systematic part: sum |w_i| for never-measured observables
    eps_sys = 0.0
    for i in range(M):
        if N_hits[i] == 0:
            wi = w[i]
            eps_sys += wi if wi >= 0.0 else -wi  # |w[i]|

    # w_eff = w / N (0 if N==0); check if anything measured
    w_eff = np.zeros(M, dtype=np.float64)
    has_samples = False
    for i in range(M):
        Ni = N_hits[i]
        if Ni > 0:
            w_eff[i] = w[i] / Ni
            has_samples = True

    if not has_samples:
        return 0.0, eps_sys

    # Range term B
    abs_sum = 0.0
    for i in range(M):
        wi = w_eff[i]
        abs_sum += wi if wi >= 0.0 else -wi  # sum |w_eff[i]|

    if tighten_B:
        n_settings = is_hit_array.shape[0]
        if n_settings > 0:
            max_sum = 0.0
            for k in range(n_settings):
                s = 0.0
                for i in range(M):
                    if is_hit_array[k, i]:
                        wi = w_eff[i]
                        s += wi if wi >= 0.0 else -wi  # |w_eff[i]|
                if s > max_sum:
                    max_sum = s
            B = 2.0 * max_sum
        else:
            # no settings recorded yet -> fall back to loose bound
            B = 2.0 * abs_sum
    else:
        # loose B: independent of is_hit_array
        B = 2.0 * abs_sum

    # Variance term
    # Var = sum_{ij} w_eff[i] * N_hits_pairs[i,j] * cov_real[i,j] * w_eff[j]
    Var = 0.0
    for i in range(M):
        wi = w_eff[i]
        if wi == 0.0:
            continue
        for j in range(M):
            wj = w_eff[j]
            if wj == 0.0:
                continue
            Var += wi * N_hits_pairs[i, j] * cov_real[i, j] * wj

    if Var < 0.0:  # numerical safety against tiny negatives
        Var = 0.0
    sigma = np.sqrt(Var)

    log_term = -np.log(delta / 2.0)
    eps_stat = sigma * np.sqrt(2.0 * log_term) + (2.0 / 3.0) * B * log_term
    return eps_stat, eps_sys

@njit
def _bernstein_empirical_audibert_core(delta, N_hits, w, is_hit_array, C, tighten_B):
    M = w.shape[0]

    # eps_sys + build hpp (full length, zeros where N_i==0)
    eps_sys = 0.0
    hpp = np.zeros(M, dtype=np.float64)
    has_samples = False

    for i in range(M):
        Ni = N_hits[i]
        wi = w[i]
        wi_abs = wi if wi >= 0.0 else -wi
        if Ni == 0:
            eps_sys += wi_abs
        else:
            hpp[i] = wi_abs / Ni
            has_samples = True

    if not has_samples:
        return 0.0, eps_sys

    # B term
    if tighten_B and is_hit_array.shape[0] > 0:
        K = is_hit_array.shape[0]
        max_sum = 0.0
        for k in range(K):
            s = 0.0
            row = is_hit_array[k]
            for i in range(M):
                if row[i]:
                    s += hpp[i]
            if s > max_sum:
                max_sum = s
        B = max_sum
    else:
        B = 0.0
        for i in range(M):
            B += hpp[i]

    # V_emp = hpp^T C hpp
    V_emp = 0.0
    for i in range(M):
        hi = hpp[i]
        if hi == 0.0:
            continue
        row = C[i]
        s = 0.0
        for j in range(M):
            hj = hpp[j]
            if hj != 0.0:
                s += row[j] * hj
        V_emp += hi * s

    if V_emp < 0.0:
        V_emp = 0.0

    log_term = np.log(3.0 / delta)
    eps_stat = np.sqrt(2.0 * log_term * V_emp) + 3.0 * log_term * B
    return eps_stat, eps_sys

@njit
def _chebyshev_tightest_core(N_hits, N_hits_pairs, w, cov_real):
    """
    Variance of the estimator restricted to observables with N_hits > 0:

        Var = sum_{i,j} (w_i / N_i) N_ij Cov(P_i, P_j) (w_j / N_j).
    """
    M = len(w)

    w_eff = np.zeros(M, dtype=np.float64)

    for i in range(M):
        if N_hits[i] > 0:
            w_eff[i] = w[i] / N_hits[i]

    variance = 0.0

    for i in range(M):
        wi = w_eff[i]
        if wi == 0.0:
            continue

        for j in range(M):
            wj = w_eff[j]
            if wj == 0.0:
                continue

            variance += (
                wi
                * N_hits_pairs[i, j]
                * cov_real[i, j]
                * wj
            )

    return variance


@njit
def _chebyshev_tighter_core(delta, N_hits, N_hits_pairs, w):
    M = N_hits.shape[0]

    hpp = np.zeros(M, dtype=np.float64)
    eps_sys = 0.0
    has_measured = False
    for i in range(M):
        Ni = N_hits[i]
        wi_abs = w[i] if w[i] >= 0.0 else -w[i]
        if Ni > 0:
            hpp[i] = wi_abs / Ni
            has_measured = True
        else:
            eps_sys += wi_abs

    if not has_measured:
        return 0.0, eps_sys

    sigma_sq = 0.0
    for i in prange(M):
        hi = hpp[i]
        if hi == 0.0:
            continue
        row_sum = 0.0
        Ni_row = N_hits_pairs[i]
        for j in range(M):
            hj = hpp[j]
            if hj != 0.0:
                row_sum += Ni_row[j] * hj
        sigma_sq += hi * row_sum

    if sigma_sq < 0.0:  # numerical guard
        sigma_sq = 0.0

    eps_stat = np.sqrt(sigma_sq) / np.sqrt(delta) if sigma_sq > 0.0 else 0.0
    
    return eps_stat, eps_sys

@njit
def _hoeffding_tighter_core(delta, N_hits, N_hits_pairs, w):
    M = N_hits.shape[0]

    # Systematic error: sum |w_i| for unmeasured observables (N_i == 0)
    eps_sys = 0.0
    has_samples = False
    for i in range(M):
        if N_hits[i] == 0:
            wi = w[i]
            eps_sys += wi if wi >= 0.0 else -wi
        else:
            has_samples = True

    # If nothing has been measured, statistical term is zero
    if not has_samples:
        return 0.0, eps_sys

    # Build h_eff = |w|/N where N>0, else 0
    h_eff = np.zeros(M, dtype=np.float64)
    for i in range(M):
        Ni = N_hits[i]
        if Ni > 0:
            wi = w[i]
            if wi < 0.0:
                wi = -wi
            h_eff[i] = wi / Ni

    # Quadratic form q = h_eff^T * N_hits_pairs * h_eff (symmetric; safe real)
    q = 0.0
    for i in range(M):
        hi = h_eff[i]
        if hi == 0.0:
            continue
        s = 0.0
        row = N_hits_pairs[i]
        for j in range(M):
            s += row[j] * h_eff[j]
        q += hi * s

    if q < 0.0:  # numerical safety
        q = 0.0

    # B = 2 * sqrt(q)
    B = 2.0 * math.sqrt(q)

    # eps_stat = B * sqrt( (1/2) * log(2/delta) )
    # use delta/2 inside log for numerical stability => -0.5*log(delta/2)
    log_factor = -0.5 * math.log(delta / 2.0)
    eps_stat = B * math.sqrt(log_factor) if B > 0.0 else 0.0

    return eps_stat, eps_sys

################################################################################
# L1 sampler (i.e., one sample at a time, with importance sampling) guarantees #
################################################################################

def get_Hoeffding_bound_L1_sampler(epsilon, shots, w):
    """ Returns the delta such that the corresponding energy deviation is not larger than epsilon.
        Specifically, delta = 2 exp(- N epsilon^2 / (2 ||h||^2_{l_1}) )
    """
    return 2*np.exp(-0.5*epsilon**2*shots/np.sum(np.abs(w))**2)

def get_Chebyshev_bound_L1_sampler(epsilon, shots, w):
    """ Returns the delta such that the corresponding energy deviation is not larger than epsilon.
        If N = shots = 0, delta is set to the maximum value, 1.
        Else, delta = ||h||^2_{l_1} / (N epsilon^2)
    """
    if shots == 0:
        return 1
    
    return np.sum(np.abs(w))**2 / (shots * epsilon**2)

def get_epsilon_Hoeffding_L1_sampler(delta, shots, w):
    """ Returns the epsilon such that the corresponding Hoeffding bound is not larger than delta.
        If N = shots = 0, epsilon is set equal to the maximum systematic error, the 1-norm of the
        vector that stores the coefficients. For small, nonzero shots, we also check if the Hoeffding
        bound is lower than this systematic error, otherwise we keep the systematic error. For
        the standard case of high shots, epsilon = sqrt{2 log(2/delta)} ||h||_{l_1} / sqrt{N}
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
        
    if shots == 0:
        # Noting has been measured, so assign total systematic error
        return np.sum(np.abs(w))
    
    # For small shots, bound can be greater than total systematic error, so take min
    return min(np.sum(np.abs(w)), np.sqrt(2/shots*np.log(2/delta)) * np.sum(np.abs(w)))

def get_epsilon_Chebyshev_L1_sampler(delta, shots, w):
    """ Returns the epsilon such that the corresponding Chebyshev bound is not larger than delta.
        If N = shots = 0, epsilon is set equal to the maximum systematic error, the 1-norm of the
        vector that stores the coefficients. For small, nonzero shots, we also check if the Chebyshev
        bound is lower than this systematic error, otherwise we keep the systematic error. For
        the standard case of high shots, epsilon = ||h||_{l_1} / sqrt{N delta}
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
        
    if shots == 0:
        # Noting has been measured, so assign total systematic error
        return np.sum(np.abs(w))
    
    # For small shots, bound can be greater than total systematic error, so take min
    return min(1/np.sqrt(shots*delta) * np.sum(np.abs(w)), np.sum(np.abs(w)))

####################################################################
# Overlapping groups (i.e., multiple samples per round) guarantees #
####################################################################

def get_Bernstein_bound(epsilon, N_hits, w):
    """ Returns the delta such that the corresponding energy deviation is not larger than epsilon.
        If at least one of the N_hits is 0, delta is set equal to 1.
        Else, delta = exp(-1/4 ( [ epsilon / (2||h'||_{l_1}) ] - 1)^2 )
    """
    if np.min(N_hits) == 0:
        delta = 1
    else:
        delta = np.exp(-0.25*(epsilon/2/np.sum(np.abs(w)/np.sqrt(N_hits))-1)**2)
    return delta

def N_delta_Bernstein(delta):
    # Square of bottom line of Eq. (29), Supp. Inf. of published version of ShadowGrouping paper
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return 4*(2*np.sqrt(-np.log(delta))+1)**2

def get_epsilon_Bernstein(delta, N_hits, w, warnings=False):
    """ Returns the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = 2*|weights/sqrt(N_hits)| * (1 + 2sqrt(log(1/delta))).
    """
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        w_abs  = np.abs(w[N_hits > 0])
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        norm   = np.sum(w_abs)
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        norm2  = np.sum(w_abs)
        eps_stat = norm * np.sqrt(N_delta_Bernstein(delta))
        if eps_stat > 2*norm*(1+norm/norm2) and warnings:
            print("Warning! Epsilon out of validity range.")
    else:
        eps_stat = 0.0

    return eps_stat + eps_sys
    
def get_epsilon_Bernstein_tighter(delta, N_hits, N_hits_pairs, w, is_hit_array, warnings=False):
    """
    Tighter vector-Bernstein guarantee with per-setting tightening of B.

    epsilon_stat = (1/2)*sqrt(N_delta_Bernstein(delta)) * sigma,
    where sigma^2 = 4 * h''^T N_hits_pairs h'',  h''_i = |w_i| / N_i (for N_i>0),
    and B = 4 * max_k sum_i 1{setting k hits i} * h''_i  (for diagnostics/range checks).

    Systematic term: sum_{N_i=0} |w_i|.
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # Systematic error (never-measured)
    eps_sys = np.sum(np.abs(w[N_hits == 0]))

    mask = (N_hits > 0)
    if not np.any(mask):
        # no statistical part if nothing measured
        return eps_sys

    hpp = np.abs(w[mask]) / N_hits[mask]

    # ---- B term (tightened via per-setting is_hit rows) ----
    if is_hit_array is not None and is_hit_array.size > 0:
        is_hit_sel = is_hit_array[:, mask].astype(int, copy=False)  # shape: (K, m)
        if is_hit_sel.shape[0] > 0:
            settings_weights = is_hit_sel @ hpp
            B = 4.0 * float(np.max(settings_weights)) if settings_weights.size > 0 else 0.0
        else:
            # no settings recorded yet: conservative upper bound
            B = 4.0 * float(np.sum(hpp))
    else:
        # no settings recorded yet: conservative upper bound
        B = 4.0 * float(np.sum(hpp))

    # ---- sigma from pair counts (accounts for repetitions) ----
    N_pairs_sel = N_hits_pairs[np.ix_(mask, mask)]
    sigma_sq = 4.0 * float(hpp @ (N_pairs_sel @ hpp))
    if sigma_sq < 0.0:  # numerical safety
        sigma_sq = 0.0
    sigma = np.sqrt(sigma_sq)

    eps_stat = 0.5 * sigma * np.sqrt(N_delta_Bernstein(delta)) if sigma > 0.0 else 0.0

    # Optional diagnostics (avoid divide-by-zero)
    if B > 0.0:
        validity_range = sigma + 3.0 * sigma * sigma / B
        delta_min = np.exp(-9.0 * sigma * sigma / (4.0 * B * B))
        if eps_stat > validity_range and warnings:
            print("Warning! Epsilon out of validity range. Either increase number of "
                  "measurement rounds or increase inconfidence bound delta.")
            print("Range of validity of Theorem 3 in terms of epsilon: [%.6g, %.6g]" % (sigma, validity_range))
            print("Relevant range of inconfidence delta: [%.6g, 1]" % (delta_min))

    return eps_stat + eps_sys

def N_delta_Bernstein_no_restricted_validity(delta):
    # Similar approach to N_delta_Bernstein, but a slightly more involved expression
    # due to the presence of both range and variance terms
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    variance_factor = 1 + np.sqrt(-2*np.log(delta))
    range_factor = -8*np.log(delta)/3
    alpha_delta = variance_factor + np.sqrt(variance_factor**2 + range_factor)
    N_delta = alpha_delta**2
    
    return N_delta

def get_epsilon_Bernstein_no_restricted_validity(delta, N_hits, w):
    """ Returns the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = sigma * [1 + sqrt(2 log(1/delta)) ] + 2B/3 * log(1/delta).
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        w_abs  = np.abs(w[N_hits > 0])
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        sigma  = 2 * np.sum(w_abs) # Eq. (25), Supp. Inf. of published version of ShadowGrouping paper
    
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        B = 4 * np.sum(w_abs) # Eq. (23), Supp. Inf. of published version of ShadowGrouping paper
                              # and extra factor of 2 from Eq. (14) as well
        eps_stat = sigma * ( 1 + np.sqrt(-2*np.log(delta)) ) - 2*B*np.log(delta)/3
    else:
        eps_stat = 0.0

    return eps_stat + eps_sys

def get_epsilon_Bernstein_tighter_no_restricted_validity(
    delta, N_hits, N_hits_pairs, w, is_hit_array):
    """Bernstein (tighter, no restricted validity) with per-setting B tightening.

    epsilon_stat = sigma * [1 + sqrt(2 log(1/delta))] + (2/3) * B * log(1/delta),
    where:
      - h''_i = |w_i| / N_i for N_i>0 (else excluded),
      - sigma^2 = 4 * h''^T N_hits_pairs h'',
      - B = 4 * max_k sum_i 1{setting k hits i} * h''_i  (tightened via is_hit_array).

    The systematic term is sum_{N_i=0} |w_i|.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")

    # Systematic error: never-measured observables
    eps_sys = float(np.sum(np.abs(w[N_hits == 0])))

    # Mask to measured observables
    mask = (N_hits > 0)
    if not np.any(mask):
        # nothing measured yet -> no statistical term
        return eps_sys

    # h'' on measured coordinates
    hpp = np.abs(w[mask]) / N_hits[mask]   # shape (m,)

    # ---- Tightened B using per-setting compatibility ----
    if is_hit_array is not None and is_hit_array.size > 0:
        # restrict columns to measured observables
        is_hit_sel = is_hit_array[:, mask].astype(int, copy=False)
        if is_hit_sel.shape[0] > 0:
            settings_weights = is_hit_sel @ hpp
            B = 4.0 * float(np.max(settings_weights)) if settings_weights.size > 0 else 0.0
        else:
            # no settings recorded yet (K=0): conservative upper bound
            B = 4.0 * float(np.sum(hpp))
    else:
        # no settings recorded yet: conservative upper bound
        B = 4.0 * float(np.sum(hpp))

    # ---- sigma via pair counts ----
    N_pairs_sel = N_hits_pairs[np.ix_(mask, mask)]
    sigma_sq = 4.0 * float(hpp @ (N_pairs_sel @ hpp))
    if sigma_sq < 0.0:
        sigma_sq = 0.0
    sigma = np.sqrt(sigma_sq)

    log_term = -np.log(delta)
    eps_stat = sigma * (1.0 + np.sqrt(2.0 * log_term)) + (2.0 / 3.0) * B * log_term

    return eps_stat + eps_sys

def N_delta_Bernstein_scalar(delta):
    # Similar approach to N_delta_Bernstein; just different numerical factors
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return -4*np.log(delta/2)

def get_epsilon_Bernstein_scalar(delta, N_hits, w, warnings=False):
    """ Returns the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = |weights/sqrt(N_hits)| * 2sqrt(log(2/delta)).
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        w_abs  = np.abs(w[N_hits > 0])
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        sigma  = np.sum(w_abs) # No factor of 2 because there is no need to center RVs
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        B = 2*np.sum(w_abs) # Saved factor of 2 relative to vector Bernstein guarantee
                            # because Lemma 6.16 from Ledoux & Talagrand does not apply
                            # but still need to bound |X_i - \EE[X_i]| even without centering
                            # RVs (see Eq. (13) from v5.3 of shared notes on Bernstein inequalities),
                            # so other factor of 2 cannot be saved
        
        eps_stat = sigma*2*np.sqrt(-np.log(delta/2))
        
        validity_range = 3*sigma**2/B
        delta_min = np.exp(-9*sigma**2/(4*B**2))
        
        if eps_stat > validity_range and warnings:
            print("Warning! Epsilon out of validity range. Either increase number of measurement rounds or increase inconfidence bound delta.")
            print("Range of validity of Theorem 3 in terms of epsilon: [%f, %f]" %(0, validity_range) )
            print("Relevant range of inconfidence delta: [%f, 1]" %(delta_min))
    else:
        eps_stat = 0.0
    
    return eps_stat + eps_sys

def get_epsilon_Bernstein_scalar_tighter(delta, N_hits, N_hits_pairs, w,
                                         is_hit_array, warnings=False, tighten_B=True):
    """Scalar Bernstein (tighter) with optional per-setting B tightening and N_i=0 handling.

    epsilon_stat = sigma * 2 * sqrt(log(2/delta)),
    where:
      - h''_i = |w_i| / N_i for N_i>0 (excluded otherwise),
      - sigma^2 = h''^T N_hits_pairs h'',
      - If tighten_B is True:
            B = 2 * max_k sum_i 1{setting k hits i} * h''_i
        else:
            B = 2 * sum_i h''_i

    The systematic term is sum_{N_i=0} |w_i|.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # Systematic error: observables never measured
    N_hits = np.asarray(N_hits)
    w = np.asarray(w, dtype=float)
    eps_sys = float(np.sum(np.abs(w[N_hits == 0])))

    # Mask to measured observables
    mask = (N_hits > 0)
    if not np.any(mask):
        return eps_sys

    # h'' on measured coordinates
    N_hits_meas = N_hits[mask]
    hpp = np.abs(w[mask]) / N_hits_meas

    # B: tightened vs loose
    if tighten_B and (is_hit_array is not None) and (is_hit_array.size > 0):
        # Restrict is_hit_array to measured observables
        is_hit_sel = is_hit_array[:, mask].astype(int, copy=False)
        if is_hit_sel.shape[0] > 0:
            settings_weights = is_hit_sel @ hpp  # shape (n_settings,)
            if settings_weights.size > 0:
                B = 2.0 * float(np.max(settings_weights))
            else:
                # degenerate case -> fall back to loose
                B = 2.0 * float(np.sum(hpp))
        else:
            # no rows -> fall back to loose
            B = 2.0 * float(np.sum(hpp))
    else:
        # Either tighten_B=False or no useful is_hit_array available:
        # use the looser global bound.
        B = 2.0 * float(np.sum(hpp))

    # sigma via pair counts
    N_hits_pairs = np.asarray(N_hits_pairs, dtype=float)
    N_pairs_sel = N_hits_pairs[np.ix_(mask, mask)]
    sigma_sq = float(hpp @ (N_pairs_sel @ hpp))
    if sigma_sq < 0.0:
        sigma_sq = 0.0
    sigma = np.sqrt(sigma_sq)

    eps_stat = sigma * 2.0 * np.sqrt(-np.log(delta / 2.0))

    if B > 0.0:
        validity_range = 3.0 * sigma**2 / B
        delta_min = np.exp(-9.0 * sigma**2 / (4.0 * B**2))
        if eps_stat > validity_range and warnings:
            print("Warning! Epsilon out of validity range. Either increase the number of measurement rounds or increase the inconfidence bound delta.")
            print("Range of validity of Theorem 3 in terms of epsilon: [0, %f]" % (validity_range,))
            print("Relevant range of inconfidence delta: [%f, 1]" % (delta_min,))

    return eps_stat + eps_sys

def N_delta_Bernstein_scalar_no_restricted_validity(delta):
    # Similar approach to N_delta_Bernstein_scalar
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return -(7+np.sqrt(33))/3 * np.log(delta/2)

def get_epsilon_Bernstein_scalar_no_restricted_validity(delta, N_hits, w):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = |weights/sqrt(N_hits)| * sqrt(2 log(2/delta)) + 2*|weights/sqrt(N_hits)| * 2/3 log(2/delta) 
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        w_abs  = np.abs(w[N_hits > 0])
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        sigma  = np.sum(w_abs)
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        B      = 2 * np.sum(w_abs)
        
        eps_stat = sigma*np.sqrt(-2*np.log(delta/2)) - B * 2/3*np.log(delta/2)
    else:
        eps_stat = 0.0
    
    return eps_stat + eps_sys

def get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(
    delta, N_hits, N_hits_pairs, w, is_hit_array, tighten_B: bool = True):
    """
    Scalar Bernstein (tighter, no restricted validity):
      eps_stat = sigma * sqrt(2 log(2/delta)) + (2/3) B log(2/delta)

    with
      - h''_i = |w_i| / N_i for N_i>0 (excluded otherwise),
      - sigma^2 = h''^T N_hits_pairs h'',
      - If tighten_B is True:
            B = 2 * max_k sum_i 1{setting k hits i} * h''_i  (tightened via is_hit_array)
        else:
            B = 2 * sum_i h''_i

    Systematic term: sum_{N_i=0} |w_i|.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")

    N_hits       = np.asarray(N_hits)
    N_hits_pairs = np.asarray(N_hits_pairs, dtype=float)
    w            = np.asarray(w, dtype=float)

    # Systematic error from never-measured observables
    eps_sys = float(np.sum(np.abs(w[N_hits == 0])))

    # Mask to measured coordinates
    mask = (N_hits > 0)
    if not np.any(mask):
        # nothing measured -> no statistical term
        return eps_sys

    # h'' on measured coordinates
    N_hits_meas = N_hits[mask]
    hpp = np.abs(w[mask]) / N_hits_meas  # shape (m,)

    # B: tightened vs loose
    if tighten_B and (is_hit_array is not None) and (is_hit_array.size > 0):
        # Restrict is_hit_array to measured observables
        is_hit_sel = is_hit_array[:, mask].astype(int, copy=False)
        if is_hit_sel.shape[0] > 0:
            settings_weights = is_hit_sel @ hpp  # (n_settings,)
            if settings_weights.size > 0:
                B = 2.0 * float(np.max(settings_weights))
            else:
                # degenerate -> fall back to loose
                B = 2.0 * float(np.sum(hpp))
        else:
            # no rows -> fall back to loose
            B = 2.0 * float(np.sum(hpp))
    else:
        # Either tighten_B=False or we have no useful is_hit_array:
        # use the looser global bound.
        B = 2.0 * float(np.sum(hpp))

    # sigma via pair counts
    N_pairs_sel = N_hits_pairs[np.ix_(mask, mask)]
    sigma_sq = float(hpp @ (N_pairs_sel @ hpp))
    if sigma_sq < 0.0:
        sigma_sq = 0.0
    sigma = np.sqrt(sigma_sq)

    log_term = -np.log(delta / 2.0)
    eps_stat = sigma * np.sqrt(2.0 * log_term) + (2.0 / 3.0) * B * log_term

    return eps_stat + eps_sys

def get_epsilon_Bernstein_scalar_tightest_no_restricted_validity(
    delta, N_hits, N_hits_pairs, w, is_hit_array, cov_real,
    tighten_B: bool = True):
    """Bernstein bound (tightest scalar) using exact covariances.

    Var = w_eff^T * (N_hits_pairs ∘ cov_real) * w_eff
    where w_eff[i] = w[i] / N_hits[i] (0 if N_hits[i]==0),
    cov_real = Re[(V + V^H)/2].

    If tighten_B is True (default):
        B = 2 * max_k sum_{i: setting k hits i} |w_eff[i]|
    else:
        B = 2 * sum_i |w_eff[i]|
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    N_hits       = np.asarray(N_hits, dtype=np.int64)
    N_hits_pairs = np.asarray(N_hits_pairs, dtype=np.float64)
    w            = np.asarray(w, dtype=np.float64)
    cov_real     = np.asarray(cov_real, dtype=np.float64)

    # systematic error: never-measured observables
    eps_sys = float(np.sum(np.abs(w[N_hits == 0])))

    # build w_eff = w/N with zeros where N=0
    w_eff   = np.zeros_like(w, dtype=np.float64)
    mask_pos = (N_hits > 0)
    w_eff[mask_pos] = w[mask_pos] / N_hits[mask_pos]

    # if nothing measured -> no statistical term
    if not np.any(mask_pos):
        return eps_sys

    abs_w_eff = np.abs(w_eff)

    # range term B: tightened vs loose
    if tighten_B and (is_hit_array is not None) and (is_hit_array.size > 0):
        is_hit_array_int = np.asarray(is_hit_array, dtype=int)
        if is_hit_array_int.shape[0] > 0:
            B_array = is_hit_array_int @ abs_w_eff  # (n_settings,)
            if B_array.size > 0:
                B = 2.0 * float(np.max(B_array))
            else:
                # degenerate -> fall back to loose
                B = 2.0 * float(np.sum(abs_w_eff))
        else:
            # no settings -> fall back to loose
            B = 2.0 * float(np.sum(abs_w_eff))
    else:
        # either tighten_B=False or no useful is_hit_array
        B = 2.0 * float(np.sum(abs_w_eff))

    # Var = w_eff^T * (N_hits_pairs ∘ cov_real) * w_eff
    PSD = N_hits_pairs * cov_real
    Var = float(w_eff @ (PSD @ w_eff))
    if Var < 0.0:  # numerical safety
        Var = 0.0
    sigma = np.sqrt(Var)

    log_term = -np.log(delta / 2.0)
    eps_stat = sigma * np.sqrt(2.0 * log_term) + (2.0 / 3.0) * B * log_term

    return eps_stat + eps_sys

def get_epsilon_Bernstein_scalar_tightest_no_restricted_validity_numba(
    delta, N_hits, N_hits_pairs, w, is_hit_array, cov_real, tighten_B: bool = True):
    """Numba version of the tightest scalar Bernstein bound.

    If tighten_B is True (default):
        B = 2 * max_k sum_{i: is_hit_array[k, i]} |w_eff[i]|
    else:
        B = 2 * sum_i |w_eff[i]|

    Var = w_eff^T * (N_hits_pairs ∘ cov_real) * w_eff,
    with w_eff[i] = w[i] / N_hits[i] (0 if N_hits[i] == 0).
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")

    # coerce dtypes
    N_hits       = np.asarray(N_hits,       dtype=np.int64)
    N_hits_pairs = np.asarray(N_hits_pairs, dtype=np.float64)
    w            = np.asarray(w,            dtype=np.float64)
    cov_real     = np.asarray(cov_real,     dtype=np.float64)

    eps_stat, eps_sys = _bernstein_tightest_core(delta, N_hits, N_hits_pairs,
                                                 w, is_hit_array, cov_real, tighten_B)
    return eps_stat + eps_sys

def get_epsilon_Bernstein_empirical_Audibert(
    delta, N_hits, w, is_hit_array, cov_real,
    tighten_B: bool = True):
    """
    Empirical (scalar) Bernstein guarantee of Audibert-type:

        epsilon = sqrt( 2 * log(3/delta) * V_emp ) + 3 * log(3/delta) * B  + eps_sys

    where
        hpp_i = |w_i| / N_i   (for N_i > 0)
        B     = ||hpp||_1 = sum_i hpp_i     (optionally tightened via is_hit_array)
        V_emp = hpp^T * C * hpp
        eps_sys = sum_{N_i=0} |w_i|

    Notes
    -----
    - `cov_real` is assumed to already be the matrix C computed empirically
      (it already contains the co-measurement sums and the 1/N mean-product correction).
    - `tighten_B=True` uses: B = max_k sum_{i hit by setting k} hpp_i
      over the (unique) settings stored in `is_hit_array`. Otherwise uses global sum.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")

    N_hits = np.asarray(N_hits, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    C = np.asarray(cov_real, dtype=np.float64)

    M = w.shape[0]
    if N_hits.shape[0] != M:
        raise ValueError(f"Shape mismatch: len(w)={M}, len(N_hits)={N_hits.shape[0]}")
    if C.shape != (M, M):
        raise ValueError(f"cov_real/C must have shape {(M, M)}, got {C.shape}")

    # systematic error: never-measured observables
    eps_sys = float(np.sum(np.abs(w[N_hits == 0])))

    # measured mask
    mask = (N_hits > 0)
    if not np.any(mask):
        return eps_sys

    # h'' on measured coords
    hpp = np.abs(w[mask]) / N_hits[mask].astype(np.float64)  # shape (m,)

    # B term
    if tighten_B and (is_hit_array is not None) and (np.size(is_hit_array) > 0):
        is_hit = np.asarray(is_hit_array, dtype=np.int8)
        if is_hit.shape[1] != M:
            raise ValueError(f"is_hit_array must have M={M} columns, got {is_hit.shape[1]}")
        is_hit_sel = is_hit[:, mask]
        if is_hit_sel.shape[0] > 0:
            B = float(np.max(is_hit_sel @ hpp)) if is_hit_sel.size else 0.0
        else:
            B = float(np.sum(hpp))
    else:
        B = float(np.sum(hpp))

    # V_emp term
    C_sel = C[np.ix_(mask, mask)]

    V_emp = float(hpp @ (C_sel @ hpp))
    if V_emp < 0.0:
        V_emp = 0.0

    log_term = float(np.log(3.0 / delta))
    eps_stat = np.sqrt(2.0 * log_term * V_emp) + 3.0 * log_term * B

    return eps_sys + eps_stat

def get_epsilon_Bernstein_empirical_Audibert_numba(
    delta, N_hits, w, is_hit_array, cov_real, tighten_B: bool = True):
    """
    Numba-accelerated version of get_epsilon_Bernstein_empirical_Audibert.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")

    N_hits = np.asarray(N_hits, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    C = np.asarray(cov_real, dtype=np.float64)

    M = w.shape[0]
    if N_hits.shape[0] != M:
        raise ValueError(f"Shape mismatch: len(w)={M}, len(N_hits)={N_hits.shape[0]}")
    if C.shape != (M, M):
        raise ValueError(f"cov_real/C must have shape {(M, M)}, got {C.shape}")

    if is_hit_array is None:
        is_hit_array = np.empty((0, M), dtype=np.bool_)
    else:
        is_hit_array = np.asarray(is_hit_array, dtype=np.bool_)
        if is_hit_array.ndim != 2 or is_hit_array.shape[1] != M:
            raise ValueError(f"is_hit_array must have shape (K, {M}), got {is_hit_array.shape}")

    eps_stat, eps_sys = _bernstein_empirical_audibert_core(
        float(delta), N_hits, w, is_hit_array, C, bool(tighten_B))
    
    return eps_stat + eps_sys

def get_epsilon_Bernstein_empirical_interpolated(delta, N_hits, w, cov_real, 
                                                 cov_initialized: bool = False):
    """
    Empirical (Audibert-style) scalar Bernstein guarantee with an automatic range-term switch.

    Guarantee:
        eps = sqrt( 2 * log(3/delta) * V_emp ) + 3 * log(3/delta) * B + eps_sys

    where
      - eps_sys = sum_{i: N_i=0} |w_i|
      - V_emp   = h''^T C h''   with  h''_i = |w_i| / N_i   (only for N_i>0)
      - B is chosen as:
          * if cov_initialized is False:  B = ||h'||_1  with h'_i  = |w_i| / sqrt(N_i)
          * if cov_initialized is True:   B = ||h''||_1 with h''_i = |w_i| / N_i

    Parameters
    ----------
    delta : float in (0,1)
    N_hits : (M,) int array
    w : (M,) float array
    cov_real : (M,M) float array
        Empirical covariance matrix C (assumed already symmetrized).
    cov_initialized : bool, default False
        If True, skip zero-matrix detection and assume empirical covariances are meaningful.
        If False, we optionally check whether the measured block of C is still ~0 (using zero_tol).

    Returns
    -------
    eps : float
    """
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0,1)")

    N_hits = np.asarray(N_hits, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    cov_real = np.asarray(cov_real, dtype=np.float64)

    # systematic error from never-measured observables
    eps_sys = float(np.sum(np.abs(w[N_hits == 0])))

    mask = (N_hits > 0)
    if not np.any(mask):
        return eps_sys

    Nh = N_hits[mask].astype(np.float64, copy=False)
    w_abs = np.abs(w[mask])

    # h'' for variance term
    hpp = w_abs / Nh                      # |w|/N
    # h' for early-phase range term
    hp = w_abs / np.sqrt(Nh)              # |w|/sqrt(N)

    C_sel = cov_real[np.ix_(mask, mask)]

    if cov_initialized:
        Var = float(hpp @ (C_sel @ hpp))
        if Var < 0.0:
            Var = 0.0
        B = float(np.sum(hpp))            # ||h''||_1
    else:
        Var = 0.0
        B = float(np.sum(hp))         # ||h'||_1

    log_term = float(np.log(3.0 / float(delta)))
    eps_stat = np.sqrt(2.0 * log_term * Var) + 3.0 * log_term * B

    return eps_stat + eps_sys

@njit
def get_epsilon_Bernstein_empirical_interpolated_numba(delta, N_hits, w, cov_real, 
                                                       cov_initialized=False):
    """
    Numba version of get_epsilon_Bernstein_empirical_interpolated (trust cov_initialized).
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")

    M = N_hits.shape[0]

    # systematic error
    eps_sys = 0.0
    for i in range(M):
        if N_hits[i] == 0:
            wi = w[i]
            eps_sys += wi if wi >= 0.0 else -wi

    # check if anything measured
    has_measured = False
    for i in range(M):
        if N_hits[i] > 0:
            has_measured = True
            break
    if not has_measured:
        return eps_sys

    log_term = math.log(3.0 / delta)

    if not cov_initialized:
        # Var = 0, B = ||h'||_1 = sum |w|/sqrt(N)
        B = 0.0
        for i in range(M):
            Ni = N_hits[i]
            if Ni > 0:
                wi = w[i]
                wi = wi if wi >= 0.0 else -wi
                B += wi / math.sqrt(float(Ni))

        eps_stat = 3.0 * log_term * B
        return eps_stat + eps_sys

    # cov_initialized == True:
    # Build hpp = |w|/N (with zeros for N=0), compute Var = hpp^T C hpp
    hpp = np.zeros(M, dtype=np.float64)
    B = 0.0
    for i in range(M):
        Ni = N_hits[i]
        if Ni > 0:
            wi = w[i]
            wi = wi if wi >= 0.0 else -wi
            hi = wi / float(Ni)
            hpp[i] = hi
            B += hi

    # Var = sum_{i,j} hpp[i] * C[i,j] * hpp[j]
    Var = 0.0
    for i in range(M):
        hi = hpp[i]
        if hi == 0.0:
            continue
        row = cov_real[i]
        s = 0.0
        for j in range(M):
            hj = hpp[j]
            if hj != 0.0:
                s += row[j] * hj
        Var += hi * s

    if Var < 0.0:
        Var = 0.0

    eps_stat = math.sqrt(2.0 * log_term * Var) + 3.0 * log_term * B
    return eps_stat + eps_sys
    
def N_delta_Hoeffding_scalar(delta):
    # Similar approach to N_delta_Bernstein_scalar
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return -2*np.log(delta/2)

def get_epsilon_Hoeffding_scalar(delta, N_hits, w):
    """ Return the epsilon such that the corresponding Hoeffding bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = 2*|weights/sqrt(N_hits)| * sqrt(1/2 * log(2/delta))
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))

    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        w_abs  = np.abs(w[N_hits > 0])
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        B      = 2 * np.sum(w_abs)
        
        eps_stat = B*np.sqrt(-1/2*np.log(delta/2))
    else:
        eps_stat = 0.0
    
    return eps_stat + eps_sys
    
@njit
def get_epsilon_Hoeffding_scalar_numba(delta, N_hits, w):
    eps_sys = 0.0
    B = 0.0
    for i in range(len(N_hits)):
        if N_hits[i] == 0:
            eps_sys += abs(w[i])
        else:
            B += abs(w[i]) / np.sqrt(N_hits[i])

    eps_stat = 2 * B * np.sqrt(-0.5 * np.log(delta / 2)) if B > 0 else 0.0

    return eps_stat + eps_sys
    
def get_epsilon_Hoeffding_scalar_tighter(delta, N_hits, N_hits_pairs, w):
    """
    Hoeffding (scalar, tighter) with pair counts:
      eps_stat = B * sqrt( (1/2) * log(2/delta) ),
      B^2 = 4 * h''^T (N_hits_pairs) h'',
      h''_i = |w_i| / N_i for N_i>0 (excluded otherwise).

    Systematic term: sum_{N_i=0} |w_i|.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # Systematic error (never-measured)
    eps_sys = float(np.sum(np.abs(w[N_hits == 0])))

    # Measured mask
    mask = (N_hits > 0)
    if not np.any(mask):
        eps_stat = 0.0
        return eps_stat + eps_sys

    # h'' on measured indices only
    hpp = np.abs(w[mask]) / N_hits[mask]

    # Restrict pair counts to measured block
    N_pairs_sel = N_hits_pairs[np.ix_(mask, mask)]

    # B^2 = 4 * h''^T N_pairs_sel h''
    B_sq = 4.0 * float(hpp @ (N_pairs_sel @ hpp))
    if B_sq < 0.0:  # numerical guard
        B_sq = 0.0
    B = np.sqrt(B_sq)

    eps_stat = B * np.sqrt(-0.5 * np.log(delta / 2.0))

    return eps_stat + eps_sys

def get_epsilon_Hoeffding_scalar_tighter_numba(delta, N_hits, N_hits_pairs, w):
    """
    Numba-accelerated version of get_epsilon_Hoeffding_scalar_tighter_v2.

    epsilon = B * sqrt( (1/2) * log(2/delta) ), with
    B = 2 * sqrt( sum_k [ sum_i (|h_i|/N_i) * 1{i compatible with setting k} ]^2 )
      = 2 * sqrt( h''^T N_hits_pairs h'' ), where h''_i = |h_i|/N_i if N_i>0 else 0.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in the interval (0,1)")

    # Ensure consistent dtypes for numba
    N_hits = np.asarray(N_hits, dtype=np.int64)
    N_hits_pairs = np.asarray(N_hits_pairs, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    # Basic shape checks (cheap, outside numba)
    M = w.shape[0]
    if N_hits.shape[0] != M or N_hits_pairs.shape != (M, M):
        raise ValueError("Shape mismatch: "
                         f"len(w)={M}, len(N_hits)={N_hits.shape[0]}, "
                         f"N_hits_pairs.shape={N_hits_pairs.shape}, expected {(M,M)}")

    eps_stat, eps_sys = _hoeffding_tighter_core(delta, N_hits, N_hits_pairs, w)
    return eps_stat + eps_sys

def N_delta_Chebyshev_scalar(delta):
    # Similar approach to N_delta_Bernstein_scalar
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return 1/delta

def get_epsilon_Chebyshev_scalar(delta, N_hits, w):
    """ Return the epsilon such that the corresponding Chebyshev bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = |weights/sqrt(N_hits)| / sqrt(delta)
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        w_abs  = np.abs(w[N_hits > 0])
        w_abs /= np.sqrt(N_hits[N_hits > 0])
        sigma  = np.sum(w_abs)
        
        eps_stat = sigma/np.sqrt(delta)
    else:
        eps_stat = 0.0
    
    return eps_stat + eps_sys
    
@njit
def get_epsilon_Chebyshev_scalar_numba(delta, N_hits, w):
    eps_sys = 0.0
    sigma = 0.0

    for i in range(len(N_hits)):
        if N_hits[i] == 0:
            eps_sys += abs(w[i])
        else:
            sigma += abs(w[i]) / np.sqrt(N_hits[i])

    eps_stat = sigma / np.sqrt(delta) if sigma > 0 else 0.0

    return eps_stat + eps_sys
    
def get_epsilon_Chebyshev_scalar_tighter(delta, N_hits, N_hits_pairs, w):
    """Chebyshev (tighter) with pair counts.

    eps_stat = sigma / sqrt(delta),
    sigma^2 = h''^T (N_hits_pairs[mask,mask]) h'',
    h''_i = |w_i| / N_i for N_i>0 (indices with N_i=0 excluded).

    Systematic term: sum_{N_i=0} |w_i|.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # systematic error from never-measured observables
    eps_sys = float(np.sum(np.abs(w[N_hits == 0])))

    mask = (N_hits > 0)
    if not np.any(mask):
        eps_stat = 0.0
        return eps_stat + eps_sys

    # h'' and restricted pair-count matrix
    hpp = np.abs(w[mask]) / N_hits[mask]
    N_pairs_sel = N_hits_pairs[np.ix_(mask, mask)]

    # sigma^2 = h''^T N_pairs_sel h''
    sigma_sq = float(hpp @ (N_pairs_sel @ hpp))
    if sigma_sq < 0.0:  # numerical guard
        sigma_sq = 0.0
    sigma = np.sqrt(sigma_sq)

    eps_stat = sigma / np.sqrt(delta)
    
    return eps_stat + eps_sys

def get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits, N_hits_pairs, w):
    """
    Numba-accelerated Chebyshev (tighter) bound.

    eps_stat = sigma / sqrt(delta),
    sigma^2  = h''^T N_hits_pairs h'',
    with h''_i = |w_i| / N_i for N_i>0, and 0 otherwise.

    Systematic term: sum_{N_i=0} |w_i|.
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # Ensure consistent dtypes
    N_hits       = np.asarray(N_hits, dtype=np.int64)
    N_hits_pairs = np.asarray(N_hits_pairs, dtype=np.float64)
    w            = np.asarray(w, dtype=np.float64)

    eps_stat, eps_sys = _chebyshev_tighter_core(delta, N_hits, N_hits_pairs, w)
    
    return eps_stat + eps_sys

def get_epsilon_Chebyshev_scalar_tightest(delta, N_hits, N_hits_pairs, w, cov_real):
    """Return epsilon for the Chebyshev bound on energy deviation.
       If at least one N_hits == 0, account for systematic error.
       Var = w_eff^T (N_hits_pairs ∘ cov_real) w_eff
       with w_eff[i] = w[i]/N_i (0 if N_i==0).
    """
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    N_hits = np.asarray(N_hits, dtype=np.int64)
    N_hits_pairs = np.asarray(N_hits_pairs, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    # systematic error
    eps_sys = np.sum(np.abs(w[N_hits == 0]))

    # effective weights
    w_eff = np.zeros_like(w, dtype=np.float64)
    mask_pos = N_hits > 0
    w_eff[mask_pos] = w[mask_pos] / N_hits[mask_pos]

    if not np.any(mask_pos):
        return eps_sys

    # Var
    PSD = N_hits_pairs * cov_real
    Var = float(w_eff @ (PSD @ w_eff))
    if Var < 0.0:  # safety
        Var = 0.0

    sigma = np.sqrt(Var)
    eps_stat = sigma / np.sqrt(delta)

    return eps_stat + eps_sys
    
def get_epsilon_Chebyshev_scalar_tightest_numba(delta, N_hits, N_hits_pairs,
                                                w, cov_real, expvals_real=None,
                                                systematic_mode="l1",
                                                certified_systematic_bound=None,
                                                return_components=False):
    """
    Return a Chebyshev guarantee for the error between the truncated
    empirical estimator and the full energy.

    Parameters
    ----------
    delta : float
        Failure probability, with 0 < delta < 1.

    N_hits : array_like, shape (M,)
        Number of samples available for each Pauli observable.

    N_hits_pairs : array_like, shape (M, M)
        Number of shared samples for every pair of observables.

    w : array_like, shape (M,)
        Pauli coefficients.

    cov_real : array_like, shape (M, M)
        Exact single-shot covariance matrix.

    expvals_real : array_like, shape (M,), optional
        Exact Pauli expectation values. Required when
        systematic_mode == "exact_state".

    systematic_mode : {"l1", "exact_state", "certified"}
        "l1":
            Use sum_{unmeasured i} |w_i|. This is state-independent.

        "exact_state":
            Use |sum_{unmeasured i} w_i <P_i>|. This is valid only
            for the state whose exact expectation values were supplied.

        "certified":
            Use certified_systematic_bound, for example an independently
            obtained operator-norm bound.

    certified_systematic_bound : float, optional
        User-supplied rigorous upper bound on the truncation bias.

    return_components : bool
        If True, also return the statistical and systematic components,
        the signed truncation bias when available, and the variance.

    Returns
    -------
    epsilon_total : float
        Radius satisfying

            Pr(|E_hat_truncated - E_full| > epsilon_total) <= delta.

    components : dict, optional
        Returned only when return_components=True.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in the interval (0, 1)")

    N_hits = np.asarray(N_hits, dtype=np.int64)
    N_hits_pairs = np.asarray(N_hits_pairs, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    cov_real = np.asarray(cov_real, dtype=np.float64)

    M = len(w)

    if N_hits.shape != (M,):
        raise ValueError("N_hits must have shape (M,)")
    if N_hits_pairs.shape != (M, M):
        raise ValueError("N_hits_pairs must have shape (M, M)")
    if cov_real.shape != (M, M):
        raise ValueError("cov_real must have shape (M, M)")
    if np.any(N_hits < 0):
        raise ValueError("N_hits cannot contain negative entries")

    omitted = N_hits == 0
    truncation_bias = None

    if systematic_mode == "l1":
        # State-independent worst-case bound:
        # |sum_i w_i <P_i>| <= sum_i |w_i|.
        eps_sys = float(np.sum(np.abs(w[omitted])))

    elif systematic_mode == "exact_state":
        if expvals_real is None:
            raise ValueError(
                "expvals_real is required for systematic_mode='exact_state'"
            )

        expvals_real = np.asarray(expvals_real, dtype=np.float64)
        if expvals_real.shape != (M,):
            raise ValueError("expvals_real must have shape (M,)")

        truncation_bias = float(np.dot(w[omitted], expvals_real[omitted]))
        eps_sys = abs(truncation_bias)

    elif systematic_mode == "certified":
        if certified_systematic_bound is None:
            raise ValueError(
                "certified_systematic_bound is required for "
                "systematic_mode='certified'"
            )

        eps_sys = float(certified_systematic_bound)
        if eps_sys < 0.0:
            raise ValueError(
                "certified_systematic_bound must be nonnegative"
            )

    else:
        raise ValueError(
            "systematic_mode must be 'l1', 'exact_state', or 'certified'"
        )

    variance = _chebyshev_tightest_core(N_hits, N_hits_pairs, w, cov_real)

    # A genuinely negative value indicates inconsistent covariance/hit data.
    # Only a small floating-point residual should be clipped.
    scale = max(1.0, np.max(np.abs(w)) ** 2)
    tolerance = 1e-12 * scale

    if variance < -tolerance:
        raise ValueError(
            "Computed variance is significantly negative. Check that "
            "cov_real and N_hits_pairs describe a consistent sampling scheme.")

    variance = max(variance, 0.0)
    eps_stat = np.sqrt(variance / delta)
    epsilon_total = eps_sys + eps_stat

    if not return_components:
        return epsilon_total

    return epsilon_total, {
        "epsilon_statistical": eps_stat,
        "epsilon_systematic": eps_sys,
        "truncation_bias": truncation_bias,
        "variance": variance}

def get_single_Hoeffding_plus_union_bound(epsilon, N_hits, w):
    """ Returns the delta such that the corresponding energy deviation is not larger than epsilon.
        Hoeffding bound is applied to each observable, and is then followed by union bound.
        Specifically, delta = 2 x sum_{obs i} exp( -epsilon^2 N_i / (2 h_i^2) ).
    """
    inconf = 2 * np.exp( -0.5*(epsilon**2)*N_hits/(w**2) )
    return np.sum(inconf)

def get_epsilon_single_Hoeffding_plus_union_bound(delta, N_hits, w):
    """ Return the effective epsilon such that the inconfidence probability from the original
        Derandomization paper by Huang, Kueng and Preskill is not larger than delta.
        Inconfidence probability = 2 sum_{obs i} exp(- epsilon^2 N_i / (2 h_i^2)) leq delta
        Then, effective epsilon is epsilon times number of observables due to union bound.
        Since inconfidence bound cannot be inverted analytically, root is found numerically.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        alpha = N_hits[N_hits > 0] / (2 * np.square(w[N_hits > 0]))
    
        def Derandomization_guarantee_eq(epsilon, alpha, delta):
            return 2 * np.sum(np.exp(-alpha * epsilon**2)) - delta
    
        sol = sp.optimize.root_scalar(Derandomization_guarantee_eq, args=(alpha, delta), 
                                      bracket=[0, 100], method='brentq')
    
        if sol.converged:
            eps_stat = w[N_hits > 0].shape[0]*sol.root # Multiply by number of observables to get effective epsilon
        else:
            raise RuntimeError("Root-finding failed: no solution found for the given delta and measurement scheme.")
    else:
        eps_stat = 0.0
    
    return eps_stat + eps_sys
        
