import numpy as np, scipy as sp
from numba import njit
from shadowgrouping_v2.helper_functions import (
    sample_obs_from_setting_numba, setting_to_obs_form)

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

def get_epsilon_Bernstein(delta, N_hits, w, split=False):
    """ Returns the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = 2*|weights/sqrt(N_hits)| * (1 + 2sqrt(log(1/delta))). split = True
        provides statistical and systematic errors separately, otherwise they are summed.
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
        if eps_stat > 2*norm*(1+norm/norm2):
            print("Warning! Epsilon out of validity range.")
    else:
        eps_stat = 0.0
    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys

def get_epsilon_Bernstein_tighter(delta, N_hits, w, settings_dict, obs, split=False):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = sigma * (2 + 4sqrt(log(1/delta))), with sigma given by:
        sigma = 2 * sqrt{ sum_{setting k} [ sum_{obs i compatible with setting k} |h_i|/N_i ]^2 }.
        See second line of Eq. (25) of supp. inf. of published version of ShadowGrouping paper.
        Similarly, B = 4 * max_{all settings with dummy index k} { sum_{obs i compatible with setting k} |h_i|/N_i }.
        See first line of Eq. (23) of of supp. inf. of published version of ShadowGrouping paper.
        split = True provides statistical and systematic errors separately, otherwise they are summed.
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
                
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        settings_list = list(settings_dict.keys())
        
        settings_weights = []
        settings_reps = []
        for k in range(len(settings_list)):
            new_weight = 0
            for i in range(len(obs)):
                if sample_obs_from_setting_numba(obs[i], setting_to_obs_form(settings_list[k])):
                    new_weight = new_weight + abs(w[i])/N_hits[i]
            settings_weights.append(new_weight)
            settings_reps.append(settings_dict[settings_list[k]])
        sigma = np.square(np.array(settings_weights))
        sigma = np.dot(settings_reps, sigma)   # Need to sum over all settings, not just distinct ones, so multiply by number of repetitions
        sigma = 2*np.sqrt(np.sum(sigma))
        
        eps_stat = sigma * (1/2) * np.sqrt(N_delta_Bernstein(delta)) # The factor of 2 from sigma had been absorbed by alpha_{delta}
        
        B = 4*max(settings_weights)
        validity_range = sigma + 3*sigma**2/B  # See Eq. (44) from v5.3 of shared notes on Bernstein inequalities
        delta_min = np.exp(-9*sigma**2/(4*B**2)) # Results from setting epsilon = sigma + 3 sigma**2 / B to bound on inconfidence probability
        
        if eps_stat > validity_range:
            print("Warning! Epsilon out of validity range. Either increase number of measurement rounds or increase inconfidence bound delta.")
            print("Range of validity of Theorem 3 in terms of epsilon: [%f, %f]" %(sigma, validity_range) )
            print("Relevant range of inconfidence delta: [%f, 1]" %(delta_min))
    else:
        eps_stat = 0.0
    
    if split:
        return eps_stat, eps_sys
    else:
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

def get_epsilon_Bernstein_no_restricted_validity(delta, N_hits, w, split=False):
    """ Returns the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = sigma * [1 + sqrt(2 log(1/delta)) ] + 2B/3 * log(1/delta).
        split = True provides statistical and systematic errors separately, otherwise they are summed.
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

    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys

def get_epsilon_Bernstein_tighter_no_restricted_validity(delta, N_hits, w, settings_dict, obs, split=False):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = sigma * [1 + sqrt(2 log(1/delta)) ] + 2B/3 * log(1/delta), with sigma given by:
        sigma = 2 * sqrt{ sum_{setting k} [ sum_{obs i compatible with setting k} |h_i|/N_i ]^2 }.
        See second line of Eq. (25) of supp. inf. of published version of ShadowGrouping paper.
        Similarly, B = 4 * max_{all settings with dummy index k} { sum_{obs i compatible with setting k} |h_i|/N_i }.
        See first line of Eq. (23) of of supp. inf. of published version of ShadowGrouping paper.
        split = True provides statistical and systematic errors separately, otherwise they are summed.
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        settings_list = list(settings_dict.keys())
    
        settings_weights = []
        settings_reps = []
        for k in range(len(settings_list)):
            new_weight = 0
            for i in range(len(obs)):
                if sample_obs_from_setting_numba(obs[i], setting_to_obs_form(settings_list[k])):
                    new_weight = new_weight + abs(w[i])/N_hits[i]
            settings_weights.append(new_weight)
            settings_reps.append(settings_dict[settings_list[k]])
        sigma = np.square(np.array(settings_weights))
        sigma = np.dot(settings_reps, sigma)   # Need to sum over all settings, not just distinct ones, so multiply by number of repetitions
        sigma = 2*np.sqrt(np.sum(sigma))
        
        B = 4*max(settings_weights)
    
        eps_stat = sigma * ( 1 + np.sqrt(-2*np.log(delta)) ) - 2*B*np.log(delta)/3
    else:
        eps_stat = 0.0

    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys

def N_delta_Bernstein_scalar(delta):
    # Similar approach to N_delta_Bernstein; just different numerical factors
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return -4*np.log(delta/2)

def get_epsilon_Bernstein_scalar(delta, N_hits, w, split=False):
    """ Returns the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = |weights/sqrt(N_hits)| * 2sqrt(log(2/delta)).
        split = True provides statistical and systematic errors separately, otherwise they are summed.
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
        
        if eps_stat > validity_range:
            print("Warning! Epsilon out of validity range. Either increase number of measurement rounds or increase inconfidence bound delta.")
            print("Range of validity of Theorem 3 in terms of epsilon: [%f, %f]" %(0, validity_range) )
            print("Relevant range of inconfidence delta: [%f, 1]" %(delta_min))
    else:
        eps_stat = 0.0
    
    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys

def get_epsilon_Bernstein_scalar_tighter(delta, N_hits, w, settings_dict, obs, split=False):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = sigma * 2sqrt(log(2/delta)), with sigma given by:
        sigma = sqrt{ sum_{setting k} [ sum_{obs i compatible with setting k} |h_i|/N_i ]^2 }.
        Similarly, B = 2 * max_{all settings with dummy index k} { sum_{obs i compatible with setting k} |h_i|/N_i }.
        split = True provides statistical and systematic errors separately, otherwise they are summed.
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
                
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        settings_list = list(settings_dict.keys())
        
        settings_weights = []
        settings_reps = []
        for k in range(len(settings_list)):
            new_weight = 0
            for i in range(len(obs)):
                if sample_obs_from_setting_numba(obs[i], setting_to_obs_form(settings_list[k])):
                    new_weight = new_weight + abs(w[i])/N_hits[i]
            settings_weights.append(new_weight)
            settings_reps.append(settings_dict[settings_list[k]])
        sigma = np.square(np.array(settings_weights))
        sigma = np.dot(settings_reps, sigma)   # Need to sum over all settings, not just distinct ones, so multiply by number of repetitions
        sigma = np.sqrt(np.sum(sigma))
        B = 2*max(settings_weights)
        
        eps_stat = sigma * 2*np.sqrt(-np.log(delta/2))
        
        validity_range = 3*sigma**2/B
        delta_min = np.exp(-9*sigma**2/(4*B**2))
        
        if eps_stat > validity_range:
            print("Warning! Epsilon out of validity range. Either increase number of measurement rounds or increase inconfidence bound delta.")
            print("Range of validity of Theorem 3 in terms of epsilon: [%f, %f]" %(0, validity_range) )
            print("Relevant range of inconfidence delta: [%f, 1]" %(delta_min))
    else:
        eps_stat = 0.0
    
    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys

def N_delta_Bernstein_scalar_no_restricted_validity(delta):
    # Similar approach to N_delta_Bernstein_scalar
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return -(7+np.sqrt(33))/3 * np.log(delta/2)

def get_epsilon_Bernstein_scalar_no_restricted_validity(delta, N_hits, w, split=False):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = |weights/sqrt(N_hits)| * sqrt(2 log(2/delta)) + 2*|weights/sqrt(N_hits)| * 2/3 log(2/delta) 
        split = True provides statistical and systematic errors separately, otherwise they are summed.
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
    
    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys

def get_epsilon_Bernstein_scalar_tighter_no_restricted_validity(delta, N_hits, w, settings_dict, obs, split=False):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = sigma * sqrt(2 log(2/delta)) + 2B/3 * log(2/delta), with sigma given by:
        sigma = sqrt{ sum_{setting k} [ sum_{obs i compatible with setting k} |h_i|/N_i ]^2 }.
        Similarly, B = 2 * max_{all settings with dummy index k} { sum_{obs i compatible with setting k} |h_i|/N_i }.
        split = True provides statistical and systematic errors separately, otherwise they are summed.
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))

    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        settings_list = list(settings_dict.keys())
    
        settings_weights = []
        settings_reps = []
        for k in range(len(settings_list)):
            new_weight = 0
            for i in range(len(obs)):
                if sample_obs_from_setting_numba(obs[i], setting_to_obs_form(settings_list[k])):
                    new_weight = new_weight + abs(w[i])/N_hits[i]
            settings_weights.append(new_weight)
            settings_reps.append(settings_dict[settings_list[k]])
        sigma = np.square(np.array(settings_weights))
        sigma = np.dot(settings_reps, sigma)   # Need to sum over all settings, not just distinct ones, so multiply by number of repetitions
        sigma = np.sqrt(np.sum(sigma))
        
        B = 2*max(settings_weights)
    
        eps_stat = sigma * np.sqrt(-2*np.log(delta/2)) - 2*B*np.log(delta/2)/3
    else:
        eps_stat = 0.0

    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys
    
def N_delta_Hoeffding_scalar(delta):
    # Similar approach to N_delta_Bernstein_scalar
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return -2*np.log(delta/2)

def get_epsilon_Hoeffding_scalar(delta, N_hits, w, split=False):
    """ Return the epsilon such that the corresponding Hoeffding bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = 2*|weights/sqrt(N_hits)| * sqrt(1/2 * log(2/delta))
        split = True provides statistical and systematic errors separately, otherwise they are summed.
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
    
    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys
    
@njit
def get_epsilon_Hoeffding_scalar_numba(delta, N_hits, w, split=False):
    eps_sys = 0.0
    B = 0.0
    for i in range(len(N_hits)):
        if N_hits[i] == 0:
            eps_sys += abs(w[i])
        else:
            B += abs(w[i]) / np.sqrt(N_hits[i])

    eps_stat = 2 * B * np.sqrt(-0.5 * np.log(delta / 2)) if B > 0 else 0.0

    return (eps_stat, eps_sys) if split else (eps_stat + eps_sys, 0.0)

def get_epsilon_Hoeffding_scalar_tighter(delta, N_hits, w, settings_dict, obs, split=False):
    """ Return the epsilon such that the corresponding Hoeffding bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = B * sqrt(1/2 * log(2/delta)), with B given by:
        sigma = 2 * sqrt{ sum_{setting k} [ sum_{obs i compatible with setting k} |h_i|/N_i ]^2 }.
        split = True provides statistical and systematic errors separately, otherwise they are summed.
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        settings_list = list(settings_dict.keys())
    
        settings_weights = []
        settings_reps = []
        for k in range(len(settings_list)):
            new_weight = 0
            for i in range(len(obs)):
                if sample_obs_from_setting_numba(obs[i], setting_to_obs_form(settings_list[k])):
                    new_weight = new_weight + abs(w[i])/N_hits[i]
            settings_weights.append(new_weight)
            settings_reps.append(settings_dict[settings_list[k]])
        B = np.square(np.array(settings_weights))
        B = np.dot(settings_reps, B)   # Need to sum over all settings, not just distinct ones, so multiply by number of repetitions
        B = 2*np.sqrt(np.sum(B))
        
        eps_stat = B * np.sqrt(-1/2*np.log(delta/2))
    else:
        eps_stat = 0.0

    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys
    
@njit
def get_epsilon_Hoeffding_scalar_tighter_numba(delta, N_hits, w, settings_int, 
                                               settings_reps, obs, split=False):
    """
    Compute the Hoeffding upper bound for estimation error using a tighter strategy.
    Note that, in order not to compromise the compatibility with Numba methods, the
    settings_dict input was converted into settings_int (a 1D array of 1D arrays of int,
    each correponding to a distinct settings in observable format) and settings_reps
    (a 1D array of int that stores the number of repetitions of each setting). The
    former stores the keys of the dictionary (in a different format) and the latter
    stores the values in the same format as before.
    """
    eps_sys = 0.0
    for i in range(len(N_hits)):
        if N_hits[i] == 0:
            eps_sys += abs(w[i])

    sigma_squared = 0.0
    if np.sum(N_hits > 0) > 0:
        for k in range(len(settings_int)):
            s_weight = 0.0
            for i in range(len(obs)):
                if sample_obs_from_setting_numba(obs[i], settings_int[k]):
                    s_weight += abs(w[i]) / N_hits[i]
            sigma_squared += settings_reps[k] * (s_weight ** 2)
        B = 2 * np.sqrt(sigma_squared)
        eps_stat = B * np.sqrt(-0.5 * np.log(delta / 2))
    else:
        eps_stat = 0.0

    return (eps_stat, eps_sys) if split else (eps_stat + eps_sys, 0.0)

def N_delta_Chebyshev_scalar(delta):
    # Similar approach to N_delta_Bernstein_scalar
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    
    return 1/delta

def get_epsilon_Chebyshev_scalar(delta, N_hits, w, split=False):
    """ Return the epsilon such that the corresponding Chebyshev bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = |weights/sqrt(N_hits)| / sqrt(delta)
        split = True provides statistical and systematic errors separately, otherwise they are summed.
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
    
    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys
    
@njit
def get_epsilon_Chebyshev_scalar_numba(delta, N_hits, w, split=False):
    eps_sys = 0.0
    sigma = 0.0

    for i in range(len(N_hits)):
        if N_hits[i] == 0:
            eps_sys += abs(w[i])
        else:
            sigma += abs(w[i]) / np.sqrt(N_hits[i])

    eps_stat = sigma / np.sqrt(delta) if sigma > 0 else 0.0

    return (eps_stat, eps_sys) if split else (eps_stat + eps_sys, 0.0)

def get_epsilon_Chebyshev_scalar_tighter(delta, N_hits, w, settings_dict, obs, split=False):
    """ Return the epsilon such that the corresponding Chebyshev bound is not larger than delta.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        Else, epsilon = sigma / sqrt(delta), with sigma given by:
        sigma = sqrt{ sum_{setting k} [ sum_{obs i compatible with setting k} |h_i|/N_i ]^2 }.
        split = True provides statistical and systematic errors separately, otherwise they are summed.
    """
    
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")

    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    
    # statistical error due to observables with at least one sample
    if np.sum(N_hits > 0) > 0:
        settings_list = list(settings_dict.keys())
    
        settings_weights = []
        settings_reps = []
        for k in range(len(settings_list)):
            new_weight = 0
            for i in range(len(obs)):
                if sample_obs_from_setting_numba(obs[i], setting_to_obs_form(settings_list[k])):
                    new_weight = new_weight + abs(w[i])/N_hits[i]
            settings_weights.append(new_weight)
            settings_reps.append(settings_dict[settings_list[k]])
        sigma = np.square(np.array(settings_weights))
        sigma = np.dot(settings_reps, sigma)   # Need to sum over all settings, not just distinct ones, so multiply by number of repetitions
        sigma = np.sqrt(np.sum(sigma))
        
        eps_stat = sigma/np.sqrt(delta)
    else:
        eps_stat = 0.0

    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys
    
@njit
def get_epsilon_Chebyshev_scalar_tighter_numba(delta, N_hits, w, settings_int,
                                               settings_reps, obs, split=False):
    """
    Compute the Chebyshev upper bound for estimation error using a tighter strategy.
    Always returns a tuple: (eps_stat, eps_sys). Same splitting of settings_dict
    considered in get_epsilon_Hoeffding_scalar_tighter_numba is required.
    """
    eps_sys = 0.0
    for i in range(len(N_hits)):
        if N_hits[i] == 0:
            eps_sys += abs(w[i])

    eps_stat = 0.0
    if np.sum(N_hits > 0) > 0:
        sigma_squared = 0.0
        for k in range(len(settings_int)):
            s_weight = 0.0
            for i in range(len(obs)):
                if sample_obs_from_setting_numba(obs[i], settings_int[k]):
                    s_weight += abs(w[i]) / N_hits[i]
            sigma_squared += settings_reps[k] * (s_weight ** 2)
        sigma = np.sqrt(sigma_squared)
        eps_stat = sigma / np.sqrt(delta)

    return (eps_stat, eps_sys) if split else (eps_stat + eps_sys, 0.0)

def get_single_Hoeffding_plus_union_bound(epsilon, N_hits, w):
    """ Returns the delta such that the corresponding energy deviation is not larger than epsilon.
        Hoeffding bound is applied to each observable, and is then followed by union bound.
        Specifically, delta = 2 x sum_{obs i} exp( -epsilon^2 N_i / (2 h_i^2) ).
    """
    inconf = 2 * np.exp( -0.5*(epsilon**2)*N_hits/(w**2) )
    return np.sum(inconf)

def get_epsilon_single_Hoeffding_plus_union_bound(delta, N_hits, w, split=False):
    """ Return the effective epsilon such that the inconfidence probability from the original
        Derandomization paper by Huang, Kueng and Preskill is not larger than delta.
        Inconfidence probability = 2 sum_{obs i} exp(- epsilon^2 N_i / (2 h_i^2)) leq delta
        Then, effective epsilon is epsilon times number of observables due to union bound.
        Since inconfidence bound cannot be inverted analytically, root is found numerically.
        If at least one of the N_hits is 0, associated systematic error is accounted for.
        split = True provides statistical and systematic errors separately, otherwise they are summed.
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
    
    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys
        
def Guaranteed_accuracy(delta, N_hits, w, split=True):
    "Equation 32 of supplementary"
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0,1)")
    # systematic error due to observables that have not been measured even once
    eps_sys = np.sum(np.abs(w[N_hits == 0]))
    #eps_sys = 0
    # statistical error due to observables with at least one sample
    eps_stat = np.sum(N_delta_Bernstein(delta)*np.abs(w[N_hits > 0])/np.sqrt(N_hits[N_hits > 0]))
    if split:
        return eps_stat, eps_sys
    else:
        return eps_stat + eps_sys
