import numpy as np, copy

def pre_truncate(scheme, tau):
    """ 
    Truncates the set of observables before generating the measurement scheme.
    Specifically, the observables with coefficients of lowest absolute value
    whose sum is less than or equal to tau are discarded. scheme is altered in-place.
    """
    order_of_abs_w = np.argsort(np.abs(scheme.w))
    ordered_abs_w = np.sort(np.abs(scheme.w))
    cumulative_ordered_abs_w = np.cumsum(ordered_abs_w)
    truncation_condition = cumulative_ordered_abs_w < tau
    indices_obs_to_cut = order_of_abs_w[truncation_condition]
    
    keep = np.bitwise_not(np.in1d(np.array(list(range(len(scheme.obs)))), indices_obs_to_cut))

    if np.sum(keep) == 0:
        print("With such a large tau, all observables are discarded!")
        print("Scheme unaltered.")
        return 0
    elif np.sum(keep) == len(keep):
        print("Nothing was truncated.")
        return 0
    else:
        eps_sys = np.sum(np.abs(scheme.w[np.bitwise_not(keep)]))
        
        print("Number of truncated observables: ", int(scheme.num_obs-np.sum(keep)))
        print("Systematic error: ", eps_sys)

        scheme.w = scheme.w[keep]
        scheme.obs = scheme.obs[keep]
        scheme.N_hits = scheme.N_hits[keep]
        scheme.num_obs = len(scheme.w)

        return eps_sys

def truncate_analytical(scheme, delta, N_delta_function, 
                        get_epsilon_function, in_place=True):
    """ Truncation function to apply the truncation criterion given a 
        certain inconfidence level delta. Assumes that scheme has called 
        the function find_setting() sufficiently often. Truncates all 
        observables that fulfill the truncation criterion and saves the 
        sum of their absolute coefficient values. Returns the resulting 
        introduced systematic error epsilon and the statistical error
        associated with the corresponding guarantee for the current scheme. 
        This analytical truncation is valid for any of the guarantees that 
        are not tightened (i.e., that do not take the compatibility between 
        the observables and the settings into account, because that disrupts 
        the linearity upon which the analytical truncation is based). The 
        corresponding N_delta_function and get_epsilon_function must be 
        provided as inputs. If in_place is True, the truncation is applied
        in-place to the scheme, otherwise it is left unchanged.
    """
    # Sanity check for valid input functions
    if "tight" in get_epsilon_function.__name__.lower():
        raise ValueError("Tightened guarantee functions are not compatible with truncate_analytical.")
    
    N_unmeasured = np.sum(scheme.N_hits == 0)
    if N_unmeasured > 0:
        print("Warning! {} observable(s) have not been measured at least once.".format(N_unmeasured))
        print("If you have set alpha large, this can result in a non-optimal truncation.")
    N_crit = N_delta_function(delta)
    keep = scheme.N_hits > int(N_crit) # round down to integer value
    if np.sum(keep) == 0:
        # only systematic error
        eps_syst = np.sum(np.abs(scheme.w))
        eps_stat = 0.0
        num_trun_obs = scheme.w.shape[0]
        print("No observable reached the threshold. Ensure that you have sampled often enough or provide a smaller delta!")
        print("Scheme unaltered.")
    elif np.sum(keep) == len(keep):
        # only statistical error
        eps_syst = 0.0
        eps_stat = get_epsilon_function(delta, scheme.N_hits, scheme.w)
        num_trun_obs = 0
        print("Nothing had to be truncated.")
    else:
        eps_syst = np.sum(np.abs(scheme.w[np.bitwise_not(keep)]))
        eps_stat = get_epsilon_function(delta, scheme.N_hits[keep], scheme.w[keep])
        num_trun_obs = scheme.w.shape[0] - np.sum(keep)
        
        if in_place:
            scheme.w = scheme.w[keep]
            scheme.obs = scheme.obs[keep]
            scheme.N_hits = scheme.N_hits[keep]
            scheme.num_obs = len(scheme.w)
            
    return eps_stat, eps_syst, num_trun_obs

def truncate_binary_search(estimator, delta, get_epsilon_function, 
                           tightened=True, in_place=True, verbose=False):
    """ Applies the truncation strategy and returns the statistical error, the
        systematic error, and the number of truncated observables. If in_place 
        = True, estimator is altered in-place. Works for any inconfidence bound, which
        should be calculated via the input get_epsilon_function. If tightened =
        = True, get_epsilon_function takes as inputs delta, N_hits, w, settings_dict
        and obs, otherwise only the first three inputs are used. If verbose = True, 
        information is provided about the evolution of the error as the different 
        truncation levels are attempted.
    """
    
    N_unmeasured = np.sum(estimator.measurement_scheme.N_hits == 0)
    if N_unmeasured > 0:
        print("Warning! {} observable(s) have not been measured at least once.".format(N_unmeasured))
        print("If you have set alpha large, this can result in a non-optimal truncation.")
    
    if tightened:
        eps_no_truncation = get_epsilon_function(delta, estimator.measurement_scheme.N_hits,
                                                 estimator.measurement_scheme.w,
                                                 estimator.settings_dict,
                                                 estimator.measurement_scheme.obs)
    else:
        eps_no_truncation = get_epsilon_function(delta, estimator.measurement_scheme.N_hits,
                                                 estimator.measurement_scheme.w)

    order_of_N_hits = np.argsort(estimator.measurement_scheme.N_hits)
    ordered_abs_w = np.abs(estimator.measurement_scheme.w)[order_of_N_hits]
    cumulative_ordered_abs_w = np.cumsum(ordered_abs_w)
    truncation_candidates_condition = cumulative_ordered_abs_w < eps_no_truncation
    candidate_obs_indices = order_of_N_hits[truncation_candidates_condition] # only consider lowest-coefficient observables whose
                                                                             # sum of absolute value of coefficients is lower than
                                                                             # error bound without truncation, otherwise there is 
                                                                             # no way truncation can improve the overall error
    candidate_w = np.abs(estimator.measurement_scheme.w[candidate_obs_indices])
    candidate_N_hits = estimator.measurement_scheme.N_hits[candidate_obs_indices]
    order = np.lexsort((candidate_w, candidate_N_hits)) # order by no. samples first, then break ties with coefficient
    ordered_candidate_obs_indices = candidate_obs_indices[order]

    current_cut = len(ordered_candidate_obs_indices)//2
    cut_floor = 0
    cut_ceiling = len(ordered_candidate_obs_indices)
    
    i = 2
    best_eps_syst = 0
    best_eps_stat = eps_no_truncation
    best_eps = best_eps_syst + best_eps_stat
    keep = np.bitwise_not(np.in1d(np.array(list(range(len(estimator.measurement_scheme.obs)))), np.array([]))) # by default, keep all
    estimator_best_copy = estimator

    if verbose:
        print('Reference error: {}'.format(eps_no_truncation))
        print('\n')
    
    while current_cut < cut_ceiling and current_cut > cut_floor:
        estimator_safe_copy = copy.deepcopy(estimator)
        indices_obs_to_cut = ordered_candidate_obs_indices[:current_cut]
        keep_tentative = np.bitwise_not(np.in1d(np.array(list(range(len(estimator.measurement_scheme.obs)))), indices_obs_to_cut))
        
        if np.sum(keep_tentative) == 0:
            # only systematic error
            eps_syst = np.sum(np.abs(estimator.measurement_scheme.w))
            eps_stat = 0
        elif np.sum(keep_tentative) == len(keep_tentative):
            # only statistical error
            eps_syst = 0
            eps_stat = eps_no_truncation
        else:
            # override copy
            estimator_safe_copy.measurement_scheme.w = estimator.measurement_scheme.w[keep_tentative]
            estimator_safe_copy.measurement_scheme.obs = estimator.measurement_scheme.obs[keep_tentative]
            estimator_safe_copy.measurement_scheme.N_hits = estimator.measurement_scheme.N_hits[keep_tentative]
            estimator_safe_copy.measurement_scheme.num_obs = np.sum(keep_tentative)
            estimator_safe_copy.measurement_scheme.settings_dict = {}
            # generate a new measurement scheme for the kept observables
            estimator_safe_copy.reset()
            estimator_safe_copy.propose_next_settings(estimator.num_settings)
            # calculate guarantees
            eps_syst = np.sum(np.abs(estimator.measurement_scheme.w[np.bitwise_not(keep_tentative)]))
            
            if tightened:
                eps_stat = get_epsilon_function(delta, estimator_safe_copy.measurement_scheme.N_hits,
                                                estimator_safe_copy.measurement_scheme.w,
                                                estimator_safe_copy.settings_dict,
                                                estimator_safe_copy.measurement_scheme.obs)
            else:
                eps_stat = get_epsilon_function(delta, estimator_safe_copy.measurement_scheme.N_hits,
                                                estimator_safe_copy.measurement_scheme.w)
            # no need to undo overriding because we used the safe copy in the calculations

        if verbose:
            print('Attempting to remove the first {} observable(s).'.format(int(current_cut)))
            print('Systematic error: {}'.format(eps_syst))
            print('Statistical error: {}'.format(eps_stat))
            print('Total error: {}'.format(eps_syst + eps_stat))

        if eps_syst + eps_stat < best_eps:
            if verbose:
                print('Total error improved, so raising truncation cut.')
                print('\n')
            keep = keep_tentative
            best_eps_syst = eps_syst
            best_eps_stat = eps_stat
            best_eps = best_eps_syst + best_eps_stat
            estimator_best_copy = estimator_safe_copy
            if verbose:
                print('New reference error: {}'.format(best_eps))
                print('\n')

            cut_floor = current_cut
            if len(ordered_candidate_obs_indices)//2**i > 0:
                current_cut = current_cut + len(ordered_candidate_obs_indices)//2**i
            else:
                current_cut = current_cut + 1
        else:
            if verbose:
                print('Total error worsened, so lowering truncation cut.')
                print('\n')
            cut_ceiling = current_cut
            if len(ordered_candidate_obs_indices)//2**i > 0:
                current_cut = current_cut - len(ordered_candidate_obs_indices)//2**i
            else:
                current_cut = current_cut - 1

        i = i + 1
            
    if np.sum(keep) == 0:
        # only systematic error
        eps_syst = np.sum(np.abs(estimator.measurement_scheme.w))
        eps_stat = 0.0
        num_trun_obs = estimator.measurement_scheme.w.shape[0]
        print("All observables have been discarded. Ensure that you have sampled often enough or provide a larger delta!")
        print("Scheme unaltered.")
    elif np.sum(keep) == len(keep):
        # only statistical error
        eps_syst = 0.0
        eps_stat = eps_no_truncation
        num_trun_obs = 0
        print("Nothing had to be truncated.")
    else:
        eps_syst = np.sum(np.abs(estimator.measurement_scheme.w[np.bitwise_not(keep)]))
        eps_stat = best_eps_stat
        num_trun_obs = estimator.measurement_scheme.w.shape[0] - np.sum(keep)
        
        if in_place:
            estimator.measurement_scheme.w = estimator_best_copy.measurement_scheme.w
            estimator.measurement_scheme.obs = estimator_best_copy.measurement_scheme.obs
            estimator.measurement_scheme.N_hits = estimator_best_copy.measurement_scheme.N_hits
            estimator.measurement_scheme.num_obs = np.sum(keep)
            estimator.settings_dict = estimator_best_copy.settings_dict
            estimator.settings_buffer = estimator_best_copy.settings_buffer

    return eps_stat, eps_syst, num_trun_obs

def truncate_full_sweep(estimator, delta, get_epsilon_function, 
                        tightened=True, in_place=True, verbose=False):
    """Applies the truncation strategy and returns the statistical error, the
       systematic error, and the number of truncated observables. If in_place 
       = True, estimator is altered in-place. Works for any inconfidence bound,
       calculated via get_epsilon_function. If tightened is True, 
       get_epsilon_function takes (delta, N_hits, w, settings_dict, obs), 
       otherwise only the first three. If verbose = True, information is 
       provided about the evolution of the error as the different truncation 
       levels are attempted.
    """

    N_unmeasured = np.sum(estimator.measurement_scheme.N_hits == 0)
    if N_unmeasured > 0:
        print("Warning! {} observable(s) have not been measured at least once.".format(N_unmeasured))
        print("If you have set alpha large, this can result in a non-optimal truncation.")

    # Reference (no truncation) statistical error
    if tightened:
        eps_no_truncation = get_epsilon_function(
            delta, estimator.measurement_scheme.N_hits,
            estimator.measurement_scheme.w,
            estimator.settings_dict,
            estimator.measurement_scheme.obs)
    else:
        eps_no_truncation = get_epsilon_function(
            delta, estimator.measurement_scheme.N_hits,
            estimator.measurement_scheme.w)

    # Create candidate pool of truncatable observables
    order_of_N_hits = np.argsort(estimator.measurement_scheme.N_hits)
    ordered_abs_w = np.abs(estimator.measurement_scheme.w)[order_of_N_hits]
    cumulative_ordered_abs_w = np.cumsum(ordered_abs_w)
    truncation_candidates_condition = cumulative_ordered_abs_w < eps_no_truncation
    candidate_obs_indices = order_of_N_hits[truncation_candidates_condition]

    candidate_w = np.abs(estimator.measurement_scheme.w[candidate_obs_indices])
    candidate_N_hits = estimator.measurement_scheme.N_hits[candidate_obs_indices]
    order = np.lexsort((candidate_w, candidate_N_hits))
    ordered_candidate_obs_indices = candidate_obs_indices[order]

    best_eps_syst = 0.0
    best_eps_stat = eps_no_truncation
    best_eps = best_eps_stat + best_eps_syst
    best_keep_tentative = np.ones(len(estimator.measurement_scheme.obs), dtype=bool)
    best_estimator_copy = copy.deepcopy(estimator)

    if verbose:
        print(f"Reference error (no truncation): {eps_no_truncation}\n")

    for k in range(len(ordered_candidate_obs_indices) + 1):
        indices_to_cut = ordered_candidate_obs_indices[:k]
        keep_tentative = np.ones(len(estimator.measurement_scheme.obs), dtype=bool)
        keep_tentative[indices_to_cut] = False
        estimator_copy = copy.deepcopy(estimator)

        if np.sum(keep_tentative) == 0:
            eps_syst = np.sum(np.abs(estimator.measurement_scheme.w))
            eps_stat = 0.0
        elif np.sum(keep_tentative) == len(keep_tentative):
            eps_syst = 0.0
            eps_stat = eps_no_truncation
        else:
            # Apply truncation
            estimator_copy.measurement_scheme.w = estimator.measurement_scheme.w[keep_tentative]
            estimator_copy.measurement_scheme.obs = estimator.measurement_scheme.obs[keep_tentative]
            estimator_copy.measurement_scheme.N_hits = estimator.measurement_scheme.N_hits[keep_tentative]
            estimator_copy.measurement_scheme.num_obs = np.sum(keep_tentative)
            estimator_copy.measurement_scheme.settings_dict = {}

            # Regenerate settings
            estimator_copy.reset()
            estimator_copy.propose_next_settings(estimator.num_settings)

            eps_syst = np.sum(np.abs(estimator.measurement_scheme.w[~keep_tentative]))
            if tightened:
                eps_stat = get_epsilon_function(
                    delta, estimator_copy.measurement_scheme.N_hits,
                    estimator_copy.measurement_scheme.w,
                    estimator_copy.settings_dict,
                    estimator_copy.measurement_scheme.obs)
            else:
                eps_stat = get_epsilon_function(
                    delta, estimator_copy.measurement_scheme.N_hits,
                    estimator_copy.measurement_scheme.w)

        eps_total = eps_syst + eps_stat

        if verbose:
            print(f"Truncating {k} observables:")
            print(f"  Systematic error: {eps_syst}")
            print(f"  Statistical error: {eps_stat}")
            print(f"  Total error: {eps_total}\n")

        if eps_total < best_eps:
            best_eps = eps_total
            best_eps_stat = eps_stat
            best_eps_syst = eps_syst
            best_keep_tentative = keep_tentative
            best_estimator_copy = estimator_copy

    num_trun_obs = np.sum(~best_keep_tentative)

    if num_trun_obs == 0:
        print("Nothing had to be truncated.")
    elif np.sum(best_keep_tentative) == 0:
        print("All observables have been discarded. Provide more samples or increase delta.")
        print("Scheme unaltered.")
    elif in_place:
        estimator = best_estimator_copy

    return best_eps_stat, best_eps_syst, num_trun_obs