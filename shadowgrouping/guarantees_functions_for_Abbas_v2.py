import numpy as np, numbers

def sample_obs_from_setting(O,P):
    """ 
    Returns whether observable O can be sampled from setting P.

    E.g., sample_obs_from_setting([0, 2, 0], [1, 2, 3]) == True but
          sample_obs_from_setting([1, 2, 3], [1, 0, 3]) == False.

    It is equal to "hit_by" if setting is already complete. It is
    meant to be used only at the end of the construction of the
    setting, when we are determining which observables to sample
    from it. During the construction of the setting, "hit_by" is the
    right one to use. For Brute Force Search, Derandomization,
    Random Paulis and Adaptive Shadows, it makes no difference,
    because all settings are forced to have maximum support.
    For ShadowGrouping it makes no difference either, because, even
    if we generate an incomplete setting (i.e., with at least one
    identity), there will not be any observable that commutes with it
    and has nontrivial Paulis where the identities of the setting are.
    The replacement of "hit_by" with "sample_obs_from_setting" is relevant
    in SettingSampler whenever we provide incomplete settings.
    """
    
    if not isinstance(O, (list, np.ndarray)) or not isinstance(P, (list, np.ndarray)):
        raise TypeError("Both O and P must be lists or numpy arrays.")

    if len(O) != len(P):
        raise ValueError("O and P must be of the same length.")

    for i, (o, p) in enumerate(zip(O, P)):
        if not (isinstance(o, numbers.Integral) and 0 <= o <= 3):
            raise ValueError(f"O[{i}] = {o} is not an integer in [0, 3].")
        if not (isinstance(p, numbers.Integral) and 0 <= p <= 3):
            raise ValueError(f"P[{i}] = {p} is not an integer in [0, 3].")

        if o != 0 and o != p:
            return False

    return True

def setting_to_obs_form(setting_str):
    """
    Converts string form of setting into list form suitable for compatibility
    check with observables (also in list format) through "hit_by" function.

    E.g., setting_to_obs_form('XYZ') = [1,2,3],
          setting_to_obs_form('IXXIZY') = [0,1,1,0,3,2].

    Input must be a string where each character is in {'I','X','Y','Z'}.
    Output is a list of integers with as many elements as the string.
    Correspondence: {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}.
    
    Basically, this function reverses the action of the "setting_to_str"
    function (see below) that is called within "__settings_to_dict" in the 
    energy_estimator class to show the settings as strings in a dictionary.
    """

    if not isinstance(setting_str, str):
        raise TypeError("Input must be a string.")

    valid_chars = {'I', 'X', 'Y', 'Z'}
    for i, char in enumerate(setting_str):
        if char not in valid_chars:
            raise ValueError(f"Invalid character '{char}' at position {i}. Allowed characters are 'I', 'X', 'Y', 'Z'.")

    char_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    return [char_to_int[char] for char in setting_str]

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
                if sample_obs_from_setting(obs[i], setting_to_obs_form(settings_list[k])):
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
                if sample_obs_from_setting(obs[i], setting_to_obs_form(settings_list[k])):
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
    
    
    
