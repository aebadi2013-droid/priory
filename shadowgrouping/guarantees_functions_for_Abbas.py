# Add this function inside the Measurement_scheme class
# below the get_epsilon_Bernstein function

def get_epsilon_Bernstein_no_restricted_validity(self,delta):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, epsilon is set equal to infinity.
        Else, epsilon = sigma * [1 + sqrt(2 log(1/delta)) ] + 2B/3 * log(1/delta)
    """
    if np.min(self.N_hits) == 0:
        return np.infty
    w_abs  = np.abs(self.w)
    w_abs /= np.sqrt(self.N_hits)
    sigma  = 2 * np.sum(w_abs) # Eq. (25), Supp. Inf. of published version of ShadowGrouping paper

    w_abs /= np.sqrt(self.N_hits)
    B = 2 * np.sum(w_abs) # Eq. (23), Supp. Inf. of published version of ShadowGrouping paper
    epsilon = sigma * ( 1 + np.sqrt(-2*np.log(delta)) ) - 2*B*np.log(delta)/3

    return epsilon

# Add both these functions inside the Energy_estimator class
# above the propose_next_settings function

def get_epsilon_Bernstein_tighter(self,delta):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, epsilon is set equal to infinity.
        Else, epsilon = sigma * (2 + 4sqrt(log(1/delta))), with sigma given by:
        sigma = 2 * sqrt{ sum_{setting k} [ sum_{obs i compatible with setting k} |h_i|/N_i ]^2 }.
        See second line of Eq. (25) of supp. inf. of published version of ShadowGrouping paper.
        Similarly, B = max_{all settings with dummy index k} { sum_{obs i compatible with setting k} |h_i|/N_i }.
        See first line of Eq. (23) of of supp. inf. of published version of ShadowGrouping paper.
    """
    
    def setting_to_obs_form(setting_str):
        out = []
        for i in range(len(setting_str)):
            if setting_str[i] == 'X':
                out.append(1)
            elif setting_str[i] == 'Y':
                out.append(2)
            elif setting_str[i] == 'Z':
                out.append(3)
            else:
                out.append(0)
        return out
    
    if np.min(self.measurement_scheme.N_hits) == 0:
        return np.infty
    
    settings_list = list(self.settings_dict.keys())
    
    settings_weights = []
    settings_reps = []
    for k in range(len(settings_list)):
        new_weight = 0
        for i in range(len(self.measurement_scheme.obs)):
            if sample_obs_from_setting(self.measurement_scheme.obs[i], setting_to_obs_form(settings_list[k])):
                new_weight = new_weight + abs(self.measurement_scheme.w[i])/self.measurement_scheme.N_hits[i]
        settings_weights.append(new_weight)
        settings_reps.append(self.settings_dict[settings_list[k]])
    sigma = np.square(np.array(settings_weights))
    sigma = np.dot(settings_reps, sigma)   # Need to sum over all settings, not just distinct ones, so multiply by number of repetitions
    sigma = 2*np.sqrt(np.sum(sigma))
    
    epsilon = sigma * (1/2) * np.sqrt(N_delta(delta)) # The factor of 2 from sigma had been absorbed by alpha_{delta}
    
    B = 2*max(settings_weights)
    validity_range = 2*sigma + 3*(2*sigma)**2/B
    delta_min = np.exp(-9*sigma**2/(B**2))
    
    if epsilon > validity_range:
        print("Warning! Epsilon out of validity range. Either increase number of measurement rounds or increase inconfidence bound delta.")
        print("Range of validity of Theorem 3 in terms of epsilon: [%f, %f]" %(2*sigma, validity_range) )
        print("Relevant range of inconfidence delta: [%f, 1]" %(delta_min))
    
    return epsilon

def get_epsilon_Bernstein_tighter_no_restricted_validity(self,delta):
    """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
        If at least one of the N_hits is 0, epsilon is set equal to infinity.
        Else, epsilon = sigma * [1 + sqrt(2 log(1/delta)) ] + 2B/3 * log(1/delta), with sigma given by:
        sigma = 2 * sqrt{ sum_{setting k} [ sum_{obs i compatible with setting k} |h_i|/N_i ]^2 }.
        See second line of Eq. (25) of supp. inf. of published version of ShadowGrouping paper.
        Similarly, B = max_{all settings with dummy index k} { sum_{obs i compatible with setting k} |h_i|/N_i }.
        See first line of Eq. (23) of of supp. inf. of published version of ShadowGrouping paper.
    """

    def setting_to_obs_form(setting_str):
        out = []
        for i in range(len(setting_str)):
            if setting_str[i] == 'X':
                out.append(1)
            elif setting_str[i] == 'Y':
                out.append(2)
            elif setting_str[i] == 'Z':
                out.append(3)
            else:
                out.append(0)
        return out

    if np.min(self.measurement_scheme.N_hits) == 0:
        return np.infty

    settings_list = list(self.settings_dict.keys())

    settings_weights = []
    settings_reps = []
    for k in range(len(settings_list)):
        new_weight = 0
        for i in range(len(self.measurement_scheme.obs)):
            if sample_obs_from_setting(self.measurement_scheme.obs[i], setting_to_obs_form(settings_list[k])):
                new_weight = new_weight + abs(self.measurement_scheme.w[i])/self.measurement_scheme.N_hits[i]
        settings_weights.append(new_weight)
        settings_reps.append(self.settings_dict[settings_list[k]])
    sigma = np.square(np.array(settings_weights))
    sigma = np.dot(settings_reps, sigma)   # Need to sum over all settings, not just distinct ones, so multiply by number of repetitions
    sigma = 2*np.sqrt(np.sum(sigma))
    
    B = 2*max(settings_weights)

    epsilon = sigma * ( 1 + np.sqrt(-2*np.log(delta)) ) - 2*B*np.log(delta)/3

    return epsilon