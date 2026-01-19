import numpy as np
import itertools
import networkx as nx

        

def DS(B):
    """
    finds a dominating set based on largest degree first method
    """
    # Compute the number of neighbors for each node
    neighbournum = {node: len(list(B.neighbors(node))) for node in B.nodes}

    # Sort nodes based on number of neighbors in descending order
    nsorted_indices = sorted(neighbournum.keys(), key=lambda x: neighbournum[x], reverse=True)

    # Find a dominating set using a greedy algorithm based on node degrees
    ndominating_set = set()
    ncovered_nodes = set()

    for node in nsorted_indices:
        if len(ncovered_nodes) == len(B.nodes):
            break
        if node not in ncovered_nodes:
            ndominating_set.add(node)
            ncovered_nodes.add(node)
            ncovered_nodes.update(B.neighbors(node))

    subgraphs = {}  # Dictionary to store subgraphs

    for v in ndominating_set:
        neighbors = list(B.neighbors(v))
        subgraph_nodes = neighbors + [v]
        subgraphs[v] = B.subgraph(subgraph_nodes).copy()

    return subgraphs


def recursive_clique_detection(B):
    """
    Recursively finds cliques in the graph B.
    If a subgraph is not a clique, apply the same procedure on it.
    """
    subgraphs = DS(B)  # Get the subgraphs from DS function

    for v, subgraph in subgraphs.items():
        deg_v = subgraph.degree(v)  # Degree of v in its subgraph
        expected_edges = (deg_v * (deg_v - 1)) // 2  # Expected edges in a clique

        if subgraph.number_of_edges() == expected_edges:
            print(f"Subgraph of {v} is a clique with {expected_edges} edges.")
        else:
            print(f"Subgraph of {v} is NOT a clique. Recursing on this subgraph...")
            recursive_clique_detection(subgraph)  # Recursively apply the function



# Example usage:
B = nx.erdos_renyi_graph(10, 0.4, seed=42)  # Generate a random graph
recursive_clique_detection(B)



#print("else started to work.")
            for idx in reversed(order):
                o = self.obs[idx]
                if verbose:
                    print("Checking",o)
                if hit_by(o,shadowcliquesetting):
                    non_id = o!=0
                    # overwrite those qubits that fall in the support of o
                    shadowcliquesetting[non_id] = o[non_id]
                    if verbose:
                        print("p =",setting)
                    # break sequence is case all identities in setting are exhausted
                    if np.min(shadowcliquesetting) > 0:
                        break
            shadowclique = []
            for o in self.obs:
                if hit_by(o, shadowcliquesetting):
                    shadowclique.append(o)
            if verbose:
                print("Checking list of observables.")
            # Get highest-weight observable
            first_idx = order[-1]  # last one in ascending sort = highest weight
            first_obs = self.obs[first_idx]
            center_node = tuple(first_obs)  # Use tuple as dictionary key
            # Check cache
            if center_node in self.clique_cache:
                hit_cliques = self.clique_cache[center_node]
                print(f"Using cached cliques for {center_node}")
            else:
                # Create setting from first_obs
                non_id = first_obs != 0
                globalsetting[non_id] = first_obs[non_id]  # make it the setting
                # Now check who is hit-by this setting
                hit_list = []
                for o in self.obs:
                    if hit_by(o, globalsetting):
                        hit_list.append(o)

                if verbose:
                    print("First observable (setting):", first_obs)
                    print("Other observables hit by it:")
                    for ob in hit_list:
                        print(ob)
                #print("hit_list has", len(hit_list), "observables.")
                if not hit_list:
                    raise RuntimeError("No hit list found.")
                #Now build the graph from hit list
                hit_graph = build_hit_graph(hit_list)
                #print("hit_graph has", hit_graph.number_of_nodes(), "nodes and", hit_graph.number_of_edges(), "edges.")
                if not hit_graph:
                    raise RuntimeError("No hit graph found.")
        
                #find the cliques around the center node
                center_node = tuple(first_obs)  # This is the observable around which we’re building
                if not center_node:
                    raise RuntimeError("No center node found.")
                hit_cliques = find_cliques5(hit_graph)
                print("hit_cliques found:", len(hit_cliques))
                # Cache result
                self.clique_cache[center_node] = hit_cliques
            hit_cliques.append(shadowclique)
            #print("shadow clique added:",shadowclique)
            #print("hit_cliques found:", len(hit_cliques))
            #for i, clique in enumerate(hit_cliques):
                #print(f"Clique {i+1}: {[list(node) for node in clique]}")
            #if not hit_cliques:
                #raise RuntimeError("No cliques found.")
            if not hit_cliques:
                hit_cliques=[[center_node]]
            # Compute SC for each clique
            # Compute SC and SW for each clique
            cliques_with_epsilon = []
            delta = 0.02
            for clique in hit_cliques:
            # ---- Step 2: simulate building a setting from this clique ---- if self.N_hits[i] > 0 else 1e-6
                setting_candidate = np.zeros(self.num_qubits, dtype=int)
                for o in clique:
                    o_arr = np.array(o)
                    if hit_by(o_arr, setting_candidate):
                        non_id = o_arr != 0
                        setting_candidate[non_id] = o_arr[non_id]       
                if np.min(setting_candidate) != 0:
                    # ---- Step 3: find all observables compatible with this setting_candidate ----
                    is_hit_candidate = np.array([sample_obs_from_setting(o,setting_candidate) for o in self.obs],dtype=bool) #CHANGEHIT
                    self.N_hits += is_hit_candidate
                    # ---- Store both scores along with the clique ----
                    cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), clique)) 
                    #cliques_with_epsilon.append((self.get_epsilon_Bernstein(delta), clique))
                    #cliques_with_epsilon.append((self.get_Bernstein_bound(), c  lique))
                    #cliques_with_epsilon.append((self.get_inconfidence_bound(), clique))
                    self.N_hits -= is_hit_candidate
                
            
            if not cliques_with_epsilon:
                raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
            # Sort cliques by epsilon
            cliques_with_epsilon.sort(key=lambda x: x[0])
            _, best_clique = cliques_with_epsilon[0]
            self.selected_cliques.append(best_clique)
            if len(best_clique) == len(shadowclique) and all(np.array_equal(np.array(o1), np.array(o2)) for o1, o2 in zip(best_clique, shadowclique)):
                self.shadow_was_best_count += 1
                print("Shadow clique was selected", self.shadow_was_best_count, "times")
            print("best clique has",len(best_clique),"members")
            #setting construction
            setting = np.zeros(self.num_qubits, dtype=int)
            for o in best_clique:
                o = np.array(o)
                if verbose:
                    print("Checking", o)
                if hit_by(o, setting):
                    non_id = o != 0
                    setting[non_id] = o[non_id]
                    if verbose:
                        print("p =", setting)
                if np.min(setting) > 0:
                    break
            if np.min(setting) == 0:
                for idx in reversed(order):
                    o = self.obs[idx]
                    if verbose:
                        print("Checking",o)
                    if hit_by(o,setting):
                        non_id = o!=0
                        #overwrite those qubits that fall in the support of o
                        setting[non_id] = o[non_id]
                        if verbose:
                            print("p =",setting)
                        #break sequence is case all identities in setting are exhausted
                        if np.min(setting) > 0:
                            break
        
        tend = time()

    
        # update number of hits
        is_hit = np.array([sample_obs_from_setting(o,setting) for o in self.obs],dtype=bool) #CHANGEHIT
        self.N_hits += is_hit




class Shadow_Grouping_Update8(Measurement_scheme):
    """ Grouping method based on weights obtained from classical shadows.
        The next measurement setting p is found as follows: it is initialized as the identity operator.
        Next, we obtain an ordering of the observables in terms of their respective weight_function.
        For each observable o in the ordered list of observables in descending order, it checks qubit-wise commutativity (QWC).
        If so, the qubits in p that fall in the support of o are overwritten by those in o.
        Eventually, the list is either exhausted or p does not contain identity operators anymore.
        The function weight_function takes in the weights,epsilon and the current number of N_hits and is supposed to return an numpy-array of length len(w).
        Instead, weight_function can also be set to None (this is useful for instances where the function is actually never called).
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.shadow_was_best_count = 0
        self.clique_cache = {}  # maps observable tuple -> hit cliques
        self.clean_clique_cache = {}  
        self.clean_setting_cache = {}
        self.is_hit_clique_cache = {}
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        return
    
    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
        #print(np.sum(inconf))
        return np.sum(inconf)
    
    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            bound = -1
        else:
            bound = np.exp(-0.25*(self.eps/2/np.sum(np.abs(self.w)/np.sqrt(self.N_hits))-1)**2)
        return bound            
        
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        # sort observable list by respective weight
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        self.selected_cliques = []  # Stores best cliques from each round
        setting = np.zeros(self.num_qubits,dtype=int)
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        if verbose:
            print("Checking list of observables.")
        
        tstart = time()
        
        hit_cliques = []
        shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
        #print("else started to work.")
        for idx in reversed(order):
            o = self.obs[idx]
            if verbose:
                print("Checking",o)
            if hit_by(o,shadowcliquesetting):
                non_id = o!=0
                # overwrite those qubits that fall in the support of o
                shadowcliquesetting[non_id] = o[non_id]
                if verbose:
                    print("p =",setting)
                # break sequence is case all identities in setting are exhausted
                if np.min(shadowcliquesetting) > 0:
                    break

        shadowclique = []
        for o in self.obs:
            if hit_by(o, shadowcliquesetting):
                shadowclique.append(o)
        
        if verbose:
            print("Checking list of observables.")

        # Get highest-weight observable
        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
        center_node = tuple(first_obs)  # Use tuple as dictionary key
        
        # Check cache
        if center_node in self.clique_cache:
            hit_cliques = self.clique_cache[center_node]
            print(f"Using cached cliques for {center_node}")
        else:
            # Create setting from first_obs
            non_id = first_obs != 0
            globalsetting[non_id] = first_obs[non_id]  # make it the setting
            # Now check who is hit-by this setting
            hit_list = []
            for o in self.obs:
                if hit_by(o, globalsetting):
                    hit_list.append(o)

            if verbose:
                print("First observable (setting):", first_obs)
                print("Other observables hit by it:")
                for ob in hit_list:
                    print(ob)

            #print("hit_list has", len(hit_list), "observables.")
            if not hit_list:
                raise RuntimeError("No hit list found.")
            #Now build the graph from hit list
            hit_graph = build_hit_graph(hit_list)
            #print("hit_graph has", hit_graph.number_of_nodes(), "nodes and", hit_graph.number_of_edges(), "edges.")
            if not hit_graph:
                raise RuntimeError("No hit graph found.")
            #find the cliques around the center node
            center_node = tuple(first_obs)  # This is the observable around which we’re building
            if not center_node:
                raise RuntimeError("No center node found.")
            hit_cliques = find_cliques5(hit_graph)
            #print("we fount these cliques",hit_cliques)
            # Cache result
            self.clique_cache[center_node] = hit_cliques
            
            
        hit_cliques.append(shadowclique)
        #print("shadow clique added:",shadowclique)
        #print("hit_cliques found:", len(hit_cliques))
        #for i, clique in enumerate(hit_cliques):
            #print(f"Clique {i+1}: {[list(node) for node in clique]}")
        #if not hit_cliques:
            #raise RuntimeError("No cliques found.")
        if not hit_cliques:
            hit_cliques=[[center_node]]
        # Compute SC for each clique
        # Compute SC and SW for each clique
        print("length of hit_cliques is",len(hit_cliques))
        cliques_with_epsilon = []
        delta = 0.02
        removedcliques = 0
        completecliques = 0
        valid_settings = []
        valid_cliques = []
        # for clique in hit_cliques:
        #     setting_candidate = np.zeros(self.num_qubits, dtype=int)
        #     for o in clique:
        #         o_arr = np.array(o)
        #         if hit_by(o_arr, setting_candidate):
        #             non_id = o_arr != 0
        #             setting_candidate[non_id] = o_arr[non_id]
        #         if np.min(setting_candidate) != 0:
        #             break
            # if np.min(setting_candidate) == 0:
            #     for idx in reversed(order):
            #         o = self.obs[idx]
            #         if verbose:
            #             print("Checking",o)
            #         if hit_by(o,setting_candidate):
            #             non_id = o!=0
            #             # overwrite those qubits that fall in the support of o
            #             setting_candidate[non_id] = o[non_id]
            #             if verbose:
            #                 print("p =",setting_candidate)
            #         # break sequence is case all identities in setting are exhausted
            #         if np.min(setting_candidate) > 0:
            #             break
            #if np.min(setting_candidate) != 0:
                #print("setting is now complete")
            # ---- Step 3: find all observables compatible with this setting_candidate ----
        if center_node not in self.clean_clique_cache:
            for clique in hit_cliques:
                # ---- Step 2: simulate building a setting from this clique ---- if self.N_hits[i] > 0 else 1e-6
                setting_candidate = np.zeros(self.num_qubits, dtype=int)
                for o in clique:
                    o_arr = np.array(o)
                    if hit_by(o_arr, setting_candidate):
                        non_id = o_arr != 0
                        setting_candidate[non_id] = o_arr[non_id]
                        if np.min(setting_candidate) > 0:
                            break
                
                if np.min(setting_candidate) == 0:
                    # Incomplete setting: remove this clique from the cache
                    #self.clique_cache[center_node].remove(clique)
                    removedcliques += 1
                    print("Removed incomplete clique", removedcliques , "times")
                else:
                    completecliques += 1
                    valid_cliques.append(clique)
                    valid_settings.append(setting_candidate)
                    is_hit_candidate = []
                    is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                    #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                    #self.is_hit_clique_cache[key] = is_hit_candidate
                    #is_hit_candidate = np.array([hit_by(o,setting_candidate) for o in self.obs],dtype=bool) #CHANGEHIT
                    self.N_hits += is_hit_candidate
                    #print("complete cliques are", completecliques)
                    # ---- Store both scores along with the clique ----
                    cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), setting_candidate)) 
                    #cliques_with_epsilon.append((self.get_epsilon_Bernstein(delta), clique))
                    #cliques_with_epsilon.append((self.get_Bernstein_bound(), clique))
                    #cliques_with_epsilon.append((self.get_inconfidence_bound(), clique))
                    self.N_hits -= is_hit_candidate
        else:
            print("I already met this candidate")
            for setting_candidate in self.clean_setting_cache[center_node]:
                #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                #is_hit_candidate = self.is_hit_clique_cache[key]
                is_hit_candidate = []
                is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                self.N_hits += is_hit_candidate
                cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), setting_candidate))
                self.N_hits -= is_hit_candidate
            
        is_hit_candidate = []
        is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
        self.N_hits += is_hit_candidate
        cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), shadowcliquesetting))
        self.N_hits -= is_hit_candidate

        if valid_cliques:
            self.clean_clique_cache[center_node] = valid_cliques
            self.clean_setting_cache[center_node] = valid_settings

        if not cliques_with_epsilon:
            raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
        print("length of cliques with epsilon is", len(cliques_with_epsilon))
        # Sort cliques by epsilon
        cliques_with_epsilon.sort(key=lambda x: x[0])
        _, best_clique = cliques_with_epsilon[0]
        self.selected_cliques.append(best_clique)

        #if len(best_clique) == len(shadowclique) and all(np.array_equal(np.array(o1), np.array(o2)) for o1, o2 in zip(best_clique, shadowclique)):
         #   self.shadow_was_best_count += 1
         #   print("Shadow clique was selected", self.shadow_was_best_count, "times")

        if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
            self.shadow_was_best_count += 1
            print("Shadow clique was selected", self.shadow_was_best_count, "times")

        #print("best clique has",len(best_clique),"members")

        setting = best_clique
        #setting construction
        # setting = np.zeros(self.num_qubits, dtype=int)
        # for o in best_clique:
        #     o_arra = np.array(o)
        #     if hit_by(o_arra, setting):
        #         non_id = o_arra != 0
        #         setting[non_id] = o_arra[non_id]
        #     if np.min(setting) > 0:
        #         break
        # print("setting = ",setting)
        # if np.min(setting) == 0:
        #     print("setting is not complete")
        #     for idx in reversed(order):
        #         o = self.obs[idx]
        #         if verbose:
        #             print("Checking",o)
        #         if hit_by(o,setting):
        #             non_id = o!=0
        #             # overwrite those qubits that fall in the support of o
        #             setting[non_id] = o[non_id]
        #         if verbose:
        #             print("p =",setting)
        #         # break sequence is case all identities in setting are exhausted
        #         if np.min(setting) > 0:
        #             break
        
        tend = time()

        is_hit = []
        # update number of hits
        is_hit = hit_by_batch_numba(self.obs , setting)
        self.N_hits += is_hit
        delta = 0.02
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
        print("epsilon_Bernstein_no_restricted_validity_v2:", info["epsilon_Bernstein_no_restricted_validity_v2"])
        #if np.min(self.N_hits) == 0:
        #    print("not all observables are measured at least once")
        #else:
        #    delta = 0.02
        #    info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        #    print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info