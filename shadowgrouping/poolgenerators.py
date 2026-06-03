from tabnanny import verbose

import numpy as np
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
from time import time
import numbers
from numba import njit
from shadowgrouping_v2.helper_functions import (
    setting_to_str, char_to_int, hit_by_numba, hit_by_batch_numba, encode_setting_token, decode_setting_token, sample_obs_batch_from_setting_numba, prepare_settings_for_numba, setting_to_obs_form, sample_obs_batch_from_setting_batch_numba)
from shadowgrouping_v2.guarantees import (get_epsilon_Chebyshev_scalar_tighter_numba, 
get_epsilon_Chebyshev_scalar_numba, get_epsilon_Hoeffding_scalar_tighter_numba, 
get_epsilon_Hoeffding_scalar_numba, get_epsilon_Bernstein_scalar, 
get_epsilon_Bernstein_scalar_no_restricted_validity, get_epsilon_Bernstein_scalar_tighter_no_restricted_validity, 
Guaranteed_accuracy)
from guarantees.guarantees import (get_epsilon_Chebyshev_scalar_tighter_numba, get_epsilon_Chebyshev_scalar_tightest_numba)
from shadowgrouping_v2.allocation_mixin import _AllocationMixin
from shadowgrouping_v2.shadowgrouping_my_dev.helper_functions import settings_to_dict

##########################################################################################
### Helper functions #####################################################################
##########################################################################################
def hit_by(O,P):
    """ Returns whether o is hit by p """
    for o,p in zip(O,P):
        if not (o==0 or p==0 or o==p):
            return False
    return True

def gcomm(O, P):
    """ Returns whether the number of failed commuting pairs in O and P is even """
    fail_commuting_count = 0
    for o, p in zip(O, P):
        if not (o==0 or p==0 or o==p) :
            fail_commuting_count += 1
    #print(f"Fail_Commuting count for {O} and {P}: {fail_commuting_count}")  # Print the commuting count
    return fail_commuting_count % 2 == 0  # Check if the fail to commute in even number of indices


def pauli_string_to_array(pauli_str):
    mapping = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
    return np.array([mapping[c] for c in pauli_str])

# Helper function to build the graph
def build_hit_graph(hit_list):
    """
    Constructs an undirected graph where each node is an observable (from hit_list),
    and edges connect observables that are mutually hit_by each other.

    Args:
        hit_list (List[np.ndarray]): List of observables (e.g., numpy arrays like [1, 0, 3, 2])

    Returns:
        networkx.Graph: Graph where nodes = observables, edges = hit_by relations
    """
    G = nx.Graph()

    # Ensure the observables are in the correct format
    obs_arrays = [np.array(obs) if isinstance(obs, tuple) else obs for obs in hit_list]

    # Add nodes
    for obs in obs_arrays:
        G.add_node(tuple(obs))  # Convert to tuple if needed for consistency

    # Add edges based on mutual hit_by
    for i in range(len(obs_arrays)):
        for j in range(i + 1, len(obs_arrays)):
            obs_i = obs_arrays[i]
            obs_j = obs_arrays[j]
            #if gcomm(obs_i, obs_j):
            if hit_by(obs_i, obs_j):
                G.add_edge(tuple(obs_i), tuple(obs_j))  # Add edges between tuples

    return G

def build_hit_graph2(hit_list, weight_map=None):
    """
    Constructs an undirected graph where each node is an observable (from hit_list),
    and edges connect observables that are mutually hit_by each other.

    Args:
        hit_list (List[np.ndarray]): List of observables (e.g., numpy arrays like [1, 0, 3, 2])
        weight_map (dict, optional): A dictionary mapping observables (as tuples) to their weights.
                                      If provided, each node will have a 'weight' attribute.

    Returns:
        networkx.Graph: Graph where nodes = observables, edges = hit_by relations.
    """
    G = nx.Graph()

    # Ensure the observables are in the correct format
    obs_arrays = [np.array(obs) if isinstance(obs, tuple) else obs for obs in hit_list]

    # Add nodes
    for obs in obs_arrays:
        obs_tuple = tuple(obs)  # Convert to tuple for consistency
        G.add_node(obs_tuple)  # Add node to the graph

        # Set weight if weight_map is provided
        if weight_map and obs_tuple in weight_map:
            G.nodes[obs_tuple]['weight'] = weight_map[obs_tuple]

    # Add edges based on mutual hit_by
    for i in range(len(obs_arrays)):
        for j in range(i + 1, len(obs_arrays)):
            obs_i = obs_arrays[i]
            obs_j = obs_arrays[j]
            if hit_by(obs_i, obs_j):
                G.add_edge(tuple(obs_i), tuple(obs_j))  # Add edges between tuples

    return G


def DomClique1(A):
    """
    Finds a largest-degree-first clique partition of a graph.

    Inputs:
        A - (graph) - Graph for which the partition should be found.

    Outputs:
        (list{list{int}}) - A list containing cliques that partition A.
    """

    # Compute the number of neighbors for each node
    neighbournum = {node: len(list(A.neighbors(node))) for node in A.nodes}

    # Sort nodes based on number of neighbors in descending order
    nsorted_indices = sorted(neighbournum.keys(), key=lambda x: neighbournum[x], reverse=True)

    # Find a dominating set using a greedy algorithm based on node degrees
    ndominating_set = set()
    ncovered_nodes = set()

    for node in nsorted_indices:
        if len(ncovered_nodes) == len(A.nodes):
            break
        if node not in ncovered_nodes:
            ndominating_set.add(node)
            ncovered_nodes.add(node)
            ncovered_nodes.update(A.neighbors(node))

    # Initialize the list of maximal cliques
    MaxCliques = []

    
    for v in ndominating_set:
        neighbors = list(A.neighbors(v))
        subgraph_nodes = neighbors + [v]
        subgraph = A.subgraph(subgraph_nodes).copy()
        neighborcliques = list(nx.find_cliques(subgraph))
        cliques_sorted = sorted(neighborcliques, key=lambda clique: len(clique), reverse=True)
        uncovered_nodes = set(subgraph.nodes())

        while uncovered_nodes:
            for clique in cliques_sorted:
                if uncovered_nodes & set(clique):
                    MaxCliques.append(sorted(clique))
                    uncovered_nodes.difference_update(clique)
                    break

    return MaxCliques

# returns a largest-weight-first clique partition of a graph
def LDF_weighted(A):
    # Inputs:
    #     A - (graph) - graph for which partition should be found
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which partition A
    # This implementation expects a NetworkX graph `A` and returns cliques
    # as lists of the original node labels (same output shape as DomClique1).
    nodes = list(A.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)
    remaining = set(range(p))
    N = {}
    for i, n in enumerate(nodes):
        N[i] = set(node_to_idx[nb] for nb in A.neighbors(n))

    # build weight map for indices
    index_weight = {i: A.nodes[nodes[i]].get('weight', 0) for i in range(p)}

    aaa = []
    while remaining:
        # choose vertex with largest weight (tie-breaker: number of neighbors in remaining)
        a = max(remaining, key=lambda x: (index_weight.get(x, 0), len(N[x] & remaining)))
        aa0 = set([a])
        aa1 = N[a] & remaining
        while aa1:
            a2 = max(aa1, key=lambda x: (index_weight.get(x, 0), len(N[x] & aa1)))
            aa0.add(a2)
            aa1 &= N[a2]
        aaa.append(aa0)
        remaining -= aa0
    # Map indices back to original node labels to match DomClique1 output
    return [sorted([nodes[i] for i in aa]) for aa in aaa]


def LDF(A):
    # Inputs:
    #     A - (graph) - graph for which partition should be found
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which partition A
    # This implementation expects a NetworkX graph `A` and returns cliques
    # as lists of the original node labels (same output shape as DomClique1).
    nodes = list(A.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)
    remaining = set(range(p))
    N = {}
    for i, n in enumerate(nodes):
        N[i] = set(node_to_idx[nb] for nb in A.neighbors(n))
    aaa = []
    while remaining:
        a = max(remaining, key=lambda x: len(N[x] & remaining))
        aa0 = set([a])
        aa1 = N[a] & remaining
        while aa1:
            a2 = max(aa1, key=lambda x: len(N[x] & aa1))
            aa0.add(a2)
            aa1 &= N[a2]
        aaa.append(aa0)
        remaining -= aa0
    # Map indices back to original node labels to match DomClique1 output
    return [sorted([nodes[i] for i in aa]) for aa in aaa]

# returns a largest-weight-first clique partition of a graph. Always checks the list from the beginning for compatible observables.  
# The covered observables are only excluded from the leading nodes list, but not from the checking nodes list.
def sorted_NPBC(A):
        # Inputs:
    #     A - (graph) - graph for which partition should be found
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which partition A
    # This implementation expects a NetworkX graph `A` and returns cliques
    # as lists of the original node labels (same output shape as DomClique1).
    nodes = list(A.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)
    remaining = set(range(p))
    N = {}
    for i, n in enumerate(nodes):
        N[i] = set(node_to_idx[nb] for nb in A.neighbors(n))

    # build weight map for indices
    index_weight = {i: A.nodes[nodes[i]].get('weight', 0) for i in range(p)}

    aaa = []
    while remaining:
        # choose vertex with largest weight (tie-breaker: number of neighbors in remaining)
        a = max(remaining, key=lambda x: (index_weight.get(x, 0), len(N[x] & remaining)))
        aa0 = set([a])
        aa1 = N[a]
        while aa1:
            a2 = max(aa1, key=lambda x: (index_weight.get(x, 0), len(N[x] & aa1)))
            aa0.add(a2)
            aa1 &= N[a2]
        aaa.append(aa0)
        print (f"Selected clique: {[nodes[i] for i in aa0]} with weight {sum(index_weight.get(i, 0) for i in aa0)}")
        remaining -= aa0
        print(f"Remaining nodes: {[nodes[i] for i in remaining]}")
    # Map indices back to original node labels to match DomClique1 output
    return [sorted([nodes[i] for i in aa]) for aa in aaa]

# equation 6 from manuscript
N_delta = lambda delta: 4*(2*np.sqrt(-np.log(delta))+1)**2

##########################################################################################
### Measurement schemes used for benchmark ###############################################
##########################################################################################

class L1_sampler:
    """ Comparison class that does not reconstruct the Hamiltonian expectation value by its components, but by its relative signs. """
    
    def __init__(self,observables,weights,epsilon):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M,n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        assert epsilon > 0, "Epsilon has to be strictly positive"
        abs_vals = np.abs(weights)
        
        self.obs         = observables
        self.num_obs     = M
        self.num_qubits  = n
        self.w           = weights
        self.prob        = abs_vals / np.sum(abs_vals)
        self.eps         = epsilon
        self.shots       = 0
        self.is_sampling = True
        self.is_adaptive = False
        
        return
    
    def reset(self):
        self.shots = 0
    
    def find_setting(self,num_samples=1):
        self.shots += num_samples
        inds = np.random.choice(self.num_obs,size=(num_samples,),p=self.prob)
        return inds
        
    def get_Hoeffding_bound(self):
        return 2*np.exp(-0.5*self.eps**2*self.shots/np.sum(np.abs(self.w))**2)
    
    def get_epsilon(self,delta):
        return np.sqrt(2/self.shots*np.log(2/delta)) * np.sum(np.abs(self.w))

class Measurement_scheme:
    """ Parent class for measurement schemes. Requires
        observables: Array of shape (num_obs x num_qubits) with entries in {0,1,2,3} (the Pauli operators)
        weights:     Array of shape (num_obs) with the corresponding weight in the Hamiltonian decomposition.
                     Array is flattened upon input.
        epsilon:     Absolute error threshold, see child methods for an individual interpretation.
    """
    
    def __init__(self,observables,weights,epsilon):
        assert len(observables.shape) == 2, "Observables has to be a 2-dim array."
        M,n = observables.shape
        weights = weights.flatten()
        assert len(weights) == M, "Number of weights not matching number of provided observables."
        assert epsilon > 0, "Epsilon has to be strictly positive"
        
        self.obs           = observables
        self.num_obs       = M
        self.num_qubits    = n
        self.w             = weights
        self.eps           = epsilon
        self.scheme_params = {"eps": epsilon, "num_obs": M}
        self.N_hits        = np.zeros(M,dtype=int)
        self.N_hits_pairs    = np.zeros((M, M), dtype=int)
        self.is_adaptive   = False # useful default to be given to any child class
        self._hit_outer_cache = {}
        self.settings_dict = {}
        self.settings_buffer = {}
        self.is_overlapping = True
        return
        
    def find_setting(self):
        pass
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.N_hits_pairs  = np.zeros_like(self.N_hits_pairs)
        self._hit_outer_cache = {}
        self.settings_dict = {}
        self.settings_buffer = {}
        return
    
    def get_epsilon_sys_stat(self,delta):
        """ Applies the truncation strategy (see truncate() for details) and returns the corresponding epsilon values for the 
            systematic and the statistical error, respectively. Does not alter the scheme in-place, compared to truncate() would do.
        """
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit) # round down to integer value
        if np.sum(keep) == 0:
            # only systematic error
            eps_syst = np.sum(np.abs(self.w))
            eps_stat = 0
        elif np.sum(keep) == len(keep):
            # only statistical error
            eps_syst = 0
            eps_stat = self.get_epsilon_Bernstein(delta)
        else:
            w, N = self.w, self.N_hits
            # override temporarily
            self.w = self.w[keep]
            self.N_hits = self.N_hits[keep]
            # calculate guarantees
            eps_syst = np.sum(np.abs(w[np.bitwise_not(keep)]))
            eps_stat = self.get_epsilon_Bernstein(delta)
            # undo overwriting
            self.w = w
            self.N_hits = N
        return eps_syst, eps_stat
    
    def truncate(self,delta):
        """ Truncation function to apply the truncation criterion given a certain inconfidence level delta.
            Assumes that scheme has called the function find_setting() sufficiently often.
            Truncates all observables that fulfill the truncation criterion and save the sum of their absolute coefficient values.
            Returns the resulting introduced systematic error epsilon.
        """
        N_unmeasured = np.sum(self.N_hits == 0)
        if N_unmeasured > 0:
            print("Warning! {} observable(s) have been measured at least once.".format(N_unmeasured))
            print("If you have set alpha large, this can result in a non-optimal truncation.")
        N_crit = N_delta(delta)
        keep = self.N_hits > int(N_crit) # round down to integer value
        if np.sum(keep) == 0:
            print("No observable reached the threshold. Ensure that you have sampled often enough or provide a smaller delta!")
            print("Scheme unaltered.")
            return 0
        if np.sum(keep) == len(keep):
            print("Nothing had to be truncated.")
            return 0
        eps_sys = np.sum(np.abs(self.w[np.bitwise_not(keep)]))
        self.w = self.w[keep]
        self.obs = self.obs[keep]
        self.N_hits = self.N_hits[keep]
        return eps_sys
    

    def _append_is_hit_hit_outer(self, token, setting_indices: np.ndarray) -> np.ndarray:
        """
        Cache the information needed to apply outer(is_hit,is_hit) for a setting, keyed by `token`.
        Store the indices where is_hit==1, because, for 0/1 hits, outer(is_hit,is_hit) has ones on idx x idx.
        """
        cached = self._hit_outer_cache.get(token, None)
        if cached is not None:
            return cached

        idx = np.asarray(setting_indices, dtype=np.int32).ravel()
        if idx.size:
            # ensure canonical form (sorted unique)
            idx = np.unique(idx)
            idx.sort()

        self._hit_outer_cache[token] = idx
        return idx

class Priori_Pool(Measurement_scheme):
    """ Generates a pool of measurement settings using a graph based method and the concept of cliques. This class will provide a pool suitable for another class called 
        Best_scheme_given_pool, which will then does a convex optimization over the pool to find the best measurement scheme given the pool. 
        You should be cautious about the tokenization of settings in this class which is designed to be usable by the Best_scheme_given_pool class. 
    """
    
    def __init__(self, observables, weights, epsilon, weight_function, cov_real, compute_N_hits_pairs=True):
        # Convert Pauli strings to arrays FIRST
        #observablesarray = [pauli_string_to_array(o) for o in observables]
        # Then pass converted observables into super().__init__()
        #super().__init__(observablesarray, weights, epsilon)
        super().__init__(observables,weights,epsilon)
        #self.settings_dict = {} #222222
        self.N_hits = np.zeros_like(self.N_hits)
        self.cov_real = cov_real
        self.compute_N_hits_pairs = compute_N_hits_pairs
        if compute_N_hits_pairs:
            self.N_hits_pairs = np.zeros((self.num_obs, self.num_obs), dtype=int)
        self.weight_function = weight_function
        self.round_num = 0
        self.rounds = []
        self.selected_cliques = []  # Stores best cliques from each round
        self.cliques_with_epsilon = []
        self.settings_rounds = []
        self._cached_graph = build_hit_graph(observables)
        #self._cached_cliques = list(nx.find_cliques(self._cached_graph))
        self._cached_cliques = list(DomClique1(self._cached_graph))
        #self._cached_cliques = find_cliques5(self._cached_graph)
        self._cached_settings = []
        for clique in self._cached_cliques:
            setting_candidate = np.zeros(self.num_qubits, dtype=int)
            for o in clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting_candidate):
                    non_id = o_arr != 0
                    setting_candidate[non_id] = o_arr[non_id]
                    if np.min(setting_candidate) > 0:
                        break
            self._cached_settings.append(setting_candidate.copy())
        
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function is supposed to return an array of shape {} "
                "(i.e. number of observables) but returned an array of shape {}".format(self.w.shape, test.shape)
            )
        self.is_sampling = False
        self.commutativity_type = 'qwc'
        return

    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.N_hits_pairs  = np.zeros_like(self.N_hits_pairs)
        self._hit_outer_cache = {}
        self.settings_dict = {} #2222222222222
        self.settings_buffer = {}
        return

    # Equation 27,28 and 29
    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
        return np.sum(inconf)

    #Equation 22
    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            bound = -1
        else:
            bound = np.exp(-0.25*(self.eps/2/np.sum(np.abs(self.w)/np.sqrt(self.N_hits))-1)**2)
        return bound            

    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        if self._cached_graph is None or self._cached_cliques is None:
            self.build_graph_and_cliques()
        
        weights = self.weight_function(self.w, self.eps, self.N_hits)

        self.cliques_with_epsilon = []
        self.settings_rounds = []
        for setting_candidate in self._cached_settings:
            working = setting_candidate.copy()
            is_hit = []
            # update number of hits
            is_hit = hit_by_batch_numba(self.obs , working)
            self.N_hits += is_hit


            # Tokenize by the set of compatible observable indices
            setting_indices = np.nonzero(is_hit)[0].astype(np.int32)
            setting_indices.sort()
            token = encode_setting_token(setting_indices)
            self.settings_rounds.append(np.asarray(setting_indices, dtype=np.int32))

        # Update the dict(s) of distinct settings and how often they occur
        order_attr = getattr(self, "order", None) if hasattr(self, "order") else None
        settings_to_dict(
            self.settings_rounds,
            self.settings_dict,
            self.settings_buffer,
            order=order_attr
        )

        self.round_num += 1    
        self.rounds.append(len(self.rounds) + 1)
        if verbose:
            print("round number" , self.round_num)
        print("Pool of settings is constructed)")
        return


  
class OGM_Pool(Measurement_scheme):
    """ Comparison class to ShadowGrouping if the sampling distribution p can be provided explicitly.
        filename_for_distribution: string that points to the file containing the distribution and its corresponding settings
            see load_distribution_setting() for further information of data formatting.
        epsilon (optional): parameter solely used for comparison with other methods. Defaults to 0.1.
        
        Returns p and a dictionary info holding further details on the matching procedure.
        Note that due to the sampling, find_setting() can yield multiple settings.
    """
    def __init__(self,observables,weights,filename_for_distribution,epsilon=0.1):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.load_distribution_setting(filename_for_distribution)
        self.is_sampling = True
        self.commutativity_type = 'qwc'
        self.settings_rounds = []
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.settings_dict = {}
        self.settings_buffer = {}
        return

    def load_distribution_setting(self,filename):
        """ Helper function to read the distribution and the corresponding settings from file.
            Data must be stored as a matrix of form (N+1,n) where n = # qubits and N = # settings.
            The last row corresponds to the entries of the distribution
        """
        data = np.loadtxt(filename)
        self.p = data[-1]
        self.settings = data[:-1].T
        return

    def find_setting(self):
        """ Generate settings from the given distribution p. Can find multiple settings at once by providing a value for
            N_samples (int). Returns the setting(s) and a dictionary holding the information about the number of settings sampled.
        """
        print("length of self.settings:", len(self.settings))
        for settingcandidate in self.settings:
            working = settingcandidate.copy()
            #working = working.flatten()
            is_hit = []
            is_hit = hit_by_batch_numba(self.obs , working)
            self.N_hits += is_hit
            # Tokenize by the set of compatible observable indices
            setting_indices = np.nonzero(is_hit)[0].astype(np.int32)
            setting_indices.sort()
            token = encode_setting_token(setting_indices)
            self.settings_rounds.append(np.asarray(setting_indices, dtype=np.int32))
            # update number of hits
            

        # Update the dict(s) of distinct settings and how often they occur
        order_attr = getattr(self, "order", None) if hasattr(self, "order") else None
        settings_to_dict(
            self.settings_rounds,
            self.settings_dict,
            self.settings_buffer,
            order=order_attr
        )
        
        return
    

class AEQUO_Pool(Measurement_scheme):
    """ Generates a pool of measurement settings using a graph based method and the concept of cliques. This class will provide a pool suitable for another class called 
        Best_scheme_given_pool, which will then does a convex optimization over the pool to find the best measurement scheme given the pool. 
        You should be cautious about the tokenization of settings in this class which is designed to be usable by the Best_scheme_given_pool class. 
    """
    
    def __init__(self, observables, weights, epsilon, weight_function, cov_real, compute_N_hits_pairs=True):
        # Convert Pauli strings to arrays FIRST
        #observablesarray = [pauli_string_to_array(o) for o in observables]
        # Then pass converted observables into super().__init__()
        #super().__init__(observablesarray, weights, epsilon)
        super().__init__(observables,weights,epsilon)
        #self.settings_dict = {} #222222
        self.N_hits = np.zeros_like(self.N_hits)
        self.cov_real = cov_real
        self.compute_N_hits_pairs = compute_N_hits_pairs
        if compute_N_hits_pairs:
            self.N_hits_pairs = np.zeros((self.num_obs, self.num_obs), dtype=int)
        self.weight_function = weight_function
        self.round_num = 0
        self.rounds = []
        self.selected_cliques = []  # Stores best cliques from each round
        self.cliques_with_epsilon = []
        self.settings_rounds = []
        self._cached_graph = build_hit_graph(observables)
        self._cached_cliques = list(LDF(self._cached_graph))
        #self._cached_cliques = list(DomClique1(self._cached_graph))
        self._cached_settings = []
        for clique in self._cached_cliques:
            setting_candidate = np.zeros(self.num_qubits, dtype=int)
            for o in clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting_candidate):
                    non_id = o_arr != 0
                    setting_candidate[non_id] = o_arr[non_id]
                    if np.min(setting_candidate) > 0:
                        break
            if np.min(setting_candidate) == 0:
                setting_candidate[setting_candidate == 0] = 3 
            self._cached_settings.append(setting_candidate.copy())
        
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function is supposed to return an array of shape {} "
                "(i.e. number of observables) but returned an array of shape {}".format(self.w.shape, test.shape)
            )
        self.is_sampling = False
        self.commutativity_type = 'qwc'
        return

    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.N_hits_pairs  = np.zeros_like(self.N_hits_pairs)
        self._hit_outer_cache = {}
        self.settings_dict = {}
        self.settings_buffer = {}
        return

    # Equation 27,28 and 29
    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
        return np.sum(inconf)

    #Equation 22
    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            bound = -1
        else:
            bound = np.exp(-0.25*(self.eps/2/np.sum(np.abs(self.w)/np.sqrt(self.N_hits))-1)**2)
        return bound            

    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        if self._cached_graph is None or self._cached_cliques is None:
            self.build_graph_and_cliques()
        
        weights = self.weight_function(self.w, self.eps, self.N_hits)
        #print("cached settings:", self._cached_settings)
        self.cliques_with_epsilon = []
        self.settings_rounds = []
        for setting_candidate in self._cached_settings:
            working = setting_candidate.copy()
            is_hit = []
            # update number of hits
            is_hit = hit_by_batch_numba(self.obs , working)
            self.N_hits += is_hit
            # Tokenize by the set of compatible observable indices
            setting_indices = np.nonzero(is_hit)[0].astype(np.int32)
            setting_indices.sort()
            token = encode_setting_token(setting_indices)
            self.settings_rounds.append(np.asarray(setting_indices, dtype=np.int32))

        # Update the dict(s) of distinct settings and how often they occur
        order_attr = getattr(self, "order", None) if hasattr(self, "order") else None
        settings_to_dict(
            self.settings_rounds,
            self.settings_dict,
            self.settings_buffer,
            order=order_attr
        )

        self.round_num += 1    
        self.rounds.append(len(self.rounds) + 1)
        if verbose:
            print("round number" , self.round_num)
        print("Pool of settings is constructed")
        return


class AEQUO_Pool_weightbased(Measurement_scheme):
    """ Generates a pool of measurement settings using a graph based method and the concept of cliques. This class will provide a pool suitable for another class called 
        Best_scheme_given_pool, which will then does a convex optimization over the pool to find the best measurement scheme given the pool. 
        You should be cautious about the tokenization of settings in this class which is designed to be usable by the Best_scheme_given_pool class. 
    """
    
    def __init__(self, observables, weights, epsilon, weight_function, cov_real, compute_N_hits_pairs=True):
        # Convert Pauli strings to arrays FIRST
        #observablesarray = [pauli_string_to_array(o) for o in observables]
        # Then pass converted observables into super().__init__()
        #super().__init__(observablesarray, weights, epsilon)
        super().__init__(observables,weights,epsilon)
        #self.settings_dict = {} #222222
        self.N_hits = np.zeros_like(self.N_hits)
        self.cov_real = cov_real
        self.compute_N_hits_pairs = compute_N_hits_pairs
        if compute_N_hits_pairs:
            self.N_hits_pairs = np.zeros((self.num_obs, self.num_obs), dtype=int)
        self.weight_function = weight_function
        self.round_num = 0
        self.rounds = []
        self.selected_cliques = []  # Stores best cliques from each round
        self.cliques_with_epsilon = []
        self.settings_rounds = []
        # create weight_map mapping observable tuples to their corresponding weights
        weight_map = {tuple(obs): w for obs, w in zip(observables, weights)}
        self._cached_graph = build_hit_graph2(observables, weight_map)
        self._cached_cliques = list(LDF_weighted(self._cached_graph))
        #self._cached_cliques = list(DomClique1(self._cached_graph))
        self._cached_settings = []
        for clique in self._cached_cliques:
            setting_candidate = np.zeros(self.num_qubits, dtype=int)
            for o in clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting_candidate):
                    non_id = o_arr != 0
                    setting_candidate[non_id] = o_arr[non_id]
                    if np.min(setting_candidate) > 0:
                        break
            if np.min(setting_candidate) == 0:
                setting_candidate[setting_candidate == 0] = 3 
            self._cached_settings.append(setting_candidate.copy())
        
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function is supposed to return an array of shape {} "
                "(i.e. number of observables) but returned an array of shape {}".format(self.w.shape, test.shape)
            )
        self.is_sampling = False
        self.commutativity_type = 'qwc'
        return

    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.N_hits_pairs  = np.zeros_like(self.N_hits_pairs)
        self._hit_outer_cache = {}
        self.settings_dict = {}
        self.settings_buffer = {}
        return

    # Equation 27,28 and 29
    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
        return np.sum(inconf)

    #Equation 22
    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            bound = -1
        else:
            bound = np.exp(-0.25*(self.eps/2/np.sum(np.abs(self.w)/np.sqrt(self.N_hits))-1)**2)
        return bound            

    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        if self._cached_graph is None or self._cached_cliques is None:
            self.build_graph_and_cliques()
        
        weights = self.weight_function(self.w, self.eps, self.N_hits)
        #print("cached settings:", self._cached_settings)
        self.cliques_with_epsilon = []
        self.settings_rounds = []
        for setting_candidate in self._cached_settings:
            working = setting_candidate.copy()
            is_hit = []
            # update number of hits
            is_hit = hit_by_batch_numba(self.obs , working)
            self.N_hits += is_hit
            # Tokenize by the set of compatible observable indices
            setting_indices = np.nonzero(is_hit)[0].astype(np.int32)
            setting_indices.sort()
            token = encode_setting_token(setting_indices)
            self.settings_rounds.append(np.asarray(setting_indices, dtype=np.int32))

        # Update the dict(s) of distinct settings and how often they occur
        order_attr = getattr(self, "order", None) if hasattr(self, "order") else None
        settings_to_dict(
            self.settings_rounds,
            self.settings_dict,
            self.settings_buffer,
            order=order_attr
        )

        if np.min(self.N_hits) == 0:
            print("Warning: Some observables have not been hit at all. Consider running more rounds or check the settings.")

        self.round_num += 1    
        self.rounds.append(len(self.rounds) + 1)
        if verbose:
            print("round number" , self.round_num)
        print("Pool of settings is constructed")
        return


class OGM_NPBC(Measurement_scheme):
    """ Generates a pool of measurement settings using a graph based method and the concept of cliques. This class will provide a pool suitable for another class called 
        Best_scheme_given_pool, which will then does a convex optimization over the pool to find the best measurement scheme given the pool. 
        You should be cautious about the tokenization of settings in this class which is designed to be usable by the Best_scheme_given_pool class. 
    """
    
    def __init__(self, observables, weights, epsilon, weight_function, cov_real, compute_N_hits_pairs=True):
        # Convert Pauli strings to arrays FIRST
        #observablesarray = [pauli_string_to_array(o) for o in observables]
        # Then pass converted observables into super().__init__()
        #super().__init__(observablesarray, weights, epsilon)
        super().__init__(observables,weights,epsilon)
        #self.settings_dict = {} #222222
        self.N_hits = np.zeros_like(self.N_hits)
        self.cov_real = cov_real
        self.compute_N_hits_pairs = compute_N_hits_pairs
        if compute_N_hits_pairs:
            self.N_hits_pairs = np.zeros((self.num_obs, self.num_obs), dtype=int)
        self.weight_function = weight_function
        self.round_num = 0
        self.rounds = []
        self.selected_cliques = []  # Stores best cliques from each round
        self.cliques_with_epsilon = []
        self.settings_rounds = []
        self._cached_settings = []
        # create weight_map mapping observable tuples to their corresponding weights
        #weight_map = {tuple(obs): w for obs, w in zip(observables, weights)}
        """self._cached_graph = build_hit_graph2(observables, weight_map)
        self._cached_cliques = list(sorted_NPBC(self._cached_graph))
        self._cached_cliques = list(DomClique1(self._cached_graph))
        self._cached_settings = []
        for clique in self._cached_cliques:
            setting_candidate = np.zeros(self.num_qubits, dtype=int)
            for o in clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting_candidate):
                    non_id = o_arr != 0
                    setting_candidate[non_id] = o_arr[non_id]
                    if np.min(setting_candidate) > 0:
                        break
            if np.min(setting_candidate) == 0:
                setting_candidate[setting_candidate == 0] = 3 
            self._cached_settings.append(setting_candidate.copy())"""
        
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function is supposed to return an array of shape {} "
                "(i.e. number of observables) but returned an array of shape {}".format(self.w.shape, test.shape)
            )
        self.is_sampling = False
        self.commutativity_type = 'qwc'
        return

    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.N_hits_pairs  = np.zeros_like(self.N_hits_pairs)
        self._hit_outer_cache = {}
        self.settings_dict = {}
        self.settings_buffer = {}
        return

    # Equation 27,28 and 29
    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
        return np.sum(inconf)

    #Equation 22
    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            bound = -1
        else:
            bound = np.exp(-0.25*(self.eps/2/np.sum(np.abs(self.w)/np.sqrt(self.N_hits))-1)**2)
        return bound            

    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        order = np.argsort(self.w)[::-1]      # descending weights
        remaining = list(order)
        while remaining:
            # Highest-weight remaining observable becomes seed
            seed_idx = remaining.pop(0)
            setting = self.obs[seed_idx].copy()
            grouped_indices = [seed_idx]
            # Scan observables from largest weight to smallest
            for idx in order:
                o = self.obs[idx]
                if verbose:
                    print("Checking", o)
                if hit_by(o, setting):
                    if idx not in remaining:
                        continue
                    remaining.remove(idx)
                    grouped_indices.append(idx)
                    non_id = (o != 0)
                    # Fill identities in the setting
                    setting[non_id] = o[non_id]
                    if verbose:
                        print("p =", setting)
                    # Stop if setting has no identities left
                    if np.min(setting) > 0:
                        break
            if np.min(setting) == 0:
                setting[setting == 0] = 3
            self._cached_settings.append(setting.copy())
        print("cached settings:", self._cached_settings)
        self.cliques_with_epsilon = []
        self.settings_rounds = []
        for setting_candidate in self._cached_settings:
            working = setting_candidate.copy()
            is_hit = []
            # update number of hits
            is_hit = hit_by_batch_numba(self.obs , working)
            self.N_hits += is_hit
            # Tokenize by the set of compatible observable indices
            setting_indices = np.nonzero(is_hit)[0].astype(np.int32)
            setting_indices.sort()
            token = encode_setting_token(setting_indices)
            self.settings_rounds.append(np.asarray(setting_indices, dtype=np.int32))

        # Update the dict(s) of distinct settings and how often they occur
        order_attr = getattr(self, "order", None) if hasattr(self, "order") else None
        settings_to_dict(
            self.settings_rounds,
            self.settings_dict,
            self.settings_buffer,
            order=order_attr
        )

        self.round_num += 1    
        self.rounds.append(len(self.rounds) + 1)
        if verbose:
            print("round number" , self.round_num)
        print("Pool of settings is constructed")
        return

