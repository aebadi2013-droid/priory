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
from shadowgrouping_v2.guarantees import (get_epsilon_Chebyshev_scalar_tighter_numba, get_epsilon_Chebyshev_scalar_tightest_numba)
from shadowgrouping_v2.allocation_mixin import _AllocationMixin
from shadowgrouping_v2.shadowgrouping_my_dev.helper_functions import settings_to_dict
from shadowgrouping_v2.shadowgrouping_my_dev.full_commutativity import (
    fully_commute_batched, _GF2LinearBasis, _export_basis_compact, 
    in_span_batch_numba, fc_compat_row_numba,
    compute_exact_compat_degrees_qwc,
    compute_exact_compat_degrees_fc,
    compute_approx_compat_degrees_qwc,
    compute_approx_compat_degrees_fc)


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
        # Unique settings tracking + amortized is_hit storage
        self.seen_settings = set()
        self._is_hit_cap       = 16  # initial capacity
        self._is_hit_rows_used = 0
        self._is_hit_buf       = np.empty((self._is_hit_cap, self.num_obs), dtype=bool)
        return
    
    def _promote_forced_idx(self, order_desc, forced_idx):
        if forced_idx is None:
            return order_desc
    
        forced_idx = int(forced_idx)
        if forced_idx < 0 or forced_idx >= self.num_obs:
            raise ValueError(f"forced_idx={forced_idx} is out of range for num_obs={self.num_obs}.")
    
        # if already first, nothing to do
        if order_desc.size > 0 and int(order_desc[0]) == forced_idx:
            return order_desc
    
        mask = (order_desc != forced_idx)
        return np.concatenate((
            np.array([forced_idx], dtype=order_desc.dtype),
            order_desc[mask]
        ))

    # helper for growing is_hit buffer
    def _ensure_is_hit_capacity(self, needed_rows: int) -> None:
        if needed_rows <= self._is_hit_cap:
            return
        new_cap = max(self._is_hit_cap * 2, needed_rows)
        new_buf = np.empty((new_cap, self.num_obs), dtype=bool)
        if self._is_hit_rows_used:
            new_buf[:self._is_hit_rows_used] = self._is_hit_buf[:self._is_hit_rows_used]
        self._is_hit_buf = new_buf
        self._is_hit_cap = new_cap

    def _append_is_hit_row(self, is_hit_row: np.ndarray) -> None:
        """Append one new unique is_hit row (bool shape (M,))."""
        self._ensure_is_hit_capacity(self._is_hit_rows_used + 1)
        self._is_hit_buf[self._is_hit_rows_used] = is_hit_row
        self._is_hit_rows_used += 1

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
        self.is_overlapping = False
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
            is_hit[self.N_hits >= 2] = 0    # or False if is_hit is bool
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
        self.is_overlapping = False
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
            is_hit[self.N_hits >= 2] = 0    # or False if is_hit is bool
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


class OGM_PBC(Measurement_scheme):
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
        self.commutativity_type = commutativity_type
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
            
            # Find the seed's position in the original order
            seed_position = np.where(order == seed_idx)[0][0]

            # Start after the seed, reach the end, then wrap to the beginning
            cyclic_order = np.concatenate((
                order[seed_position + 1:],
                order[:seed_position]
            ))
            

            for idx in cyclic_order:
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
    


class LDF_Sorted_Insertion_OGM(_AllocationMixin, Measurement_scheme):
    """
    Sorted Insertion / OGM / LDF grouping scheme for energy estimation.

    This class is responsible for generating groups of jointly measurable
    Pauli strings. Budget allocation is inherited from `_AllocationMixin`.

    Group construction
    ------------------
    Supports both QWC and fully commuting grouping:

      - commutativity_type = "qwc":
            qubit-wise commuting groups;

      - commutativity_type = "fc":
            fully commuting groups, constructed through a generator-based
            symplectic representation and full-span harvesting.

    Supports both non-overlapping and overlapping grouping:

      - is_overlapping = False:
            non-overlapping partition. The ordering of the Pauli strings is
            controlled by ordering_strategy:

              ordering_strategy = "coefficient":
                    standard Sorted Insertion, ordered by decreasing |alpha_j|;

              ordering_strategy = "ldf_exact":
                    Largest-Degree-First ordering for the anti-compatibility
                    graph, using exact compatibility degrees. Since the
                    auxiliary functions compute compatibility degrees, this
                    is implemented by sorting by increasing compatibility
                    degree;

              ordering_strategy = "ldf_approx":
                    same LDF idea, but using approximate compatibility degrees.

      - is_overlapping = True and overlap_strategy = "cyclic":
            standard OGM / Algorithm-1-style cyclic-start overlapping covering.
            Each new group starts from the highest-rank Pauli string that has
            not yet appeared in a group, and the scan order is cyclic:
            seed, seed+1, ..., end, beginning, ..., seed-1.

      - is_overlapping = True and overlap_strategy = "top_first":
            Algorithm-4-style top-first overlapping covering, without the
            U_k token restriction. Each new group is forced to contain the
            highest-rank Pauli string that has not yet appeared in a group,
            but all remaining candidates are then scanned from the top of the
            global ranking.

    Notes
    -----
    The LDF ordering strategies are intended only for the non-overlapping path.
    For overlapping OGM-style grouping, the ranking remains coefficient-based.

    Allocation objectives
    ---------------------
    The inherited `_AllocationMixin.allocate_budget()` supports two objective
    choices for energy estimation:

      - allocation_objective = "variance":

            minimize the diagonal variance proxy

                sum_j |alpha_j|^2 / N_j,

            where alpha_j is the Hamiltonian coefficient of Pauli string j
            and N_j is its effective number of samples.

      - allocation_objective = "bernstein_l1":

            minimize the ShadowGrouping/Bernstein-style proxy

                sum_j |alpha_j| / sqrt(N_j).

    Optional truncation
    -------------------
    If attempt_truncation = True, only available for overlapping informed
    allocation, the allocation backend tries different active sets of Pauli
    strings ranked by |alpha_j|. Inactive Pauli strings do not drive the
    optimization objective, but they may still be measured incidentally if they
    appear in selected groups.

    Workflow
    --------
      1) find_groups()
      2) allocate_budget(), inherited from _AllocationMixin
    """

    def __init__(self, observables, weights, epsilon=0.1, total_rounds=0,
                 is_overlapping=False, commutativity_type="qwc", *,
                 ordering_strategy="coefficient",
                 overlap_strategy="cyclic",
                 informed_allocation=True, allocation_objective="bernstein_l1",
                 attempt_truncation=False, rounding_strategy="largest_fraction",
                 md_gap_tol_rel=1e-4, prior_counts=None,
                 allocate_budget_in_init=True):

        super().__init__(observables, weights, epsilon)
        self.save_scheme=False
        self.is_overlapping = bool(is_overlapping)
        self.is_sampling = False

        self.commutativity_type = str(commutativity_type).lower()
        if self.commutativity_type not in ("qwc", "fc"):
            raise ValueError("commutativity_type must be 'qwc' or 'fc'.")

        # New option for the non-overlapping path.
        self.ordering_strategy = str(ordering_strategy).lower()
        if self.ordering_strategy not in ("coefficient", "ldf_exact", "ldf_approx"):
            raise ValueError(
                "ordering_strategy must be one of "
                "'coefficient', 'ldf_exact', or 'ldf_approx'. "
                f"Got {ordering_strategy!r}."
            )

        if self.is_overlapping and self.ordering_strategy != "coefficient":
            raise ValueError(
                "ordering_strategy='ldf_exact' and 'ldf_approx' are currently "
                "defined only for the non-overlapping sorted-insertion path. "
                "For overlapping OGM-style grouping, use "
                "ordering_strategy='coefficient'."
            )

        # Existing option for the overlapping path.
        self.overlap_strategy = str(overlap_strategy).lower()
        if self.overlap_strategy not in ("cyclic", "top_first"):
            raise ValueError(
                "overlap_strategy must be either 'cyclic' or 'top_first'. "
                f"Got {overlap_strategy!r}."
            )

        self.total_rounds = int(total_rounds)
        if self.total_rounds < 0:
            raise ValueError("total_rounds must be >= 0.")

        # Allocation options used by _AllocationMixin.
        self.informed_allocation = bool(informed_allocation)

        self.allocation_objective = str(allocation_objective).lower()
        if self.allocation_objective not in ("variance", "bernstein_l1"):
            raise ValueError(
                "allocation_objective must be either 'variance' or 'bernstein_l1'. "
                f"Got {allocation_objective!r}."
            )

        self.attempt_truncation = bool(attempt_truncation)

        self.rounding_strategy = str(rounding_strategy).lower()
        if self.rounding_strategy not in ("largest_fraction", "marginal"):
            raise ValueError(
                "rounding_strategy must be either 'largest_fraction' or 'marginal'. "
                f"Got {rounding_strategy!r}."
            )

        self.md_gap_tol_rel = float(md_gap_tol_rel)
        if self.md_gap_tol_rel <= 0.0:
            raise ValueError("md_gap_tol_rel must be positive.")

        if self.attempt_truncation and not self.informed_allocation:
            raise ValueError(
                "attempt_truncation=True requires informed_allocation=True, "
                "because truncation is defined through the informed allocation objective."
            )

        if self.attempt_truncation and not self.is_overlapping:
            raise ValueError(
                "attempt_truncation=True is currently implemented only for "
                "overlapping OGM-style allocation, i.e. is_overlapping=True."
            )

        # Optional prior samples per Pauli string.
        if prior_counts is None:
            self.prior_counts = np.zeros(self.num_obs, dtype=np.int64)
        else:
            prior_counts_arr = np.asarray(prior_counts)

            if prior_counts_arr.shape != (self.num_obs,):
                prior_counts_arr = prior_counts_arr.reshape(-1)

            if prior_counts_arr.shape != (self.num_obs,):
                raise ValueError(
                    f"prior_counts must have shape ({self.num_obs},), "
                    f"got {prior_counts_arr.shape}."
                )

            if not np.all(np.isfinite(prior_counts_arr)):
                raise ValueError("prior_counts must contain only finite values.")

            if np.any(prior_counts_arr < 0):
                raise ValueError("prior_counts must be nonnegative.")

            if not np.allclose(prior_counts_arr, np.round(prior_counts_arr)):
                raise ValueError("prior_counts must contain integer sample counts.")

            self.prior_counts = np.round(prior_counts_arr).astype(np.int64, copy=True)

        # Group pool and diagnostics.
        self.cliques_pool = []
        self._setting_cursor = 0
        self.allocation_info = {
            "attempted": False,
            "mode": None,
        }

        self.truncation_info = {
            "attempted": False,
            "selected": False,
        }

        # FC-specific scalable cache.
        if self.commutativity_type == "fc":
            self._build_symplectic_cache()
            self._fc_row_cache = {}
            self.last_generator_indices = np.empty(0, dtype=np.int32)

        # Build groups and optionally allocate the measurement budget.
        self.find_groups()

        if allocate_budget_in_init:
            self.allocate_budget()

    def reset(self, updated_total_rounds=None, clear_prior_counts=False):
        """
        Reset the newly allocated measurement scheme and re-run budget allocation
        for the existing group pool.

        Parameters
        ----------
        updated_total_rounds : int or None
            If provided, update the total number of measurement rounds before
            reallocating.

        clear_prior_counts : bool
            If True, set all prior counts to zero before reallocating.
        """
        if updated_total_rounds is not None:
            self.total_rounds = int(updated_total_rounds)
            if self.total_rounds < 0:
                raise ValueError("total_rounds must be >= 0.")

        if clear_prior_counts:
            self.prior_counts[:] = 0

        if self.commutativity_type == "fc":
            self.last_generator_indices = np.empty(0, dtype=np.int32)
            self._fc_row_cache = {}

        self.allocate_budget()
        self._setting_cursor = 0
    # ------------------------------------------------------------------
    # Ordering helper
    # ------------------------------------------------------------------

    def _compute_grouping_order(self) -> np.ndarray:
        """
        Compute the Pauli-string ranking used for group construction.

        Returns
        -------
        order : np.ndarray[int64]
            Indices of Pauli strings in the order in which they should be
            considered during group construction.

        Ordering strategies
        -------------------
        coefficient:
            Sort by decreasing |alpha_j|.

        ldf_exact:
            Approximate Largest-Degree-First on the anti-compatibility graph
            using exact compatibility degrees. Since the auxiliary degree
            routines compute compatibility degrees, we sort by increasing
            compatibility degree.

        ldf_approx:
            Same as ldf_exact, but using approximate compatibility degrees.

        Notes
        -----
        LDF strategies are intended only for the non-overlapping path. This is
        enforced in __init__.
        """
        M = self.num_obs
        absw = np.abs(np.asarray(self.w, dtype=np.float64))
        tie_index = np.arange(M, dtype=np.int64)

        if self.ordering_strategy == "coefficient":
            # Primary key: -absw. Secondary deterministic tie-breaker: index.
            return np.lexsort((tie_index, -absw)).astype(np.int64)

        obs_int8 = np.asarray(self.obs, dtype=np.int8)

        if self.ordering_strategy == "ldf_exact":
            if self.commutativity_type == "qwc":
                compat_degrees = compute_exact_compat_degrees_qwc(obs_int8)
            elif self.commutativity_type == "fc":
                compat_degrees = compute_exact_compat_degrees_fc(obs_int8)
            else:
                raise ValueError(
                    "Unsupported commutativity_type for LDF ordering: "
                    f"{self.commutativity_type!r}."
                )

        elif self.ordering_strategy == "ldf_approx":
            # Fixed hyperparameter for approximate LDF ranking.
            # This is intentionally not exposed as an __init__ argument.
            n_degree_samples = 1000

            if self.commutativity_type == "qwc":
                compat_degrees = compute_approx_compat_degrees_qwc(
                    obs_int8,
                    n_samples=n_degree_samples,
                )
            elif self.commutativity_type == "fc":
                compat_degrees = compute_approx_compat_degrees_fc(
                    obs_int8,
                    n_samples=n_degree_samples,
                )
            else:
                raise ValueError(
                    "Unsupported commutativity_type for LDF ordering: "
                    f"{self.commutativity_type!r}."
                )

        else:
            raise ValueError(
                "Unsupported ordering_strategy. Expected 'coefficient', "
                "'ldf_exact', or 'ldf_approx', got "
                f"{self.ordering_strategy!r}."
            )

        compat_degrees = np.asarray(compat_degrees, dtype=np.float64)

        if compat_degrees.shape != (M,):
            compat_degrees = compat_degrees.reshape(-1)

        if compat_degrees.shape != (M,):
            raise ValueError(
                f"Compatibility degree array must have shape ({M},), "
                f"got {compat_degrees.shape}."
            )

        if not np.all(np.isfinite(compat_degrees)):
            raise ValueError("Compatibility degrees contain non-finite values.")

        # LDF on the anti-compatibility graph means largest anti-degree first.
        # Since anti_degree_i = M - compat_degree_i, this is equivalent to
        # sorting by increasing compatibility degree.
        #
        # Tie-breakers:
        #   1) larger |alpha_j|;
        #   2) smaller original index for determinism.
        return np.lexsort((tie_index, -absw, compat_degrees)).astype(np.int64)

    # ------------------------------------------------------------------
    # FC helpers
    # ------------------------------------------------------------------

    def _build_symplectic_cache(self) -> None:
        """
        Build:
          self._x_u64, self._z_u64, self._packed_u64 as NumPy uint64 arrays,
          and self._x, self._z, self._packed as Python-int mirrors.

        Assumes num_qubits <= 32 so that the packed symplectic representation
        fits into one uint64.
        """
        n = self.num_qubits

        if n > 32:
            raise ValueError(
                "This FC implementation of LDF_Sorted_Insertion_OGM assumes "
                f"num_qubits <= 32. Got num_qubits={n}. "
                "For larger n, use a chunked symplectic backend."
            )

        bitweights = (1 << np.arange(n, dtype=np.uint64))

        x_bits = ((self.obs == 1) | (self.obs == 2)).astype(np.uint64)
        z_bits = ((self.obs == 2) | (self.obs == 3)).astype(np.uint64)

        x_u64 = x_bits @ bitweights
        z_u64 = z_bits @ bitweights

        packed_u64 = x_u64 | (z_u64 << np.uint64(n))

        self._x_u64 = x_u64.astype(np.uint64, copy=False)
        self._z_u64 = z_u64.astype(np.uint64, copy=False)
        self._packed_u64 = packed_u64.astype(np.uint64, copy=False)

        self._x = [int(v) for v in self._x_u64.tolist()]
        self._z = [int(v) for v in self._z_u64.tolist()]
        self._packed = [int(v) for v in self._packed_u64.tolist()]

    def _get_fc_compat_row(self, idx: int) -> np.ndarray:
        """
        Lazy row retrieval for full commutativity.

        Returns a bool array row[j] = True if observable idx commutes with
        observable j.
        """
        row = self._fc_row_cache.get(idx)

        if row is not None:
            return row

        xi = self._x_u64[idx]
        zi = self._z_u64[idx]

        row = fc_compat_row_numba(self._x_u64, self._z_u64, xi, zi)
        self._fc_row_cache[idx] = row

        return row

    def _build_fc_group_from_order(self, order: np.ndarray,
                                   active_mask=None,
                                   restrict_to_active=False):
        """
        Build one fully commuting group by:

          1) greedily selecting an independent commuting generator basis
             along `order`;

          2) harvesting all observables in the span of those generators.

        This is the helper used by:
          - non-overlapping FC sorted insertion / LDF;
          - overlapping FC OGM with overlap_strategy = "cyclic".

        Parameters
        ----------
        order : np.ndarray
            Candidate scan order.

        active_mask : np.ndarray[bool] or None
            If provided, only indices with active_mask[idx] == True may be used
            as generator candidates.

        restrict_to_active : bool
            If True, the harvested span is intersected with active_mask. This
            is used in the non-overlapping case.

        Returns
        -------
        group_idx : np.ndarray[int32]
            Observable indices in the harvested fully commuting group.

        gen_indices : np.ndarray[int32]
            Indices of the chosen independent generators.
        """
        n = self.num_qubits
        basis = _GF2LinearBasis(max_bits=2 * n)

        gen_indices = []
        compat_mask = np.ones(self.num_obs, dtype=np.bool_)

        for idx in order:
            idxi = int(idx)

            if active_mask is not None and not active_mask[idxi]:
                continue

            if not compat_mask[idxi]:
                continue

            v = self._packed[idxi]

            if basis.add(v):
                gen_indices.append(idxi)

                row = self._get_fc_compat_row(idxi)
                compat_mask &= row

                if basis.rank >= n:
                    break

        gen_indices = np.asarray(gen_indices, dtype=np.int32)

        if gen_indices.size == 0:
            return np.empty(0, dtype=np.int32), gen_indices

        basis_rows_u64, pivot_bits_u8 = _export_basis_compact(basis)

        selected_mask = in_span_batch_numba(
            self._packed_u64,
            basis_rows_u64,
            pivot_bits_u8,
        ).astype(bool, copy=False)

        if restrict_to_active and active_mask is not None:
            selected_mask &= active_mask
        group_idx = np.flatnonzero(selected_mask).astype(np.int32)

        return group_idx, gen_indices

    def _build_fc_group_from_forced_seed(self, seed_idx: int,
                                         scan_order: np.ndarray,
                                         active_mask=None,
                                         restrict_to_active=False):
        """
        Build one fully commuting group by forcing `seed_idx` into the
        generated Abelian subgroup first, then scanning `scan_order`.

        This is the helper used by overlapping FC OGM with

            overlap_strategy = "top_first".

        The intended Algorithm-4-style behavior is:

          1) choose the highest-rank observable not yet covered;
          2) force it to be the first generator, when it is non-identity;
          3) scan the full ranking from the top;
          4) greedily add compatible independent generators;
          5) harvest all target observables in the span of the final generator
             basis.

        Parameters
        ----------
        seed_idx : int
            Observable index that must be covered by the new group.

        scan_order : np.ndarray
            Candidate scan order. For the Algorithm-4-style variant this should
            be the full global ranking `order`, not a cyclically shifted order.

        active_mask : np.ndarray[bool] or None
            If provided, only indices with active_mask[idx] == True may be used
            as generator candidates. For overlapping OGM this should normally
            be None.

        restrict_to_active : bool
            If True, the harvested span is intersected with active_mask. This
            is not normally used for overlapping OGM, but is kept for symmetry
            with `_build_fc_group_from_order`.

        Returns
        -------
        group_idx : np.ndarray[int32]
            Observable indices in the harvested fully commuting group.

        gen_indices : np.ndarray[int32]
            Indices of the chosen independent generators. If `seed_idx` is an
            identity string, it is covered by the span but is not stored as a
            generator, because the zero symplectic vector cannot increase the
            GF(2) rank.
        """
        seed_idx = int(seed_idx)

        if seed_idx < 0 or seed_idx >= self.num_obs:
            raise IndexError(
                f"seed_idx must be in [0, {self.num_obs}), got {seed_idx}."
            )

        if active_mask is not None and not active_mask[seed_idx]:
            raise ValueError(
                "The forced seed is inactive according to active_mask. "
                "This indicates inconsistent group-construction logic."
            )

        n = self.num_qubits
        basis = _GF2LinearBasis(max_bits=2 * n)

        gen_indices = []
        compat_mask = np.ones(self.num_obs, dtype=np.bool_)

        # Force the seed first. For a non-identity Pauli string this adds the
        # seed as the first independent generator. For the identity string,
        # basis.add(0) returns False; this is fine because identity is always
        # in the span of every generator basis.
        seed_v = self._packed[seed_idx]

        if basis.add(seed_v):
            gen_indices.append(seed_idx)

            row = self._get_fc_compat_row(seed_idx)
            compat_mask &= row

        # Now scan from the top of the ranking, rather than cyclically from
        # the seed. This is the defining difference of the top_first variant.
        for idx in scan_order:
            idxi = int(idx)

            if idxi == seed_idx:
                continue

            if active_mask is not None and not active_mask[idxi]:
                continue

            if not compat_mask[idxi]:
                continue

            v = self._packed[idxi]

            if basis.add(v):
                gen_indices.append(idxi)

                row = self._get_fc_compat_row(idxi)
                compat_mask &= row

                if basis.rank >= n:
                    break

        gen_indices = np.asarray(gen_indices, dtype=np.int32)

        # If the forced seed is the identity, the empty basis still spans the
        # identity string, so the group should contain at least identity
        # observables.
        basis_rows_u64, pivot_bits_u8 = _export_basis_compact(basis)

        selected_mask = in_span_batch_numba(
            self._packed_u64,
            basis_rows_u64,
            pivot_bits_u8,
        ).astype(bool, copy=False)

        if restrict_to_active and active_mask is not None:
            selected_mask &= active_mask
        group_idx = np.flatnonzero(selected_mask).astype(np.int32)


        return group_idx, gen_indices

    # ------------------------------------------------------------------
    # Group post-processing and construction
    # ------------------------------------------------------------------

    def _postprocess_groups(self, groups):
        """
        Canonicalize, deduplicate, and coverage-repair a list of observable groups.

        Steps:
          1) remove empty groups;
          2) convert each group to sorted unique np.int32 indices;
          3) deduplicate groups by encode_setting_token while preserving first
             occurrence;
          4) ensure every observable appears in at least one group by adding
             singleton groups for any missing observable.
        """
        out = []
        seen = set()

        for g in groups:
            arr = np.asarray(g, dtype=np.int32).ravel()

            if arr.size == 0:
                continue

            arr = np.unique(arr)
            arr.sort()

            token = encode_setting_token(arr)

            if token in seen:
                continue

            seen.add(token)
            out.append(arr)

        if out:
            covered = np.zeros(self.num_obs, dtype=bool)

            for arr in out:
                covered[arr] = True

            if not np.all(covered):
                missing = np.flatnonzero(~covered).astype(np.int32)

                for v in missing:
                    out.append(np.array([int(v)], dtype=np.int32))

        else:
            out = [
                np.array([i], dtype=np.int32)
                for i in range(self.num_obs)
            ]

        return out

    def find_groups(self):
        """
        Build groups of observables.

        Non-overlapping case:
          - QWC: sorted-insertion-style partition, where the ordering is
                 controlled by ordering_strategy;
          - FC : generator-based commuting subgroup construction with full-span
                 harvesting, where the ordering is controlled by
                 ordering_strategy.

        Overlapping case with overlap_strategy = "cyclic":
          - QWC: standard OGM / Algorithm-1-style cyclic-start construction;
          - FC : cyclic-start generator-based subgroup construction with
                 full-span harvesting.

        Overlapping case with overlap_strategy = "top_first":
          - QWC: Algorithm-4-style construction without U_k tokens. The seed
                 observable is forced first, then the scan restarts from the
                 top of the global ranking;
          - FC : forced-seed generator construction. The seed observable is
                 forced into the Abelian subgroup first, then the generator
                 scan restarts from the top of the global ranking.
        """
        if self.commutativity_type not in ("qwc", "fc"):
            raise ValueError(
                "LDF_Sorted_Insertion_OGM.find_groups: commutativity_type must be "
                "'qwc' or 'fc'."
            )

        if self.overlap_strategy not in ("cyclic", "top_first"):
            raise ValueError(
                "LDF_Sorted_Insertion_OGM.find_groups: overlap_strategy must be "
                "'cyclic' or 'top_first'."
            )

        order = self._compute_grouping_order()

        M = self.num_obs

        def _first_uncovered_pos(covered_mask: np.ndarray) -> int:
            for j, idx in enumerate(order):
                if not covered_mask[idx]:
                    return j
            return -1

        def _cyclic_seq_from_pos(jpos: int) -> np.ndarray:
            if jpos <= 0:
                return order

            return np.concatenate((order[jpos:], order[:jpos]), axis=0)

        groups = []

        # --------------------------------------------------------------
        # Non-overlapping sorted-insertion / LDF partition.
        # --------------------------------------------------------------
        if not self.is_overlapping:
            remaining = np.ones(M, dtype=bool)

            while np.any(remaining):
                if self.commutativity_type == "qwc":
                    setting = np.zeros(self.num_qubits, dtype=np.int8)

                    for idx in order:
                        if not remaining[idx]:
                            continue

                        o = self.obs[idx]

                        if hit_by_numba(o, setting):
                            non_id = o != 0
                            setting[non_id] = o[non_id]

                            if np.min(setting) > 0:
                                break

                    rem_ids = np.flatnonzero(remaining)

                    is_hit_sub = sample_obs_batch_from_setting_numba(
                        self.obs[rem_ids],
                        setting,
                    )
                    group_idx = rem_ids[np.flatnonzero(is_hit_sub)].astype(np.int32)

                else:
                    group_idx, gen_indices = self._build_fc_group_from_order(
                        order,
                        active_mask=remaining,
                        restrict_to_active=True,
                    )

                    self.last_generator_indices = gen_indices

                if group_idx.size == 0:
                    idx0 = None

                    for idx in order:
                        if remaining[idx]:
                            idx0 = int(idx)
                            break

                    group_idx = np.array([idx0], dtype=np.int32)

                groups.append(group_idx)
                remaining[group_idx] = False

        # --------------------------------------------------------------
        # Overlapping OGM-style covering.
        #
        # In this class, overlapping grouping uses coefficient ordering only.
        # This is enforced in __init__.
        # --------------------------------------------------------------
        else:
            covered = np.zeros(M, dtype=bool)

            while not np.all(covered):
                jpos = _first_uncovered_pos(covered)

                if jpos < 0:
                    break

                seed_idx = int(order[jpos])

                # ------------------------------------------------------
                # Standard OGM / Algorithm-1-style cyclic scan.
                # ------------------------------------------------------
                if self.overlap_strategy == "cyclic":
                    seq = _cyclic_seq_from_pos(jpos)

                    if self.commutativity_type == "qwc":
                        setting = np.zeros(self.num_qubits, dtype=np.int8)

                        for idx in seq:
                            o = self.obs[idx]

                            if hit_by_numba(o, setting):
                                non_id = o != 0
                                setting[non_id] = o[non_id]

                                if np.min(setting) > 0:
                                    break

                        is_hit = sample_obs_batch_from_setting_numba(
                            self.obs,
                            setting,
                        )
                        group_idx = np.flatnonzero(is_hit).astype(np.int32)

                    else:
                        group_idx, gen_indices = self._build_fc_group_from_order(
                            seq,
                            active_mask=None,
                            restrict_to_active=False,
                        )

                        self.last_generator_indices = gen_indices

                # ------------------------------------------------------
                # Algorithm-4-style top-first scan, without U_k tokens.
                # ------------------------------------------------------
                elif self.overlap_strategy == "top_first":
                    if self.commutativity_type == "qwc":
                        setting = np.zeros(self.num_qubits, dtype=np.int8)

                        # Force the seed first.
                        o_seed = self.obs[seed_idx]
                        non_id_seed = o_seed != 0
                        setting[non_id_seed] = o_seed[non_id_seed]

                        # Then scan the full ranking from the top.
                        for idx in order:
                            idxi = int(idx)

                            if idxi == seed_idx:
                                continue

                            o = self.obs[idxi]

                            if hit_by_numba(o, setting):
                                non_id = o != 0
                                setting[non_id] = o[non_id]

                                if np.min(setting) > 0:
                                    break

                        is_hit = sample_obs_batch_from_setting_numba(
                            self.obs,
                            setting,
                        )
                        group_idx = np.flatnonzero(is_hit).astype(np.int32)

                    else:
                        group_idx, gen_indices = self._build_fc_group_from_forced_seed(
                            seed_idx=seed_idx,
                            scan_order=order,
                            active_mask=None,
                            restrict_to_active=False,
                        )

                        self.last_generator_indices = gen_indices

                else:
                    raise ValueError(
                        "Unsupported overlap_strategy. Expected 'cyclic' or "
                        f"'top_first', got {self.overlap_strategy!r}."
                    )

                # Safety fallback. In normal operation, both cyclic and
                # top_first branches should cover seed_idx.
                if group_idx.size == 0:
                    group_idx = np.array([seed_idx], dtype=np.int32)

                groups.append(group_idx)
                covered[group_idx] = True

        groups = self._postprocess_groups(groups)

        self.cliques_pool = groups

        return groups


    def find_setting(self, forced_idx=None):
        """
        Return one observable group in the same format as
        Shadow_Grouping_FC.find_setting():

            setting_indices, info

        Groups are selected according to the allocated measurement counts when
        available. Otherwise, they are returned cyclically.
        """
        if not self.cliques_pool:
            self.find_groups()

        if not self.cliques_pool:
            return np.empty(0, dtype=np.int32), {}

        # Optional forced observable: return a group containing it.
        if forced_idx is not None:
            forced_idx = int(forced_idx)

            if forced_idx < 0 or forced_idx >= self.num_obs:
                raise IndexError(
                    f"forced_idx must be in [0, {self.num_obs}), "
                    f"got {forced_idx}."
                )

            for group_idx, group in enumerate(self.cliques_pool):
                if np.any(group == forced_idx):
                    setting_indices = np.asarray(
                        group,
                        dtype=np.int32,
                    ).copy()
                    setting_indices.sort()
                    selected_mask = np.zeros(self.num_obs, dtype=bool)
                    selected_mask[setting_indices] = True
                    self.N_hits += selected_mask.astype(np.int64)
                    return setting_indices, {
                        "group_index": group_idx,
                    }

            raise ValueError(
                f"No group contains observable index {forced_idx}."
            )

        # Initialize cursor lazily.
        if not hasattr(self, "_setting_cursor"):
            self._setting_cursor = 0

        group_index = self._setting_cursor % len(self.cliques_pool)
        self._setting_cursor += 1

        setting_indices = np.asarray(
            self.cliques_pool[group_index],
            dtype=np.int32,
        ).copy()
        setting_indices.sort()

        info = {
            "group_index": group_index,
        }
        selected_mask = np.zeros(self.num_obs, dtype=bool)
        selected_mask[setting_indices] = True
        self.N_hits += selected_mask.astype(np.int64)
        return setting_indices, info


class Shadow_Grouping_FC(Measurement_scheme):
    """
    ShadowGrouping variant that supports:
      - commutativity_type='qwc': QWC product-basis settings
      - commutativity_type='fc' : FC settings built from an ordered generator basis.
                                 After generators are chosen, we "hit" all observables
                                 in the span of those generators in one go.
    """

    def __init__(self, observables, weights, epsilon, weight_function,
                 save_scheme=False, handle_ties=True, compute_N_hits_pairs=True,
                 commutativity_type="fc"):
        super().__init__(observables, weights, epsilon)
        self.save_scheme = False
        self.weight_function = weight_function
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w)

        self.is_sampling = False
        self.handle_ties = handle_ties
        self.compute_N_hits_pairs = compute_N_hits_pairs

        if commutativity_type not in ("fc", "qwc"):
            raise ValueError("commutativity_type must be 'fc' or 'qwc'.")
        self.commutativity_type = commutativity_type

        self._build_symplectic_cache()
        self.last_generator_indices = np.empty(0, dtype=np.int32)
        self._fc_row_cache = {}

    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.N_hits_pairs = np.zeros_like(self.N_hits_pairs)
        self.seen_settings = set()
        self._is_hit_rows_used = 0
        self._hit_outer_cache = {}
        self.settings_dict = {}
        self.settings_buffer = {}
        self.last_generator_indices = np.empty(0, dtype=np.int32)

        # Clear row cache on reset (safe default)
        self._fc_row_cache = {}

        if self.save_scheme:
            self.all_settings_list = []
            self.num_diff_settings_list = []
            self.diff_settings_counter = 0

    def _build_symplectic_cache(self) -> None:
        """
        Build:
          self._x_u64, self._z_u64, self._packed_u64 (NumPy uint64)
          self._x, self._z, self._packed (Python ints, if needed elsewhere)
        """
        n = self.num_qubits
        bitweights = (1 << np.arange(n, dtype=np.uint64))

        x_bits = ((self.obs == 1) | (self.obs == 2)).astype(np.uint64)
        z_bits = ((self.obs == 2) | (self.obs == 3)).astype(np.uint64)

        x_u64 = x_bits @ bitweights
        z_u64 = z_bits @ bitweights

        packed_u64 = x_u64 | (z_u64 << np.uint64(n))

        self._x_u64 = x_u64.astype(np.uint64, copy=False)
        self._z_u64 = z_u64.astype(np.uint64, copy=False)
        self._packed_u64 = packed_u64.astype(np.uint64, copy=False)

        # Python-int mirrors (sometimes handy; not required for caching path)
        self._x = [int(v) for v in self._x_u64.tolist()]
        self._z = [int(v) for v in self._z_u64.tolist()]
        self._packed = [int(v) for v in self._packed_u64.tolist()]

    def _get_fc_compat_row(self, idx: int) -> np.ndarray:
        """
        Lazy row retrieval: returns a bool array row[j]=commute(idx,j).
        Caches only rows for indices that become generators.
        """
        row = self._fc_row_cache.get(idx)
        if row is not None:
            return row

        xi = self._x_u64[idx]
        zi = self._z_u64[idx]

        row = fc_compat_row_numba(self._x_u64, self._z_u64, xi, zi)

        # store and return
        self._fc_row_cache[idx] = row
        return row

    def _register_setting(self, setting_indices, selected_mask=None,
                          setting_token=None):
        """
        Account for one executed measurement setting.

        Both queued LDF settings and settings constructed from the ordinary
        ShadowGrouping ranking pass through this method.
        """
        setting_indices = np.asarray(
            setting_indices,
            dtype=np.int32,
        ).ravel()
        setting_indices = np.unique(setting_indices)
        setting_indices.sort()

        if setting_indices.size == 0:
            raise RuntimeError("Cannot register an empty measurement setting.")

        if (
            np.any(setting_indices < 0)
            or np.any(setting_indices >= self.num_obs)
        ):
            raise IndexError(
                "A measurement setting contains an observable index outside "
                f"[0, {self.num_obs})."
            )

        if selected_mask is None:
            selected_mask = np.zeros(self.num_obs, dtype=bool)
            selected_mask[setting_indices] = True
        else:
            selected_mask = np.asarray(
                selected_mask,
                dtype=bool,
            ).reshape(-1)
            if selected_mask.shape != (self.num_obs,):
                raise ValueError(
                    "selected_mask must have shape "
                    f"({self.num_obs},), got {selected_mask.shape}."
                )

            if not np.array_equal(
                np.flatnonzero(selected_mask).astype(np.int32),
                setting_indices,
            ):
                raise ValueError(
                    "selected_mask and setting_indices describe different "
                    "hit sets."
                )

        canonical_token = encode_setting_token(setting_indices)
        if setting_token is None:
            setting_token = canonical_token
        elif setting_token != canonical_token:
            raise ValueError(
                "setting_token does not match the canonical observable-index "
                "set."
            )

        self.N_hits += selected_mask.astype(np.int64)

        if self.compute_N_hits_pairs:
            idx = self._append_is_hit_hit_outer(
                setting_token,
                setting_indices,
            )
            if idx.size:
                self.N_hits_pairs[np.ix_(idx, idx)] += 1

        if setting_token not in self.seen_settings:
            self._append_is_hit_row(selected_mask)
            self.seen_settings.add(setting_token)
            if self.save_scheme:
                self.diff_settings_counter += 1
                self.num_diff_settings_list.append(
                    self.diff_settings_counter
                )
                self.all_settings_list.append(
                    list(map(int, setting_indices))
                )
        elif self.save_scheme:
            self.num_diff_settings_list.append(self.diff_settings_counter)
            self.all_settings_list.append(list(map(int, setting_indices)))

        return setting_indices


    def find_setting(self,forced_idx=None):
        weights = self.weight_function(self.w, self.eps, self.N_hits)

        if self.handle_ties:
            M = len(weights)
            rounded = np.round(weights, decimals=12)
            primary = -rounded
            secondary = -np.arange(M, dtype=np.int64)
            order_desc = np.lexsort((secondary, primary))
        else:
            order_desc = np.argsort(weights)[::-1]
            
        # If forced_idx is provided, place it at top the ranking
        order_desc = self._promote_forced_idx(order_desc, forced_idx)

        if self.commutativity_type == "qwc":
            setting = np.zeros(self.num_qubits, dtype=int)
            gen_indices = []

            for idx in order_desc:
                o = self.obs[idx]
                if hit_by_numba(o, setting):
                    non_id = (o != 0)
                    sets_new = non_id & (setting == 0)
                    if np.any(sets_new):
                        gen_indices.append(int(idx))
                    setting[non_id] = o[non_id]
                    if np.min(setting) > 0:
                        break

            selected_mask = hit_by_batch_numba(self.obs, setting).astype(bool)
            self.last_generator_indices = np.asarray(gen_indices, dtype=np.int32)

        else:
            n = self.num_qubits
            basis = _GF2LinearBasis(max_bits=2 * n)

            gen_indices = []

            compat_mask = np.ones(self.num_obs, dtype=np.bool_)

            # Greedily add independent generators (<= n)
            for idx in order_desc:
                idxi = int(idx)

                # Prune quickly using accumulated compat_mask
                if not compat_mask[idxi]:
                    continue

                # Independence test
                v = self._packed[idxi]
                if basis.add(v):
                    gen_indices.append(idxi)

                    # Update compat_mask using cached (or newly computed) row
                    row = self._get_fc_compat_row(idxi)
                    compat_mask &= row

                    if basis.rank >= n:
                        break

            self.last_generator_indices = np.asarray(gen_indices, dtype=np.int32)

            basis_rows_u64, pivot_bits_u8 = _export_basis_compact(basis)
            selected_mask = in_span_batch_numba(self._packed_u64, basis_rows_u64, pivot_bits_u8)
            selected_mask = selected_mask.astype(bool, copy=False)

        setting_indices = np.nonzero(selected_mask)[0].astype(np.int32)
        setting_indices.sort()
        token = encode_setting_token(setting_indices)

        self.N_hits += selected_mask.astype(np.int64)

        if self.compute_N_hits_pairs:
            idx = self._append_is_hit_hit_outer(token, setting_indices)
            if idx.size:
                self.N_hits_pairs[np.ix_(idx, idx)] += 1

        info = {}

        if token not in self.seen_settings:
            self._append_is_hit_row(selected_mask)
            self.seen_settings.add(token)
            if self.save_scheme:
                self.diff_settings_counter += 1
                self.num_diff_settings_list.append(self.diff_settings_counter)
                self.all_settings_list.append(list(map(int, setting_indices)))
        else:
            if self.save_scheme:
                self.num_diff_settings_list.append(self.diff_settings_counter)
                self.all_settings_list.append(list(map(int, setting_indices)))

        return setting_indices , info
