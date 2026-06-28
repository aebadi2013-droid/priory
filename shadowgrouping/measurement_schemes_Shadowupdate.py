from multiprocessing.util import info

import numpy as np
import networkx as nx
from itertools import product
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


def sample_obs_from_setting(O,P):
    for o, p in zip(O, P):
        if o != 0 and o != p:
            return False
    return True

def hit_by_numba(O, P):
    """
    Numba-accelerated version of hit_by for a single observable and setting.
    """
    n = len(O)
    for i in range(n):
        o = O[i]
        p = P[i]
        if not (o == 0 or p == 0 or o == p):
            return False
    return True

def hit_by_batch_numba(O_batch, P):
    n_obs, n_qubits = O_batch.shape
    result = np.empty(n_obs, dtype=np.bool_)
    
    for i in range(n_obs):
        compatible = True
        for j in range(n_qubits):
            o = O_batch[i, j]
            p = P[j]
            if not (o == 0 or p == 0 or o == p):
                compatible = False
                break
        result[i] = compatible
    
    return result

def setting_to_str(arr):
    out = ""
    for a in np.array(arr).flatten():
        out += str(a)
    return out

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


# Helper function to find cliques based on the dominating set
def find_cliques0(graph, center_node, max_depth=10):
    """
    Recursively finds cliques in the neighborhood of the given center node.
    Returns a list of cliques (each a list of node indices).
    """

    removed_nodes = set()
    cliques = []
    processed_nodes = set()

    def DS(B, processed_nodes):
        neighbournum = {node: len(list(B.neighbors(node))) for node in B.nodes}
        nsorted_indices = sorted(neighbournum.keys(), key=lambda x: neighbournum[x], reverse=True)
        ndominating_set = set()
        ncovered_nodes = set()
        for node in nsorted_indices:
            if node not in processed_nodes and node not in ncovered_nodes:
                ndominating_set.add(node)
                ncovered_nodes.add(node)
                ncovered_nodes.update(B.neighbors(node))
        return ndominating_set

    def recursive_clique_detection(B, cliques, processed_nodes, depth):
        nonlocal removed_nodes
        if depth > max_depth or len(B.nodes) == 0:
            return

        dominating_set = DS(B, processed_nodes)
        for v in dominating_set:
            neighbors = list(B.neighbors(v))
            subgraph_nodes = neighbors + [v]
            subgraph = B.subgraph(subgraph_nodes).copy()
            n = len(subgraph.nodes)
            expected_edges = (n * (n - 1)) // 2
            actual_edges = subgraph.number_of_edges()

            #print(f"Depth: {depth}, Node: {v}")
            #print(f"Subgraph nodes: {subgraph.nodes}")
            #print(f"Expected edges: {expected_edges}, Actual edges: {actual_edges}")

            if actual_edges == expected_edges:
                clique = list(subgraph.nodes)
                if removed_nodes:
                    clique += list(removed_nodes)
                    removed_nodes.clear()
                cliques.append(clique)
                processed_nodes.update(subgraph.nodes)
            else:
                processed_nodes.add(v)
                removed_nodes.add(v)
                if v in subgraph_nodes:
                    subgraph_nodes.remove(v)
                new_subgraph = subgraph.subgraph(subgraph_nodes).copy()
                recursive_clique_detection(new_subgraph, cliques, processed_nodes, depth + 1)

    # Build subgraph around center_node to limit search scope
    if center_node not in graph:
        return []
    center_neighbors = list(graph.neighbors(center_node))
    neighborhood_nodes = center_neighbors + [center_node]
    subgraph = graph.subgraph(neighborhood_nodes).copy()

    #print("Subgraph nodes:", subgraph.nodes)
    #print("Subgraph edges:", subgraph.edges)

    recursive_clique_detection(subgraph, cliques, processed_nodes, 0)
    return cliques

def find_cliques2(graph, max_depth=10):
    """
    Recursively finds cliques in the entire graph (no center node).
    Returns a list of cliques (each a list of node indices).
    """
    removed_nodes = set()
    cliques = []
    processed_nodes = set()

    def DS(B, processed_nodes):
        neighbournum = {node: len(list(B.neighbors(node))) for node in B.nodes}
        nsorted_indices = sorted(neighbournum.keys(), key=lambda x: neighbournum[x], reverse=True)
        ndominating_set = set()
        ncovered_nodes = set()
        for node in nsorted_indices:
            if node not in processed_nodes and node not in ncovered_nodes:
                ndominating_set.add(node)
                ncovered_nodes.add(node)
                ncovered_nodes.update(B.neighbors(node))
        return ndominating_set

    def recursive_clique_detection(B, cliques, processed_nodes, depth):
        nonlocal removed_nodes
        if depth > max_depth or len(B.nodes) == 0:
            return

        dominating_set = DS(B, processed_nodes)
        for v in dominating_set:
            neighbors = list(B.neighbors(v))
            subgraph_nodes = neighbors + [v]
            subgraph = B.subgraph(subgraph_nodes).copy()
            n = len(subgraph.nodes)
            expected_edges = (n * (n - 1)) // 2
            actual_edges = subgraph.number_of_edges()

            if actual_edges == expected_edges:
                clique = list(subgraph.nodes)
                if removed_nodes:
                    clique += list(removed_nodes)
                    removed_nodes.clear()
                cliques.append(clique)
                processed_nodes.update(subgraph.nodes)
            else:
                processed_nodes.add(v)
                removed_nodes.add(v)
                if v in subgraph_nodes:
                    subgraph_nodes.remove(v)
                new_subgraph = subgraph.subgraph(subgraph_nodes).copy()
                recursive_clique_detection(new_subgraph, cliques, processed_nodes, depth + 1)

    recursive_clique_detection(graph.copy(), cliques, processed_nodes, 0)
    return cliques


#find cliques without removing nodes
def find_cliques3(graph, center_node=None, max_depth=20, min_size=1):
    """
    Recursively finds cliques in the neighborhood of the given center node.
    Returns a list of cliques (each a list of node indices), allowing overlapping cliques.
    """
    cliques = []

    def DS(B):
        """Simple dominating set heuristic based on degree."""
        neighbournum = {node: len(list(B.neighbors(node))) for node in B.nodes}
        nsorted_indices = sorted(neighbournum.keys(), key=lambda x: neighbournum[x], reverse=True)
        ndominating_set = set()
        ncovered_nodes = set()
        for node in nsorted_indices:
            if node not in ncovered_nodes:
                ndominating_set.add(node)
                ncovered_nodes.add(node)
                ncovered_nodes.update(B.neighbors(node))
        return ndominating_set

    def recursive_clique_detection(B, depth):
        if depth > max_depth or len(B.nodes) == 0:
            return
        dominating_set = DS(B)
        for v in dominating_set:
            neighbors = list(B.neighbors(v))
            subgraph_nodes = neighbors + [v]
            subgraph = B.subgraph(subgraph_nodes).copy()
            n = len(subgraph.nodes)
            expected_edges = (n * (n - 1)) // 2
            actual_edges = subgraph.number_of_edges()

            if actual_edges == expected_edges:
                clique = list(subgraph.nodes)
                if len(clique) >= min_size:
                    cliques.append(clique)
            else:
                if v in subgraph_nodes:
                    subgraph_nodes.remove(v)
                new_subgraph = subgraph.subgraph(subgraph_nodes).copy()
                recursive_clique_detection(new_subgraph, depth + 1)

    # Limit scope to center_node's neighborhood, or use full graph
    if center_node is not None:
        if center_node not in graph:
            return []
        neighborhood_nodes = list(graph.neighbors(center_node)) + [center_node]
        subgraph = graph.subgraph(neighborhood_nodes).copy()
    else:
        subgraph = graph.copy()

    recursive_clique_detection(subgraph, 0)
    return cliques

import networkx as nx

def find_cliques4(graph, max_depth=10, weight_map=None):
    """
    Recursively finds cliques in the entire graph using a dominating set strategy.
    Returns a list of cliques (each a list of node indices).

    Args:
        graph (networkx.Graph): The graph to search for cliques.
        max_depth (int): The maximum recursion depth.
        weight_map (dict, optional): A dictionary mapping nodes to their weights.
                                      If provided, the nodes will be sorted by weights instead of degrees.
    """
    removed_nodes = []

    def DS(B, processed_nodes):
        if len(B.nodes) == 0:
            return {}
        
        # Sort nodes by weight, descending (instead of by degree)
        if weight_map:
            nsorted_indices = sorted(B.nodes, key=lambda x: weight_map.get(x, 0), reverse=True)
        else:
            # Fallback to degree-based sorting if no weight_map is provided
            neighbournum = {node: len(list(B.neighbors(node))) for node in B.nodes}
            nsorted_indices = sorted(neighbournum.keys(), key=lambda x: neighbournum[x], reverse=True)

        ndominating_set = set()
        ncovered_nodes = set()

        for node in nsorted_indices:
            if node not in processed_nodes and node not in ncovered_nodes:
                ndominating_set.add(node)
                ncovered_nodes.add(node)
                ncovered_nodes.update(B.neighbors(node))
        return ndominating_set

    def recursive_clique_detection(B, cliques, processed_nodes=set(), depth=0):
        nonlocal removed_nodes
        if depth > max_depth or len(B.nodes) == 0:
            return

        dominating_set = DS(B, processed_nodes)

        for v in dominating_set:
            neighbors = list(B.neighbors(v))
            subgraph_nodes = neighbors + [v]
            subgraph = B.subgraph(subgraph_nodes).copy()

            n_nodes = len(subgraph.nodes)
            expected_edges = (n_nodes * (n_nodes - 1)) // 2
            actual_edges = subgraph.number_of_edges()
            removed_nodes[:] = removed_nodes[:depth]

            if actual_edges == expected_edges:
                clique = list(subgraph.nodes)
                clique.extend(removed_nodes)
                cliques.append(clique)
                processed_nodes.add(v)
            else:
                removed_nodes.append(v)
                subgraph_nodes.remove(v)
                new_subgraph = subgraph.subgraph(subgraph_nodes).copy()
                isolated_nodes = [n for n in new_subgraph.nodes() if new_subgraph.degree(n) == 0]

                if isolated_nodes:
                    for n in isolated_nodes:
                        clique = [n] + removed_nodes
                        cliques.append(clique)
                        processed_nodes.add(n)

                if all(n in processed_nodes for n in new_subgraph.nodes()):
                    if removed_nodes:
                        cliques.append(list(removed_nodes))

                recursive_clique_detection(new_subgraph, cliques, processed_nodes, depth + 1)

    cliques = []
    recursive_clique_detection(graph.copy(), cliques)
    return cliques


def find_cliques5(graph, max_depth=10):

    def DS(B, processed_nodes):
        """
        Finds a dominating set using the largest degree first method,
        excluding processed nodes from the dominating set.
        """
        if len(B.nodes) == 0:  # Base case: Empty graph
            return {}
        # Check if all nodes are processed
        
        # Compute degrees and sort nodes by degree in descending order
        neighbournum = {node: len(list(B.neighbors(node))) for node in B.nodes}
        nsorted_indices = sorted(neighbournum.keys(), key=lambda x: neighbournum[x], reverse=True)
        
        ndominating_set = set()
        ncovered_nodes = set()
        
        for node in nsorted_indices:
            if node not in processed_nodes and node not in ncovered_nodes:
                ndominating_set.add(node)
                ncovered_nodes.add(node)
                ncovered_nodes.update(B.neighbors(node))
    
        return ndominating_set
    
    def recursive_clique_detection(B, cliques, processed_nodes, removed_nodes, depth=0, max_depth=20):
        """
        Recursively finds cliques in the graph B.
        If a subgraph is not a clique, recursively check its neighbors.
        Stores found cliques in `cliques` list.
        """
        if processed_nodes is None:
            processed_nodes = set()
        if removed_nodes is None:
            removed_nodes = []

        
        #if depth > max_depth:
            #print(f"maximum depth reached")
        
        if depth > max_depth or len(B.nodes) == 0:
            return  # Stop recursion if depth exceeds max_depth or graph is empty
    
        # Step 1: Find the dominating set, excluding processed nodes
        dominating_set = DS(B, processed_nodes)
    
        
        for v in dominating_set:
                   
            # Step 2: For each node v in the dominating set, take the subgraph of neighbors around it
            neighbors = list(B.neighbors(v))
            subgraph_nodes = neighbors + [v]
            subgraph = B.subgraph(subgraph_nodes).copy()  # Create subgraph around v
            
            n_nodes = len(subgraph.nodes)  # Number of nodes in the subgraph
            expected_edges = (n_nodes * (n_nodes - 1)) // 2  # Expected number of edges for a clique
            actual_edges = subgraph.number_of_edges()
            removed_nodes[:] = removed_nodes[:depth]  # Truncate to current depth
            #print(f"depth of search : {depth}")
            #print(f"Checking node {v}:")
            #print(f"Subgraph nodes: {subgraph.nodes}")
            #print(f"Expected edges: {expected_edges}, Actual edges: {actual_edges}")
            
            # Check if the subgraph is large enough and contains the correct number of edges to form a clique
            if actual_edges == expected_edges:
                # Step 4: If it's a clique, store it
                #print(f"Clique found: {list(subgraph.nodes)}")
                cliques.append(list(subgraph.nodes))
                processed_nodes.add(v)  # Mark all nodes in the clique as processed
                # Add the last removed node to this newly found clique (if any exist)
                cliques[-1].extend(removed_nodes)  # Add it to the last found clique
                #print(f"Added last removed node {removed_nodes} to the clique {cliques[-1]}")
    
            
            else:
    
                # Add the center node to the last clique found (if any)
                #if cliques:
                #   cliques[-1].append(v)  # Add v to the last clique
                # Step 5: If not a clique, recursively process the subgraph without the center node
                #processed_nodes.add(v)  # Add the center node v to the processed nodes
                removed_nodes.append(v)
                subgraph_nodes.remove(v)  # Remove the center node from the subgraph nodes
                #print(f"node number {v} is removed")
                # Create new subgraph excluding the center node v
                new_subgraph = subgraph.subgraph(subgraph_nodes).copy()
                # If the new subgraph has no edges but has nodes, consider those as singleton cliques
                # Check for isolated nodes in the new subgraph (nodes with no edges)
                isolated_nodes = [n for n in new_subgraph.nodes() if new_subgraph.degree(n) == 0]
                
                if isolated_nodes:
                    for n in isolated_nodes:
                        #print(f"Node {n} is isolated — treated as its own clique.")
                        cliques.append([n])  # Add isolated node as a clique
                        processed_nodes.add(n)  # Mark the isolated node as processed
                        # Add the last removed node to this newly found clique (if any exist)
                        cliques[-1].extend(removed_nodes)  # Add it to the last found clique
                        #print(f"Added last removed node {removed_nodes} to the clique {cliques[-1]}")
    
                if all(n in processed_nodes for n in new_subgraph.nodes()):
                    #print(" All nodes in the new subgraph are already processed. Building a clique from removed_nodes.")
                    if removed_nodes:
                        cliques.append(list(removed_nodes))  # Use removed_nodes as a fallback clique
                        #print(f"Formed clique from removed_nodes: {cliques[-1]}")
                
                recursive_clique_detection(new_subgraph, cliques, processed_nodes, removed_nodes, depth + 1, max_depth)
    removed_nodes = []
    cliques = []  # Store detected cliques
    recursive_clique_detection(graph, cliques, set(), [], max_depth=20)
    return cliques


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


def greedy_clique_cover(G):
    """
    Approximate clique cover using a greedy peeling heuristic.
    Iteratively finds a large clique and removes it until all nodes are covered.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    
    Returns
    -------
    list[list]
        List of cliques (each clique is a list of nodes).
    """
    
    def greedy_clique(H):
        """Find a large clique in H using a greedy heuristic."""
        nodes = sorted(H.nodes(), key=lambda x: H.degree(x), reverse=True)
        clique = []
        for node in nodes:
            if all(H.has_edge(node, neighbor) for neighbor in clique):
                clique.append(node)
        return clique

    H = G.copy()
    cliques = []
    while H.number_of_nodes() > 0:
        clique = greedy_clique(H)
        cliques.append(clique)
        H.remove_nodes_from(clique)
    return cliques


def approximate_clique_cover(G, strategy="largest_first"):
    """
    Approximate clique cover using greedy coloring
    of the complement graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    strategy : str
        Greedy coloring strategy for networkx.coloring.greedy_color.
        Options: "largest_first", "random_sequential", 
                 "smallest_last", "DSATUR", etc.

    Returns
    -------
    list[list]
        List of cliques (each clique is a list of nodes).
    """
    Gc = nx.complement(G)
    coloring = nx.coloring.greedy_color(Gc, strategy=strategy)
    cover = {}
    for node, color in coloring.items():
        cover.setdefault(color, []).append(node)
    return list(cover.values())


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
        self._is_hit_cap       = 16  # initial capacity
        self._is_hit_buf       = np.empty((self._is_hit_cap, self.num_obs), dtype=bool)
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
    
    def get_epsilon_Bernstein(self,delta):
        """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
            If at least one of the N_hits is 0, epsilon is set equal to infinity.
            Else, epsilon = 2*|weights/sqrt(N_hits)| * (1 + 2sqrt(log(1/delta)))
        """
        if np.min(self.N_hits) == 0:
            return np.inf
        w_abs  = np.abs(self.w)
        w_abs /= np.sqrt(self.N_hits)
        norm   = np.sum(w_abs)
        w_abs /= np.sqrt(self.N_hits)
        norm2  = np.sum(w_abs)
        epsilon = norm * np.sqrt(N_delta(delta)) #equation 29 of the supplementary
        if epsilon > 2*norm*(1+2*norm/norm2):
            #print("Warning! Epsilon out of validity range.")
            pass
        return epsilon

    def get_epsilon_Bernstein_no_restricted_validity(self,delta):
        """ Return the epsilon such that the corresponding Bernstein bound is not larger than delta.
            If at least one of the N_hits is 0, epsilon is set equal to infinity.
            Else, epsilon = sigma * [1 + sqrt(2 log(1/delta)) ] + 2B/3 * log(1/delta)
        """
        if np.min(self.N_hits) == 0:
            return np.inf
        w_abs  = np.abs(self.w)
        w_abs /= np.sqrt(self.N_hits)
        sigma  = 2 * np.sum(w_abs) # Eq. (25), Supp. Inf. of published version of ShadowGrouping paper

        w_abs /= np.sqrt(self.N_hits)
        B = 2 * np.sum(w_abs) # Eq. (23), Supp. Inf. of published version of ShadowGrouping paper
        epsilon = sigma * ( 1 + np.sqrt(-2*np.log(delta)) ) - 2*B*np.log(delta)/3
        return epsilon
    def get_epsilon_Bernstein_tighter_no_restricted_validity(self, delta, N_hits, w, settings_dict, obs, split=False):
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
            raise ValueError("delta must be in (0,1)")
        # systematic error: observables never measured
        eps_sys = np.sum(np.abs(w[N_hits == 0]))
        if np.any(N_hits > 0):
            settings_list = list(settings_dict.keys())
            settings_reps = np.array(list(settings_dict.values()), dtype=np.int64)
            # convert settings into array form
            settings_obs = np.array([setting_to_obs_form(s) for s in settings_list], dtype=np.int8)
            # build compatibility matrix (n_settings, n_obs)
            compat = sample_obs_batch_from_setting_batch_numba(obs, settings_obs)  # shape (n_obs, n_settings)
            compat = compat.T  # shape (n_settings, n_obs)
            # effective weights |h_i| / N_i
            w_eff = np.zeros_like(w, dtype=np.float64)
            mask = N_hits > 0
            w_eff[mask] = np.abs(w[mask]) / N_hits[mask]
            # per-setting weights
            settings_weights = compat @ w_eff   # (n_settings,)
            # sigma = 2 * sqrt( sum_k reps[k] * settings_weights[k]^2 )
            sigma2 = np.dot(settings_reps, settings_weights**2)
            sigma = 2.0 * np.sqrt(sigma2)
            # B = 4 * max_k settings_weights[k]
            B = 4.0 * np.max(settings_weights)
            # Bernstein formula (Eq. 23 + Eq. 25)
            log_term = -np.log(delta)
            eps_stat = sigma * (1.0 + np.sqrt(2.0 * log_term)) + (2.0 / 3.0) * B * log_term
        else:
            eps_stat = 0.0
        return (eps_stat, eps_sys) if split else (eps_stat + eps_sys)


    def get_epsilon_Bernstein_no_restricted_validity_v2(self,delta,split=False):
        """ Returns the epsilon such that the corresponding Bernstein bound is not larger than delta.
            If at least one of the N_hits is 0, associated systematic error is accounted for.
            Else, epsilon = sigma * [1 + sqrt(2 log(1/delta)) ] + 2B/3 * log(1/delta).
            split = True provides statistical and systematic errors separately, otherwise they are summed.
        """
    
        if not (0 < delta < 1):
            raise ValueError("delta must be in the interval (0,1)")
    
        # systematic error due to observables that have not been measured even once
        eps_sys = np.sum(np.abs(self.w[self.N_hits == 0]))
    
        # statistical error due to observables with at least one sample
        if np.sum(self.N_hits > 0) > 0:
            w_abs  = np.abs(self.w[self.N_hits > 0])
            w_abs /= np.sqrt(self.N_hits[self.N_hits > 0])
            sigma  = 2 * np.sum(w_abs) # Eq. (25), Supp. Inf. of published version of ShadowGrouping paper
    
            w_abs /= np.sqrt(self.N_hits[self.N_hits > 0])
            """B = 4 * np.sum(w_abs) # Eq. (23), Supp. Inf. of published version of ShadowGrouping paper
                                  # and extra factor of 2 from Eq. (14) as well"""
            B = 2 * np.sum(w_abs) # Eq. (23), Supp. Inf. of published version of ShadowGrouping paper
            eps_stat = sigma * ( 1 + np.sqrt(-2*np.log(delta)) ) - 2*B*np.log(delta)/3
        else:
            eps_stat = 0.0

        if split:
            return eps_stat, eps_sys
        else:
            return eps_stat + eps_sys

    def get_epsilon_Bernstein_no_restricted_validity_v3(self, delta):
        """Return epsilon such that the corresponding Bernstein bound is not larger than delta.
           Terms with N_hits == 0 are ignored.
           epsilon = sigma * [1 + sqrt(2 log(1/delta)) ] + 2B/3 * log(1/delta)
        """
        # Mask out zero entries
        mask = self.N_hits > 0
    
        if not np.any(mask):  # all N_hits are zero
            return np.inf
    
        w_abs = np.abs(self.w[mask]) / np.sqrt(self.N_hits[mask])
        sigma = 2 * np.sum(w_abs)  # Eq. (25)
    
        w_abs = np.abs(self.w[mask]) / self.N_hits[mask]
        B = 2 * np.sum(w_abs)  # Eq. (23)
    
        epsilon = sigma * (1 + np.sqrt(-2 * np.log(delta))) - (2 * B * np.log(delta) / 3)
        return epsilon

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

    def _append_is_hit_row(self, is_hit_row: np.ndarray) -> None:
        """Append one new unique is_hit row (bool shape (M,))."""
        self._ensure_is_hit_capacity(self._is_hit_rows_used + 1)
        self._is_hit_buf[self._is_hit_rows_used] = is_hit_row
        self._is_hit_rows_used += 1

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

class Priori(Measurement_scheme):
    """ First find all cliques using DomClique then select the heaviest in each round
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
        self.eps_values_v3 = []
        self.eps_chebyshev_tighter = []
        self.eps_chebyshev_tightest = []
        self.provablegaurantee = []
        self.inconfindence = []
        self.shadow_was_best_count = 0
        self.selected_cliques = []  # Stores best cliques from each round
        self.cliques_with_epsilon = []
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

    def total_hit_weight(self, weights, is_hit):
        weights = np.asarray(weights, dtype=float)
        is_hit = np.asarray(is_hit, dtype=bool)
        return (weights * is_hit).sum()
    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        if self._cached_graph is None or self._cached_cliques is None:
            self.build_graph_and_cliques()
        
        weights = self.weight_function(self.w, self.eps, self.N_hits)
        tstart = time()
        order = np.argsort(weights)
        completecliques = 0
        #self.cliques_with_epsilon = []
        delta = 0.33
        #alpha = 51733.57
        incompletesetting = 0
        if np.any(self.N_hits == 0):
            """settinglist = []
            if self.cliques_with_epsilon:
                settingslist = [s for _, s in self.cliques_with_epsilon]"""
            self.cliques_with_epsilon = []
            shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
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
            #for setting_candidate in self.clean_setting_cache[center_node]:
            #for cached_settings in self._cached_settings.values(): #ggggggg Start
            for setting_candidate in self._cached_settings:
                working = setting_candidate.copy()
                #print(working)
                """if np.min(setting_candidate) == 0:
                    if settinglist:
                        for o in settingslist:
                            if verbose:
                                print("Checking",o)
                            if hit_by(o,setting_candidate):
                                non_id = o!=0
                                setting_candidate[non_id] = o[non_id]
                            if verbose:
                                print("p =",setting_candidate)
                            if np.min(setting_candidate) > 0:
                                print("completed setting using setting list")
                                break"""
                if np.min(working) == 0:
                    #print(working)
                    incompletesetting += 1
                    for idx in reversed(order):
                        o = self.obs[idx]
                        if verbose:
                            print("Checking",o)
                        if hit_by(o,working):
                            non_id = o!=0
                            # overwrite those qubits that fall in the support of o
                            working[non_id] = o[non_id]
                        if verbose:
                            print("p =",working)
                        # break sequence is case all identities in setting are exhausted
                        if np.min(working) > 0:
                            break
                    completecliques += 1
                else:
                    completecliques += 1
                is_hit_candidate = []
                is_hit_candidate = hit_by_batch_numba(self.obs , working)
                self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), working))
                #self.N_hits += is_hit_candidate
                #self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), setting_candidate))
                #self.N_hits -= is_hit_candidate #ggggg end
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), shadowcliquesetting))
            #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), setting_candidate))
            #self.N_hits += is_hit_candidate
            #self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), shadowcliquesetting))
            #self.cliques_with_epsilon.append((self.get_Bernstein_bound(), shadowcliquesetting))
            #print("epsilon for shadow clique is",self.get_epsilon_Bernstein_no_restricted_validity(delta))
            #self.N_hits -= is_hit_candidate
            self.N_hits += is_hit_candidate
            epsilon_shadow = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.N_hits -= is_hit_candidate
            self.cliques_with_epsilon.sort(key=lambda x: x[0], reverse = True)
            if verbose:
                print("length of cliques with epsilon is", len(self.cliques_with_epsilon))
            if verbose:
                print("number of incomplete settings", incompletesetting)
            _, best_clique = self.cliques_with_epsilon[0]
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
            else:
                if verbose:
                    print("epsilon for best clique is",self.cliques_with_epsilon[0])
            setting = best_clique
            if not any(np.array_equal(existing, shadowcliquesetting) for existing in self._cached_settings):
                self._cached_settings.append(shadowcliquesetting.copy())
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , best_clique)
            self.N_hits += is_hit_candidate
            epsilon_best = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.N_hits -= is_hit_candidate
        else:
            #for cached_settings in self._cached_settings.values(): #ggggggg Start
            #self.cliques_with_epsilon = []
            """settinglist = []
            settingslist = [s for _, s in self.cliques_with_epsilon]"""
            self.cliques_with_epsilon = []
            shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
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
            for setting_candidate in self._cached_settings:
                working = setting_candidate.copy()
                #print(working)
                """if np.min(setting_candidate) == 0:
                    for o in settingslist:
                        if verbose:
                            print("Checking",o)
                        if hit_by(o,setting_candidate):
                            non_id = o!=0
                            setting_candidate[non_id] = o[non_id]
                        if verbose:
                            print("p =",setting_candidate)
                        if np.min(setting_candidate) > 0:
                            print("completed setting using setting list")
                            break"""
                if np.min(working) == 0:
                    incompletesetting += 1
                    for idx in reversed(order):
                        o = self.obs[idx]
                        if verbose:
                            print("Checking",o)
                        if hit_by(o,working):
                            non_id = o!=0
                            # overwrite those qubits that fall in the support of o
                            working[non_id] = o[non_id]
                        if verbose:
                            print("p =",working)
                        # break sequence is case all identities in setting are exhausted
                        if np.min(working) > 0:
                            break
                    completecliques += 1
                else:
                    completecliques += 1
                is_hit_candidate = []
                is_hit_candidate = hit_by_batch_numba(self.obs , working)
                #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), setting_candidate))
                self.N_hits += is_hit_candidate
                self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v3(delta), working))
                #self.cliques_with_epsilon.append((get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False), setting_candidate))
                self.N_hits -= is_hit_candidate #ggggg end
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), shadowcliquesetting))
            #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), setting_candidate))
            self.N_hits += is_hit_candidate
            self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v3(delta), shadowcliquesetting))
            epsilon_shadow = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            #self.cliques_with_epsilon.append((get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False), shadowcliquesetting))
            #self.cliques_with_epsilon.append((self.get_Bernstein_bound(), shadowcliquesetting))
            #print("epsilon for shadow clique is",self.get_epsilon_Bernstein_no_restricted_validity(delta))
            self.N_hits -= is_hit_candidate
            self.cliques_with_epsilon.sort(key=lambda x: x[0])
            if verbose:
                print("length of cliques with epsilon is", len(self.cliques_with_epsilon))
            _, best_clique = self.cliques_with_epsilon[0]
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
            else:
                if verbose:
                    print("epsilon for best clique is",self.cliques_with_epsilon[0])
            setting = best_clique
            if not any(np.array_equal(existing, shadowcliquesetting) for existing in self._cached_settings):
                self._cached_settings.append(shadowcliquesetting.copy())
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , best_clique)
            self.N_hits += is_hit_candidate
            epsilon_best = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.N_hits -= is_hit_candidate

        tend = time()
        if verbose:
            print("Shadow clique was selected", self.shadow_was_best_count, "times")
        is_hit = []
        # update number of hits
        is_hit = hit_by_batch_numba(self.obs , setting)
        self.N_hits += is_hit

        
        # Tokenize by the set of compatible observable indices
        setting_indices = np.nonzero(is_hit)[0].astype(np.int32)
        setting_indices.sort()
        token = encode_setting_token(setting_indices)


        if self.compute_N_hits_pairs:
            # Cache (or retrieve) the canonical index list for this token
            idx = self._append_is_hit_hit_outer(token, setting_indices)
            # Apply the outer update without building an outer product
            if idx.size:
                self.N_hits_pairs[np.ix_(idx, idx)] += 1



        delta = 0.33
        self.round_num += 1    
        self.rounds.append(len(self.rounds) + 1)
        if verbose:
            print("round number" , self.round_num)
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["Provable Gaurantee"] = Guaranteed_accuracy(delta, self.N_hits, self.w, split=False)
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
        info["epsilon_Bernstein_no_restricted_validity_v3"] = self.get_epsilon_Bernstein_no_restricted_validity_v3(delta)
        info["epsilon_Bernstein_scalar_no_restricted_validity"] = get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False)
        self.eps_values_v3.append(info["epsilon_Bernstein_no_restricted_validity_v2"])
        self.inconfindence.append(info["inconfidence_bound"])
        info["epsilon difference"]= abs(epsilon_best - epsilon_shadow)
        #print("difference between best epsilon and shadow epsilon = ", info["epsilon difference"])
        #self.eps_values_v3.append(info["epsilon difference"])
        #self.eps_values_v3.append(info["epsilon_Bernstein_no_restricted_validity_v3"])
        #print("epsilon_Bernstein_scalar_no_restricted_validity:", info["epsilon_Bernstein_scalar_no_restricted_validity"])
        if verbose:
            print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta, split=True)
        if verbose:
            print("epsilon_Bernstein_no_restricted_validity_v2:", info["epsilon_Bernstein_no_restricted_validity_v2"])
        info["epsilon_chebyshev_tighter"] = get_epsilon_Chebyshev_scalar_tighter_numba(delta, self.N_hits, self.N_hits_pairs, self.w)
        self.eps_chebyshev_tighter.append(info["epsilon_chebyshev_tighter"])
        info["epsilon_chebyshev_tightest"] = get_epsilon_Chebyshev_scalar_tightest_numba(delta, self.N_hits, self.N_hits_pairs, self.w, self.cov_real)
        self.eps_chebyshev_tightest.append(info["epsilon_chebyshev_tightest"])
        #print("Inconfidence Bound :", info["inconfidence_bound"])
        #print("Provable Gauarantee :", info["Provable Gaurantee"])
        self.provablegaurantee.append(info["Provable Gaurantee"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting_indices, info


class Posteriori(Measurement_scheme):
    """ 
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        #self.settings_dict = {} #2222222222222
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.shadow_was_best_count = 0
        self.round_num = 0
        self.incompletesettingselected = 0
        self.clique_cache = {}  # maps observable tuple -> hit cliques
        self.clean_clique_cache = {}  
        self.clean_setting_cache = {}
        self.is_hit_clique_cache = {}
        self.cliques_with_epsilon = []
        self.processed_center_node = []
        self.rounds = []
        self.eps_values_v3 = []
        self.inconfidence = []
        self.provablegaurantee = []
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.settings_dict = {} #22222222222
        self.settings_buffer = {}
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

    def total_hit_weight(self, weights, is_hit):
        weights = np.asarray(weights, dtype=float)
        is_hit = np.asarray(is_hit, dtype=bool)
        return (weights * is_hit).sum()


    def to_numba_format(self):
        if not self.settings_dict:
        # K = 0 rows, but keep the second dim = num_qubits
            #print("setting_dict is still empty")
            return (np.empty((0, self.num_qubits), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
        #print("setting_dict is not empty")
        return prepare_settings_for_numba(self.settings_dict)


        
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        # sort observable list by respective weight
        #settings_int, settings_reps = self.to_numba_format() 2222222222
        #settings_int, settings_reps = prepare_settings_for_numba(self.settings_dict)
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        #print(f"alpha is = ", alpha)
        order = np.argsort(weights)
        self.selected_cliques = []  # Stores best cliques from each round
        setting = np.zeros(self.num_qubits,dtype=int)
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        if verbose:
            print("Checking list of observables.")
        #print("settings dict is", self.settings_dict)
        #print("settins int is", settings_int, "and setting reps is", settings_reps)
        # Get highest-weight observable
        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
        center_node = tuple(first_obs)  # Use tuple as dictionary key
        print("center node is", first_idx, "and its weight is", weights[first_idx])


        # make local dtyped views for numba WITHOUT touching self.*
        obs_local     = np.asarray(self.obs,    dtype=np.int64)    # shape: (num_obs, num_qubits)
        N_hits_local  = np.asarray(self.N_hits, dtype=np.int64)    # shape: (num_obs,)
        w_local       = np.asarray(self.w,      dtype=np.float64)  # shape: (num_obs,)

        # (optional) bail early if no settings yet
        #if settings_int.size == 0:
            # nothing allocated -> skip calling the function this round
            #pass

        delta = 0.02
        tstart = time()
        
        if np.any(self.N_hits == 0):
            shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
            for idx in reversed(order):
                o = self.obs[idx]
                if verbose:
                    print("Checking",o)
                if hit_by(o,shadowcliquesetting):
                    non_id = o!=0
                    shadowcliquesetting[non_id] = o[non_id]
                if verbose:
                    print("p =",setting)
                if np.min(shadowcliquesetting) > 0:
                    break
            if verbose:
                print("Checking list of observables.")
            #cliques_with_epsilon = []
            delta = 0.02
            removedcliques = 0
            completecliques = 0
            valid_settings = []
            valid_cliques = []
            #start
            if center_node not in self.processed_center_node:  #gggggggg
                self.processed_center_node.append(center_node)
                hit_list = []
                hit_cliques = []
                non_id = first_obs != 0
                globalsetting[non_id] = first_obs[non_id]
                for o in self.obs:
                    if hit_by(o, globalsetting):
                        hit_list.append(o)
                if verbose:
                    print("First observable (setting):", first_obs)
                    print("Other observables hit by it:")
                    for ob in hit_list:
                        print(ob)
                if not hit_list:
                    raise RuntimeError("No hit list found.")
                hit_graph = build_hit_graph(hit_list)
                if not hit_graph:
                    raise RuntimeError("No hit graph found.")
                hit_cliques = find_cliques5(hit_graph)
                #print("length of hit cliques is", len(hit_cliques))
                self.clique_cache[center_node] = hit_cliques
                if not hit_cliques:
                    hit_cliques=[[center_node]]
                self.clean_setting_cache[center_node] = []
                self.clean_clique_cache[center_node] = []
                for clique in hit_cliques:
                    setting_candidate = np.zeros(self.num_qubits, dtype=int)
                    for o in clique:
                        o_arr = np.array(o)
                        if hit_by(o_arr, setting_candidate):
                            non_id = o_arr != 0
                            setting_candidate[non_id] = o_arr[non_id]
                        if np.min(setting_candidate) > 0:
                            break
                    """if np.min(setting_candidate) == 0:
                        for idx in reversed(order):
                            o = self.obs[idx]
                            if verbose:
                                print("Checking",o)
                            if hit_by(o,setting_candidate):
                                non_id = o!=0
                                setting_candidate[non_id] = o[non_id]
                            if verbose:
                                print("p =",setting_candidate)
                            if np.min(setting_candidate) > 0:
                                break"""
                    self.clean_clique_cache[center_node].append(clique)
                    if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache[center_node]):
                        self.clean_setting_cache[center_node].append(setting_candidate)
            else:
                print("I already found cliques for this node")
                """else:
                    completecliques += 1
                    self.clean_clique_cache[center_node].append(clique)
                    if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache[center_node]):
                        self.clean_setting_cache[center_node].append(setting_candidate)"""
            #for cached_settings in self.clean_setting_cache.values(): #ggggggg Start
                #for setting_candidate in cached_settings:
            #print("length of setting cache is", len(self.clean_setting_cache[center_node]))
            #settingslist = []
            #if self.cliques_with_epsilon:
            #    settingslist = [s for _, s in self.cliques_with_epsilon]
            #    print("length of settings list is", len(settingslist))
            self.cliques_with_epsilon = []
            self.incompletesettingcache = []
            #print("length of setting cache", len(self.clean_setting_cache))
            numincomplete = 0
            #numsetting = 0
            #idle = 0
            for setting_candidate in self.clean_setting_cache[center_node]:
            #for cached_settings in self.clean_setting_cache.values(): #ggggggg Start
                #for setting_candidate in cached_settings:
                working = setting_candidate.copy()
                #print("checking candidate", working)
                """if np.min(working) == 0:
                    if settingslist:
                        for o in settingslist:
                            if verbose:
                                print("Checking",o)
                            if hit_by(o,working):
                                non_id = o!=0
                                working[non_id] = o[non_id]
                            if verbose:
                                print("p =",working)
                            if np.min(working) > 0:
                                #print("completed setting using setting list")
                                numsetting += 1
                                break"""
                if np.min(working) == 0:
                    #id = (working == 0)
                    for idx in reversed(order):
                        o = self.obs[idx]
                        if verbose:
                            print("Checking",o)
                        if hit_by(o,working):
                            non_id = o!=0
                            working[non_id] = o[non_id]
                        if verbose:
                            print("p =",working)
                        if np.min(working) > 0:
                            #print("completed using shadow list")
                            self.incompletesettingcache.append(working)
                            numincomplete += 1
                            #idle += int(np.count_nonzero(id))
                            break
                is_hit_candidate = []
                delta = 0.02
                is_hit_candidate = hit_by_batch_numba(self.obs , working)
                #self.N_hits += is_hit_candidate
                #self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v3(delta), working))
                #self.N_hits -= is_hit_candidate
                self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), working))
                #print("length of cliques with epsilon is position 2", len(self.cliques_with_epsilon))
                #delta = 0.33
            #epsilon_shadow = []
            is_hit_candidate = []
            if not any((existing == shadowcliquesetting).all() for existing in self.clean_setting_cache[center_node]):
                    self.clean_setting_cache[center_node].append(shadowcliquesetting)
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            #self.N_hits += is_hit_candidate
            #self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v3(delta), shadowcliquesetting))
            #self.N_hits -= is_hit_candidate
            self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), shadowcliquesetting))
            #self.N_hits += is_hit_candidate
            #epsilon_shadow = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            #self.N_hits -= is_hit_candidate
            if not self.cliques_with_epsilon:
                raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
            print("length of cliques with epsilon is", len(self.cliques_with_epsilon))
            #print("completed using shadow ", numincomplete)
            #print("completed using setting list ", numsetting)
            #print("size of idle part", idle)
            self.cliques_with_epsilon.sort(key=lambda x: x[0], reverse = True)
            _, best_clique = self.cliques_with_epsilon[0]
            self.selected_cliques.append(best_clique)
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
            else:
                print("epsilon for best clique is",self.cliques_with_epsilon[0])
            setting = best_clique
            if not any((existing == best_clique).all() for existing in self.incompletesettingcache):
                self.incompletesettingselected += 1
            print("incomplete setting was selected",self.incompletesettingselected,"times")
            print("Shadow clique was selected", self.shadow_was_best_count, "times")
            #epsilon_best = []
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , best_clique)
            self.N_hits += is_hit_candidate
            epsilon_best = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.N_hits -= is_hit_candidate
        else:
            shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
            for idx in reversed(order):
                o = self.obs[idx]
                if verbose:
                    print("Checking",o)
                if hit_by(o,shadowcliquesetting):
                    non_id = o!=0
                    shadowcliquesetting[non_id] = o[non_id]
                if verbose:
                    print("p =",setting)
                if np.min(shadowcliquesetting) > 0:
                    break
            if verbose:
                print("Checking list of observables.")
                
            delta = 0.02
            removedcliques = 0
            completecliques = 0
            valid_settings = []
            valid_cliques = []
            #START
            if center_node not in self.processed_center_node:  #gggggggg
                self.processed_center_node.append(center_node)
                hit_list = []
                hit_cliques = []
                non_id = first_obs != 0
                globalsetting[non_id] = first_obs[non_id]
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
                if not hit_graph:
                    raise RuntimeError("No hit graph found.")
                hit_cliques = find_cliques5(hit_graph)
                #print("we fount these cliques",hit_cliques)
                # Cache result
                self.clique_cache[center_node] = hit_cliques
                #hit_cliques.append(shadowclique)
                if not hit_cliques:
                    hit_cliques=[[center_node]]
                self.clean_setting_cache[center_node] = []
                self.clean_clique_cache[center_node] = []
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
                        """if np.min(setting_candidate) == 0:
                            for idx in reversed(order):
                                o = self.obs[idx]
                                if verbose:
                                    print("Checking",o)
                                if hit_by(o,setting_candidate):
                                    non_id = o!=0
                                    # overwrite those qubits that fall in the support of o
                                    setting_candidate[non_id] = o[non_id]
                                if verbose:
                                    print("p =",setting_candidate)
                                # break sequence is case all identities in setting are exhausted
                                if np.min(setting_candidate) > 0:
                                    print("completed setting using shadow list")
                                    break"""
                    self.clean_clique_cache[center_node].append(clique)
                    if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache[center_node]):
                        self.clean_setting_cache[center_node].append(setting_candidate)
            else:
                print("I already found cliques for this node")
            
                """else:
                    completecliques += 1
                    self.clean_clique_cache[center_node].append(clique)
                    if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache[center_node]):
                        self.clean_setting_cache[center_node].append(setting_candidate)"""
            #settingslist = []
            #settingslist = [s for _, s in self.cliques_with_epsilon]
            #print("length of settings list is", len(settingslist))
            self.cliques_with_epsilon = []
            self.incompletesettingcache = []
            numincomplete = 0
            #numsetting = 0
            for cached_settings in self.clean_setting_cache.values(): #ggggggg Start
                for setting_candidate in cached_settings:
                    working = setting_candidate.copy()
            #for setting_candidate in self.clean_setting_cache[center_node]:
                    """if np.min(working) == 0:
                        for o in settingslist:
                            if verbose:
                                print("Checking",o)
                            if hit_by(o,working):
                                non_id = o!=0
                                working[non_id] = o[non_id]
                            if verbose:
                                print("p =",working)
                            if np.min(working) > 0:
                                #print("completed setting using setting list")
                                numsetting += 1
                                break"""
                    
                    if np.min(working) == 0:
                        for idx in reversed(order):
                            o = self.obs[idx]
                            if verbose:
                                print("Checking",o)
                            if hit_by(o,working):
                                non_id = o!=0
                                # overwrite those qubits that fall in the support of o
                                working[non_id] = o[non_id]
                            if verbose:
                                print("p =",working)
                            # break sequence is case all identities in setting are exhausted
                            if np.min(working) > 0:
                                #print("completed setting using shadow list")
                                self.incompletesettingcache.append(working)
                                numincomplete += 1
                                break
                    is_hit_candidate = []
                    is_hit_candidate = hit_by_batch_numba(self.obs , working)
                    #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), setting_candidate))
                    self.N_hits += is_hit_candidate
                    self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), working))
                    #self.cliques_with_epsilon.append((get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False), setting_candidate))
                    self.N_hits -= is_hit_candidate #gggggg end
            is_hit_candidate = []
            if not any((existing == shadowcliquesetting).all() for existing in self.clean_setting_cache[center_node]):
                    self.clean_setting_cache[center_node].append(shadowcliquesetting)
            #self.clean_setting_cache[center_node].append(shadowcliquesetting)
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), setting_candidate))
            self.N_hits += is_hit_candidate
            #self.cliques_with_epsilon.append((get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False), shadowcliquesetting))
            epsilon_shadow = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), shadowcliquesetting))
            self.N_hits -= is_hit_candidate
            if not self.cliques_with_epsilon:
                raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
            print("length of cliques with epsilon is", len(self.cliques_with_epsilon))
            #print("completed using shadow ", numincomplete)
            #print("completed using setting list ", numsetting)
            # Sort cliques by epsilon
            self.cliques_with_epsilon.sort(key=lambda x: x[0])
            _, best_clique = self.cliques_with_epsilon[0]
            #print("epsilon for best clique is",self.cliques_with_epsilon[0])
            self.selected_cliques.append(best_clique)
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
            else:
                print("epsilon for best clique is",self.cliques_with_epsilon[0])
            #print("best clique has",len(best_clique),"members")
            setting = best_clique
            if not any((existing == best_clique).all() for existing in self.incompletesettingcache):
                self.incompletesettingselected += 1
            print("incomplete setting was selected",self.incompletesettingselected,"times")
            print("Shadow clique was selected", self.shadow_was_best_count, "times")
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , best_clique)
            self.N_hits += is_hit_candidate
            epsilon_best = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.N_hits -= is_hit_candidate
        tend = time()
        is_hit = []
        is_hit = hit_by_batch_numba(self.obs , setting)
        self.N_hits += is_hit
        delta = 0.02
        self.round_num += 1    
        self.rounds.append(len(self.rounds) + 1)
        print("round number" , self.round_num)
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["Provable Gaurantee"] = Guaranteed_accuracy(delta, self.N_hits, self.w, split=False)
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
        info["epsilon_Bernstein_no_restricted_validity_v3"] = self.get_epsilon_Bernstein_no_restricted_validity_v3(delta)
        info["epsilon_Bernstein_scalar_no_restricted_validity"] = get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False)
        #info["epsilon difference"]= abs(epsilon_best - epsilon_shadow)
        #info["incomplete setting selected"] = self.incompletesettingselected
        #info["shadow setting selected"] = self.shadow_was_best_count
        self.eps_values_v3.append(info["epsilon_Bernstein_no_restricted_validity_v2"])
        #print("difference between best epsilon and shadow epsilon = ", info["epsilon difference"])
        #self.eps_values_v3.append(info["epsilon difference"])
        self.inconfidence.append(info["inconfidence_bound"])
        #print("epsilon_Bernstein_scalar_no_restricted_validity:", info["epsilon_Bernstein_scalar_no_restricted_validity"])
        print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta, split=True)
        print("epsilon_Bernstein_no_restricted_validity_v2:", info["epsilon_Bernstein_no_restricted_validity_v2"])
        #print("shadow was selected",info["shadow setting selected"],"times")
        #print("incomplete setting was selected",info["incomplete setting selected"],"times")
        #print("inconfidence bound:",info["inconfidence_bound"])
        #print("Provable Gauarantee :", info["Provable Gaurantee"])
        self.provablegaurantee.append(info["Provable Gaurantee"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info


class Shadow_Grouping(Measurement_scheme):
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
    
    def __init__(self,observables,weights,epsilon,weight_function, cov_real, compute_N_hits_pairs=True):
        super().__init__(observables,weights,epsilon)
        #self.settings_dict = {}
        self.N_hits = np.zeros_like(self.N_hits)
        self.cov_real = cov_real
        self.compute_N_hits_pairs = compute_N_hits_pairs
        if compute_N_hits_pairs:
            self.N_hits_pairs = np.zeros((self.num_obs, self.num_obs), dtype=int)
        self.weight_function = weight_function
        self.rounds = []
        self.eps_values_v3 = []
        self.eps_chebyshev_tighter = []
        self.eps_chebyshev_tightest = []
        self.inconfidence = []
        self.provablegaurantee = []
        self.round_num = 0
        self.commutativity_type = 'qwc'
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.N_hits_pairs  = np.zeros_like(self.N_hits_pairs)
        self._hit_outer_cache = {}
        self.settings_dict = {} 
        self.settings_buffer = {}
        return
    
    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
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
        setting = np.zeros(self.num_qubits,dtype=int)
        #print("N_hits before update:", self.N_hits)
        #if not np.any(self.N_hits) == 0:
            #print("now every observable is checked at least once")
        # Get highest-weight observable

        # Get highest-weight observable
        # first_idx = order[-1]  # last one in ascending sort = highest weight
        # first_obs = self.obs[first_idx]
        # center_node = tuple(first_obs)  # Use tuple as dictionary key
        # print("center node is", first_idx, "and its weight is", weights[first_idx])
        
        if verbose:
            print("Checking list of observables.")
        tstart = time()
        for idx in reversed(order):
            o = self.obs[idx]
            if verbose:
                print("Checking",o)
            if hit_by(o,setting):
                non_id = o!=0
                # overwrite those qubits that fall in the support of o
                setting[non_id] = o[non_id]
                if verbose:
                    print("p =",setting)
                # break sequence is case all identities in setting are exhausted
                if np.min(setting) > 0:
                    break
                    
        tend = time()

        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit

        # Tokenize by the set of compatible observable indices
        setting_indices = np.nonzero(is_hit)[0].astype(np.int32)
        setting_indices.sort()
        token = encode_setting_token(setting_indices)


        if self.compute_N_hits_pairs:
            # Cache (or retrieve) the canonical index list for this token
            idx = self._append_is_hit_hit_outer(token, setting_indices)
            # Apply the outer update without building an outer product
            if idx.size:
                self.N_hits_pairs[np.ix_(idx, idx)] += 1


        delta = 0.33
        self.round_num += 1
        self.rounds.append(len(self.rounds) + 1)
        # further info for comparisons
        info = {}
        """info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["Provable Gaurantee"] = Guaranteed_accuracy(delta, self.N_hits, self.w, split=False)
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein"] = self.get_epsilon_Bernstein(delta)
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
        #print("epsilon_Bernstein_no_restricted_validity_v2:", info["epsilon_Bernstein_no_restricted_validity_v2"])
        info["epsilon_Bernstein_no_restricted_validity_v3"] = self.get_epsilon_Bernstein_no_restricted_validity_v3(delta)
        info["epsilon_Bernstein_scalar_no_restricted_validity"] = get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False)
        self.eps_values_v3.append(info["epsilon_Bernstein_no_restricted_validity_v2"])
        self.inconfidence.append(info["inconfidence_bound"])
        #print("epsilon_Bernstein_scalar_no_restricted_validity:", info["epsilon_Bernstein_scalar_no_restricted_validity"])
        #self.eps_values_v3.append(info["epsilon_Bernstein_no_restricted_validity_v3"])
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        if verbose:
            print("round number", self.round_num)
            print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("epsilon_Bernstein_no_restricted_validity_v2:", info["epsilon_Bernstein_no_restricted_validity_v2"])
        #print("Inconfidence Bound :", info["inconfidence_bound"])
        #print("Provable Gauarantee :", info["Provable Gaurantee"])
        self.provablegaurantee.append(info["Provable Gaurantee"])
        info["epsilon_chebyshev_tighter"] = get_epsilon_Chebyshev_scalar_tighter_numba(delta, self.N_hits, self.N_hits_pairs, self.w)
        self.eps_chebyshev_tighter.append(info["epsilon_chebyshev_tighter"])
        info["epsilon_chebyshev_tightest"] = get_epsilon_Chebyshev_scalar_tightest_numba(delta, self.N_hits, self.N_hits_pairs, self.w, self.cov_real)
        self.eps_chebyshev_tightest.append(info["epsilon_chebyshev_tightest"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        #print("update0 info is",info)"""
        return setting_indices, info

class ShadowBucket(Measurement_scheme):
    """ do shadowgrouping and store the generated settings at the same time, in next rounds, compare the epsilon of the shadow clique with the epsilon of the best clique from the cache and select the best one.
    """
    
    def __init__(self, observables, weights, epsilon, weight_function):
        # Convert Pauli strings to arrays FIRST
        #observablesarray = [pauli_string_to_array(o) for o in observables]
    
        # Then pass converted observables into super().__init__()
        #super().__init__(observablesarray, weights, epsilon)
        super().__init__(observables,weights,epsilon)
        #self.settings_dict = {} 222222222222
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.round_num = 0
        self.rounds = []
        self.eps_values_v3 = []
        self.provablegaurantee = []
        self.inconfindence = []
        self.shadow_was_best_count = 0
        self._cached_settings = []
        
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function is supposed to return an array of shape {} "
                "(i.e. number of observables) but returned an array of shape {}".format(self.w.shape, test.shape)
            )
        self.is_sampling = False
        return

    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.settings_dict = {} #2222222222222
        self.settings_buffer = {} #2222222222222
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

    def total_hit_weight(self, weights, is_hit):
        weights = np.asarray(weights, dtype=float)
        is_hit = np.asarray(is_hit, dtype=bool)
        return (weights * is_hit).sum()
    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        #if self._cached_graph is None or self._cached_cliques is None:
        #    self.build_graph_and_cliques()
        
        weights = self.weight_function(self.w, self.eps, self.N_hits)
        tstart = time()
        order = np.argsort(weights)
        completecliques = 0
        #self.cliques_with_epsilon = []
        delta = 0.02
        #alpha = 51733.57
        incompletesetting = 0
        if np.any(self.N_hits == 0):
            """settinglist = []
            if self.cliques_with_epsilon:
                settingslist = [s for _, s in self.cliques_with_epsilon]"""
            self.cliques_with_epsilon = []
            shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
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
            if not any(np.array_equal(existing, shadowcliquesetting) for existing in self._cached_settings):
                self._cached_settings.append(shadowcliquesetting.copy())
            #for setting_candidate in self.clean_setting_cache[center_node]:
            #for cached_settings in self._cached_settings.values(): #ggggggg Start
            for setting_candidate in self._cached_settings:
                working = setting_candidate.copy()
                #print(working)
                """if np.min(setting_candidate) == 0:
                    if settinglist:
                        for o in settingslist:
                            if verbose:
                                print("Checking",o)
                            if hit_by(o,setting_candidate):
                                non_id = o!=0
                                setting_candidate[non_id] = o[non_id]
                            if verbose:
                                print("p =",setting_candidate)
                            if np.min(setting_candidate) > 0:
                                print("completed setting using setting list")
                                break"""
                if np.min(working) == 0:
                    #print(working)
                    incompletesetting += 1
                    for idx in reversed(order):
                        o = self.obs[idx]
                        if verbose:
                            print("Checking",o)
                        if hit_by(o,working):
                            non_id = o!=0
                            # overwrite those qubits that fall in the support of o
                            working[non_id] = o[non_id]
                        if verbose:
                            print("p =",working)
                        # break sequence is case all identities in setting are exhausted
                        if np.min(working) > 0:
                            break
                    completecliques += 1
                else:
                    completecliques += 1
                is_hit_candidate = []
                is_hit_candidate = hit_by_batch_numba(self.obs , working)
                self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), working))
                #self.N_hits += is_hit_candidate
                #self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), setting_candidate))
                #self.N_hits -= is_hit_candidate #ggggg end
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), shadowcliquesetting))
            #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), setting_candidate))
            #self.N_hits += is_hit_candidate
            #self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), shadowcliquesetting))
            #self.cliques_with_epsilon.append((self.get_Bernstein_bound(), shadowcliquesetting))
            #print("epsilon for shadow clique is",self.get_epsilon_Bernstein_no_restricted_validity(delta))
            #self.N_hits -= is_hit_candidate
            self.N_hits += is_hit_candidate
            epsilon_shadow = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.N_hits -= is_hit_candidate
            self.cliques_with_epsilon.sort(key=lambda x: x[0], reverse = True)
            print("length of cliques with epsilon is", len(self.cliques_with_epsilon))
            print("number of incomplete settings", incompletesetting)
            _, best_clique = self.cliques_with_epsilon[0]
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
            else:
                print("epsilon for best clique is",self.cliques_with_epsilon[0])
            setting = best_clique
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , best_clique)
            self.N_hits += is_hit_candidate
            epsilon_best = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.N_hits -= is_hit_candidate
        else:
            #for cached_settings in self._cached_settings.values(): #ggggggg Start
            #self.cliques_with_epsilon = []
            """settinglist = []
            settingslist = [s for _, s in self.cliques_with_epsilon]"""
            self.cliques_with_epsilon = []
            shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
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
            if not any(np.array_equal(existing, shadowcliquesetting) for existing in self._cached_settings):
                self._cached_settings.append(shadowcliquesetting.copy())
            for setting_candidate in self._cached_settings:
                working = setting_candidate.copy()
                #print(working)
                """if np.min(setting_candidate) == 0:
                    for o in settingslist:
                        if verbose:
                            print("Checking",o)
                        if hit_by(o,setting_candidate):
                            non_id = o!=0
                            setting_candidate[non_id] = o[non_id]
                        if verbose:
                            print("p =",setting_candidate)
                        if np.min(setting_candidate) > 0:
                            print("completed setting using setting list")
                            break"""
                if np.min(working) == 0:
                    incompletesetting += 1
                    for idx in reversed(order):
                        o = self.obs[idx]
                        if verbose:
                            print("Checking",o)
                        if hit_by(o,working):
                            non_id = o!=0
                            # overwrite those qubits that fall in the support of o
                            working[non_id] = o[non_id]
                        if verbose:
                            print("p =",working)
                        # break sequence is case all identities in setting are exhausted
                        if np.min(working) > 0:
                            break
                    completecliques += 1
                else:
                    completecliques += 1
                is_hit_candidate = []
                is_hit_candidate = hit_by_batch_numba(self.obs , working)
                #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), setting_candidate))
                self.N_hits += is_hit_candidate
                self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v3(delta), working))
                #self.cliques_with_epsilon.append((get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False), setting_candidate))
                self.N_hits -= is_hit_candidate #ggggg end
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), shadowcliquesetting))
            #self.cliques_with_epsilon.append((self.total_hit_weight(weights, is_hit_candidate), setting_candidate))
            self.N_hits += is_hit_candidate
            self.cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v3(delta), shadowcliquesetting))
            epsilon_shadow = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            #self.cliques_with_epsilon.append((get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False), shadowcliquesetting))
            #self.cliques_with_epsilon.append((self.get_Bernstein_bound(), shadowcliquesetting))
            #print("epsilon for shadow clique is",self.get_epsilon_Bernstein_no_restricted_validity(delta))
            self.N_hits -= is_hit_candidate
            self.cliques_with_epsilon.sort(key=lambda x: x[0])
            print("length of cliques with epsilon is", len(self.cliques_with_epsilon))
            _, best_clique = self.cliques_with_epsilon[0]
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
            else:
                print("epsilon for best clique is",self.cliques_with_epsilon[0])
            setting = best_clique
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , best_clique)
            self.N_hits += is_hit_candidate
            epsilon_best = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
            self.N_hits -= is_hit_candidate

        tend = time()
        print("Shadow clique was selected", self.shadow_was_best_count, "times")
        is_hit = []
        # update number of hits
        is_hit = hit_by_batch_numba(self.obs , setting)
        self.N_hits += is_hit
        delta = 0.02
        self.round_num += 1    
        self.rounds.append(len(self.rounds) + 1)
        print("round number" , self.round_num)
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["Provable Gaurantee"] = Guaranteed_accuracy(delta, self.N_hits, self.w, split=False)
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
        info["epsilon_Bernstein_no_restricted_validity_v3"] = self.get_epsilon_Bernstein_no_restricted_validity_v3(delta)
        info["epsilon_Bernstein_scalar_no_restricted_validity"] = get_epsilon_Bernstein_scalar_no_restricted_validity(delta, self.N_hits, self.w, split=False)
        self.eps_values_v3.append(info["epsilon_Bernstein_no_restricted_validity_v2"])
        self.inconfindence.append(info["inconfidence_bound"])
        info["epsilon difference"]= abs(epsilon_best - epsilon_shadow)
        #print("difference between best epsilon and shadow epsilon = ", info["epsilon difference"])
        #self.eps_values_v3.append(info["epsilon difference"])
        #self.eps_values_v3.append(info["epsilon_Bernstein_no_restricted_validity_v3"])
        #print("epsilon_Bernstein_scalar_no_restricted_validity:", info["epsilon_Bernstein_scalar_no_restricted_validity"])
        print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta, split=True)
        print("epsilon_Bernstein_no_restricted_validity_v2:", info["epsilon_Bernstein_no_restricted_validity_v2"])
        #print("Inconfidence Bound :", info["inconfidence_bound"])
        #print("Provable Gauarantee :", info["Provable Gaurantee"])
        self.provablegaurantee.append(info["Provable Gaurantee"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info


class Brute_force_matching(Shadow_Grouping):
    """ Comparison class to Shadow_Grouping. Runs through all 3**num_qubit possibilities, thus finding the optimal next
        measurement setting p.
        The target (str or user_function) specifies the member function (if str) to maximize over (defaults to Bernstein bound).
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,target="Bernstein_bound"):
        super().__init__(observables,weights,epsilon,None)
        if isinstance(target,str):
            self.target_is_member_function = True
            try:
                self.weights = getattr(self,"get_"+target)
            except:
                print("Warning! Unknown member-function get_{} called. Defaulting to get_Bernstein_bound instead.".format(target))
                self.weights = self.get_Bernstein_bound
        else:
            self.target_is_member_function = False
            self.weights = target
        self.is_sampling = False
        return
    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        best_setting, best_weight = [], np.inf
        if verbose:
            print("Brute-force searching all measurement settings")
        tstart = time()
        for P in product(range(1,4),repeat=self.num_qubits):
            temp_hit = np.array([hit_by(o,P) for o in self.obs])
            self.N_hits += temp_hit
            temp = self.weights() if self.target_is_member_function else np.sum(self.weights(self.w,self.eps,self.N_hits))
            self.N_hits -= temp_hit
            if temp < best_weight:
                best_setting, best_weight = [P], temp
            elif temp == best_weight:
                best_setting.append(P)
        tend = time()
        if verbose:
            print("Best assignment are {} with max weight of {}".format(best_setting,best_weight))
        
        # if multiple setting have been found, returns one at random
        n = len(best_setting)
        if n==1:
            setting = best_setting[0]
        else:
            setting = best_setting[np.random.choice(n)]
            
        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        
        info = {"best_settings":      best_setting,
                "total_weight":       best_weight,
                "inconfidence_bound": self.get_inconfidence_bound(),
                "Bernstein bound":    self.get_Bernstein_bound(),
                "run_time":           tend - tstart
               }
            
        return np.array(setting), info        

class AdaptiveShadows(Measurement_scheme):
    """ Comparison class to Shadow_Grouping, based on https://github.com/charleshadfield/adaptiveshadows/.
        Starts-off as classical shadows (uniformly at random) but biases the distribution
        the more the Pauli bases have been set. Does not require any hyperparameters.
        epsilon (optional): parameter solely used for comparison with other methods. Defaults to 0.1.
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon=0.1, cov_real=None):
        super().__init__(observables,weights,epsilon)
        self.is_sampling = True
        self.cov_real = cov_real
        return
    
    def __isCompatible(self, pauli, j, qubits_shift, bases_shift):
        """ Helper function to check whether the current pauli term is compatible with the current
            partial assignment and whether the pauli term has a non-identity at the current qubit index.
        """
        if pauli[qubits_shift[j]] == 0:
            return False
        for k in range(j):
            i = qubits_shift[k]
            if not pauli[i] in (0, bases_shift[k]):
                return False
        return True
    
    def __generateBeta(self, j, qubits_shift, bases_shift):
        """ Calculate the probabilities for drawing either X,Y or Z for the j-th qubit in permuted order.
            This assignment is conditioned on the previously assigned qubits in the current iteration.
        """
        constants = [0.0, 0.0, 0.0]
        # loop through all Pauli terms with their respective weights
        for coeff, pauli in zip(self.w, self.obs):
            # if current term is still compatible with current assignment
            # and does not yield an identity at the current qubit index,
            # adjust the corresponding weights
            if self.__isCompatible(pauli, j, qubits_shift, bases_shift):
                index = pauli[qubits_shift[j]] - 1 # index pauli[...] cannot be the identity
                constants[index] += coeff**2
        beta_unnormalized = np.sqrt(constants)
        norm = np.sum(beta_unnormalized)
        if norm == 0:
            beta = np.ones(3)/3
        else:
            beta = beta_unnormalized / norm
        return beta
    
    def __generateBasisSingle(self, j: int, qubits_shift: list, bases_shift: list) -> str:
        """ Sample the operator for the j-th qubit in permuted order. """
        assert len(bases_shift) == j
        beta = self.__generateBeta(j, qubits_shift, bases_shift)
        basis = np.random.choice([1, 2, 3], p=beta)
        return basis
    
    def find_setting(self,verbose=False):
        """ Generate the next Pauli measurement string by randomly permuting the qubits and sampling from
            beta = otimes_i beta_i
        """
        n = self.num_qubits
        # randomly permute the qubit order
        tstart = time()
        qubits_shift = list(np.random.permutation(n))
        bases_shift = []
        for j in range(n):
            basisSingle = self.__generateBasisSingle(j, qubits_shift, bases_shift)
            bases_shift.append(basisSingle)
        # undo the permutation by adding the single operators to output basis B
        setting = []
        for i in range(n):
            j = qubits_shift.index(i)
            setting.append(bases_shift[j])
            
        tend = time()
            
        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        setting_indices = np.nonzero(is_hit)[0].astype(np.int32)
        setting_indices.sort()
        token = encode_setting_token(setting_indices)
        info = {}
            
        return setting_indices, info
            
class SettingSampler(Measurement_scheme):
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
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
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

    def find_setting(self,N_samples=1):
        """ Generate settings from the given distribution p. Can find multiple settings at once by providing a value for
            N_samples (int). Returns the setting(s) and a dictionary holding the information about the number of settings sampled.
        """
        inds = np.random.choice(len(self.p),size=(N_samples,),p=self.p)
        Q = self.settings[inds]
        for ind, repeats in zip(*np.unique(inds,return_counts=True)):
            # update number of hits for each of the unique elements in Q
            # by counting over the index vector, instead
            is_hit = np.array([hit_by(o,self.settings[ind]) for o in self.obs],dtype=int)
            self.N_hits += is_hit*repeats
        if N_samples==1:
            Q = Q.flatten()
        return Q, {"N_samples": N_samples}
    
class Derandomization(Measurement_scheme):

    """ Finds the next measurement setting following the derandomization procedure.
        Optionally, a parameter delta in [0,1] can be provided to vary the degree of randomness (delta == 1 fully random, delta == 0 as proposed).
        If num_measurements is provided, the corresponding inconfidence bound is adapted to that.
        If use_one_norm, implements a 1-norm weighting to the bound as proposed in the paper.
    """

    def __init__(self,observables,weights,epsilon,cov_real=None,delta=0,num_measurements=None,use_one_norm=False):
        super().__init__(observables,weights,epsilon)
        
        self.num_measurements = num_measurements
        # (n x M) integer array with entries in {0,1,2,3} == {E,X,Y,Z}
        self.localities = np.zeros((self.num_qubits+1,self.num_obs),dtype=int) # keep the last zero as the support of an empty Pauli string
        self.localities[:-1,:] = np.array([np.sum(observables[:,i:]!=0,axis=1) for i in range(self.num_qubits)])
        self.N_hits = np.zeros(self.num_obs,dtype=int)
        self.eps_greedy = delta
        self.scheme_params["eps_greedy"] = delta
        self.scheme_params["use_one_norm"] = use_one_norm
        self.cov_real = cov_real
        
        if use_one_norm:
            self.use_one_norm = True
            self.w_factor = np.abs(self.w)
            self.w_factor /= np.max(self.w_factor)
            #self.wmax = np.max(np.abs(self.w))
            self.nu = 1 - np.exp(-epsilon*epsilon/2)
        else:
            self.use_one_norm = False
            self.w_factor = self.w**2
            self.nu = 1 - np.exp(-epsilon*epsilon/2/self.w/self.w)
            
        self.log_locality_factor = np.log(1-self.nu/(3**self.localities[0]))
        
        self.assignments = [] # for the next measurement setting
        self.m_k_counter = [0,0] # convenience internal counter = (num_settings so far, current qubit pos)
        self.last_assignment = None
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.assignments = []
        self.m_k_counter = [0,0]
        self.last_assignment = None
        return
    
    def get_inconfidence_bound(self):
        inconf = np.exp( -0.5*self.eps*self.eps*self.N_hits/(self.w**2) )
        return np.sum(inconf)

    def __step(self, action):
        """ Tries out the effect of the chosen assignment.
            Returns the corresponding inconfidence bound upon this choice and an increment.
            It is a boolean list in case a new measurement setting is produced and None-type else.
        """
        
        self.assignments.append(action) # actions are in {1,2,3}
        self.m_k_counter[1] += 1
        # check whether to roll over to next measurement setting
        if len(self.assignments) >= self.num_qubits:
            self.m_k_counter = [self.m_k_counter[0]+1, 0]
            # start new measurement setting and check whether the previous setting hits any observables
            self.last_assignment = self.assignments.copy()
            increment = np.array([hit_by(self.obs[i],self.last_assignment) for i in range(self.num_obs)],dtype=int)
            self.N_hits += increment
            self.assignments = []
        else:
            increment = None

        return self.derandom_bound(), increment
    
    def __step_back(self,increment=None):
        """ Reverts the effect of _step() in terms of internal counters. """
        if len(self.assignments) == 0:
            # revert to old measurement setting in case of roll-over
            self.m_k_counter[0] -= 1 # decrease num_settings by one
            self.m_k_counter[1] = self.num_qubits - 2
            assert increment is not None, "Increment should not have been None-type when rolling back."
            self.N_hits -= increment
            if self.last_assignment is not None:
                self.assignments = self.last_assignment[:-1]
            else:
                self.m_k_counter = [0,0] # reinitialize in this case
        else:
            self.assignments.pop()
            self.m_k_counter[1] -= 1
        return

    def derandom_bound(self):
        """ Given a set of previous assignments in self.assignments, calculates the current inconfidence bound. """
        m,qubit_k = self.m_k_counter
        p = self.assignments
        temp = self.nu/(3**self.localities[qubit_k])
        # calculate product of the second term for the first k qubit operators
        sign = np.array([hit_by(o[:qubit_k],p) for o in self.obs])
        temp = np.log(1-temp*sign) # element-wise operations
        # first term for every observable
        if self.use_one_norm:
            temp -= self.eps*self.eps/2*self.N_hits
            temp /= self.w_factor
        else:
            temp -= self.eps*self.eps/2*self.N_hits/self.w_factor
        # third term for every observable if applicable
        if self.num_measurements is not None:
            temp += (self.num_measurements-m-1)*self.log_locality_factor
        bound = np.sum(np.exp(temp))
        return bound
    
    def find_setting(self, verbose=False, previous_bound=None):
        """ Tries all three possible Pauli assignments and picks epsilon-greedy to minimize the inconf. bound  """
        assert self.assignments == [], "Current assignment list is not empty. Please empty first."
        if self.num_measurements is not None:
            if self.m_k_counter[0] >= self.num_measurements:
                print("Warning! Measurement scheme already reached the max. number of measurements, given by {}. Returned an empty assignment".format(self.num_measurements))
                return [], {}
        previous_bound = self.get_inconfidence_bound() if previous_bound is None else previous_bound
        info = {"previous_bound": previous_bound}
        tstart = time()
        if verbose:
            print("Running epsilon-greedy derandomized scheme with epsilon = {}".format(self.eps_greedy))
        for n in range(self.num_qubits):
            if np.random.rand() < self.eps_greedy:
                # check for random action with probability eps_var
                action = np.random.choice(3) + 1
                inconf, increment = self.__step(action)
                assert increment is None or n+1 == self.num_qubits, "Increment was not None-type but should have been."
            else:
                # pick among argmin else
                temp = []
                for i in range(1,4):
                    inconf, increment = self.__step(i)
                    assert increment is None or n+1 == self.num_qubits, "Increment was not None-type but should have been."
                    temp.append(previous_bound - inconf)
                    self.__step_back(increment)
                action = np.argmax(temp) + 1
                inconf, increment = self.__step(action)
            previous_bound = inconf
            if verbose:
                temp = self.assignments if n + 1 < self.num_qubits else self.last_assignment
                print(temp)
        tend = time()
        assert increment is not None, "Increment was None-type but should have been list."        
            
        # further information
        #info["total_weight"] = np.sum(self.get_inconf()[increment])
        #info["inconfidence_bound"] = self.get_inconfidence_bound()
        #info["Bernstein bound"] = self.get_Bernstein_bound()
        #info["run_time"] = tend - tstart
        #if verbose:
            #print("Finished assigning with total weight of",info["total_weight"])

        setting_indices = np.nonzero(increment)[0].astype(np.int32)
        setting_indices.sort()
        token = encode_setting_token(setting_indices)
        info = {}
            
        return setting_indices, info
        
        #return np.array(self.last_assignment), info



import networkx as nx
import matplotlib.pyplot as plt

class DomClique(Measurement_scheme):
    def __init__(self, observables, weights):
        """
        Initialize the QubitGraphAnalyzer with observables and weights.
        
        Args:
            observables (list): A list of observables.
            weights (list): A list of weights corresponding to the observables.
        """
        if len(observables) != len(weights):
            raise ValueError("The length of 'observables' and 'weights' must be the same.")
        
        self.observables = observables
        self.obs = observables
        self.w = weights
        self.graph = nx.Graph()
        self.num_qubits = observables.shape[1]  # Assuming observables is a NumPy array
        self.is_adaptive = False
        self.twe, self.tcwe = 0, 0  # Edge weight calculations
        self.nwe, self.tnwe = 0, 0  # Node weight calculations
        self.lwe, self.tlwe = 0, 0  # Local weight calculations
        self.wavg = np.zeros(len(observables))
        #self.neighbournum = np.zeros(len(observables))
        self.nodeweight = np.zeros(len(observables))
        #self._build_graph()
        #self.update_variance_estimate()
        self.is_sampling = True
        # Initialize N_hits as a dictionary or any other structure you need
        self.N_hits = np.zeros(len(observables),dtype=int)
        self._build_graph()  
        self.neighbournum = {node: len(list(self.graph.neighbors(node))) for node in self.graph.nodes}
        self.sort_nodes()
        self.greedy_ndominating_set()
        self.maximal_cliques()
        

    def reset(self):
        """
        Reset all attributes to their initial state, clearing the graph and any computed properties.
        """
        # Clear the graph
        self.graph.clear()
        # Reset graph-related weights and totals
        self.twe, self.tcwe = 0, 0  # Edge weight calculations
        self.nwe, self.tnwe = 0, 0  # Node weight calculations
        self.lwe, self.tlwe = 0, 0  # Local weight calculations
        self.N_hits = np.zeros(len(self.N_hits),dtype=int)

        # Reset node attributes
        self.wavg = np.zeros(len(self.observables))
        #self.neighbournum = np.zeros(len(self.observables))
        self.nodeweight = np.zeros(len(self.observables))
        #self.update_variance_estimate()

    def find_setting(self):
        #print("shape of clique in main form",clique)
        #print("what is dominating set",self.ndominating_set)
        # transform into Pauli string for compatibility with parent class
        print("maximum cliques",self.MaxCliques)
        setting = self._clique_to_Pauli_observable()[0]  # No errors here
        #print("Shape of the clique in DomClique:", clique.shape)
        # update class counters
        #clique = clique .flatten()
        #print("Shape of the clique after flaatten:", clique.shape)
        #if tuple(clique) not in self.N_hits:
        #self.N_hits[tuple(clique)] = 0  # Initialize to 0
        self.N_hits[self.MaxCliques] += 1  # Now increment safely

        
        #setting = setting [0]
        #print("outcome setting of DomClique",setting)
        # Print the shape of setting
        #print("Shape of the setting in DomClique:", setting.shape)
        #print(type(setting))  
        #print(setting.shape)
        setting = np.atleast_1d(setting)  # Convert scalar to array if needed
        print("outcome of DomClique",setting)
        return setting,{}
    
    def _build_graph(self):
        """Build the graph by adding edges based on commutativity."""
        # Ensure all nodes are added to the graph before adding edges
        for i in range(len(self.observables)):
            self.graph.add_node(i)  # Add node unconditionally
        # Now add edges based on commutativity
        for i in range(len(self.observables)):
            tavg, conn = 0, 0
            for j in range(i + 1, len(self.observables)):
                if hit_by(self.observables[i], self.observables[j]):
                    we = round(np.abs(self.w[i]) * np.abs(self.w[j]), 5)
                    self.graph.add_edge(i, j, weight=we)
                    self.twe += we
                    self.nwe += np.abs(self.w[i]) + np.abs(self.w[j])
                    self.lwe += np.abs(self.w[i]) * np.abs(self.w[j])
                    tavg += np.abs(self.w[i]) * np.abs(self.w[j])
                    conn += 1

            self.wavg[i] = 0 if conn == 0 else tavg / conn
            #self.neighbournum[i] = conn
            self.nodeweight[i] = tavg

        # Calculate theoretical total edge and node weights
        for i in range(len(self.observables)):
            for j in range(i + 1, len(self.observables)):
                self.tcwe += round(np.abs(self.w[i]) * np.abs(self.w[j]), 5)
                self.tnwe += np.abs(self.w[i]) + np.abs(self.w[j])
                self.tlwe += np.abs(self.w[i]) * np.abs(self.w[j])

        #nx.draw(self.graph, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, font_weight='bold')
        # Return the built graph
        return self.graph

    def sort_nodes(self):
        """
        Sort nodes based on number of neighbours, total weight, and average weight, 
        and print the sorted results.
        """
        # Sort nodes based on number of neighbours
        self.nsorted_indices = sorted(self.neighbournum.keys(), key=lambda x: self.neighbournum[x], reverse=True)
        return self.nsorted_indices

    def greedy_ndominating_set(self):
        """
        Find a dominating set using a greedy algorithm based on node degrees.
        
        Returns:
            set: A dominating set of nodes determined by node degrees.
        """
        #node_degrees = dict(self.G.degree())  # Calculate node degrees
        #nsorted_indices = sorted(node_degrees, key=node_degrees.get, reverse=True)  # Sort nodes by degree

        self.ndominating_set = set()
        ncovered_nodes = set()

        for node in self.nsorted_indices:
            if len(ncovered_nodes) == len(self.graph.nodes):
                break
            if node not in ncovered_nodes:
                self.ndominating_set.add(node)
                ncovered_nodes.add(node)
                ncovered_nodes.update(self.graph.neighbors(node))

        return self.ndominating_set

    def greedy_wdominating_set(self):
        """
        Find a dominating set using a greedy algorithm based on node weights.
        
        Returns:
            set: A dominating set of nodes determined by node weights.
        """
        wsorted_indices = sorted(range(len(self.nodeweight)), key=lambda x: self.nodeweight[x], reverse=True)

        wdominating_set = set()
        wcovered_nodes = set()

        for node in wsorted_indices:
            if len(wcovered_nodes) == len(self.G.nodes):
                break
            if node not in wcovered_nodes:
                wdominating_set.add(node)
                wcovered_nodes.add(node)
                wcovered_nodes.update(self.G.neighbors(node))

        return wdominating_set

    def greedy_adominating_set(self):
        """
        Find a dominating set using a greedy algorithm based on average node weights.
        
        Returns:
            set: A dominating set of nodes determined by average node weights.
        """
        asorted_indices = sorted(range(len(self.wavg)), key=lambda x: self.wavg[x], reverse=True)

        adominating_set = set()
        acovered_nodes = set()

        for node in asorted_indices:
            if len(acovered_nodes) == len(self.G.nodes):
                break
            if node not in acovered_nodes:
                adominating_set.add(node)
                acovered_nodes.add(node)
                acovered_nodes.update(self.G.neighbors(node))

        return adominating_set


    def maximal_cliques(self):
        self.MaxCliques = []  # Initialize the list of maximal cliques
        for v in self.ndominating_set:
            self.neighbors = list(self.graph.neighbors(v))
            self.subgraph_nodes = self.neighbors + [v]
            self.subgraph = self.graph.subgraph(self.subgraph_nodes).copy()
            self.neighborcliques = list(nx.find_cliques(self.subgraph))
            self.cliques_sorted = sorted(self.neighborcliques, key=lambda clique: len(clique), reverse=True)
            uncovered_nodes = set(self.subgraph.nodes())

            while uncovered_nodes:
                for clique in self.cliques_sorted:
                    if uncovered_nodes & set(clique):
                        self.MaxCliques.append(sorted([int(node) for node in clique]))
                        uncovered_nodes.difference_update(clique)
                        break

        #self.bestcliques = [node for clique in self.MaxCliques for node in clique]
        
        return self.MaxCliques


    def _clique_to_Pauli_observable(self):
        """ Helper function that returns the sampled clique to a Pauli string (since qubit-wise commutativity is assumed).
            Performs a check whether this string actually commutes with all observables within the sampled clique.
            Returns a valid measurement setting as required for the parent class and the altered clique for further internal usage.
        """
        # the commutativity graph includes the identity term - we can simply drop it
        #clique = np.array(clique[1:]) - 1 if clique[0] == 0 else np.array(clique) - 1
        self.flattened_cliques = np.array([node for clique in self.MaxCliques for node in clique], dtype=int)-1
        self.clique_members = self.obs[self.flattened_cliques]
        setting = np.max(self.clique_members, axis=0)
        filtered = setting != 0
        self.clique_members[self.clique_members==0] = 4 # throw away identities
        # Now, np.min(clique_members,axis=0) has to match up with its np.max(...) except where setting == 0
        self.double_check = np.min(self.clique_members, axis=0)
        #print("clique_members:", clique_members)
        #print("setting:", setting)
        #print("double_check:", double_check)
        #print("Filtered indices:", np.where(filtered))
        #print("Values at filtered indices (setting):", setting[filtered])
        #print("Values at filtered indices (double_check):", double_check[filtered])

        assert np.allclose(setting[filtered],self.double_check[filtered]), "The clique {} does not allow for a qubit-wise commutativity-compatible measurement setting.".format(self.MaxCliques)
        return setting

    

class Best_scheme_given_pool(_AllocationMixin, Measurement_scheme):
    """
    Group-pool-agnostic measurement scheme for energy estimation.

    This class does not generate groups internally. Instead, it imports a pool
    of measurement settings from the `settings_dict` of another measurement
    scheme, interprets each key as a token encoding the sorted indices of Pauli
    observables measured by that setting, and then reallocates the measurement
    budget over that imported pool.

    The budget allocation is inherited from `_AllocationMixin`.

    Source-pool convention
    ----------------------
    The source scheme must store settings using

        token = encode_setting_token(setting_indices)

    where `setting_indices` is a sorted array of observable indices measured
    by that setting. This convention is assumed for QWC, FC, and kC alike.

    For k-commutativity, the source scheme must also provide

        source_scheme.fc_blocks_dict

    mapping each setting token to its kC block partition metadata. This metadata
    is copied into the new scheme.

    Important
    ---------
    Unlike `Sorted_Insertion_OGM` and `Greedy_Clique_Cover`, this class does not
    perform coverage repair by adding singleton groups. The imported pool is
    treated as fixed. If some positive-weight observable is not covered by the
    imported pool and truncation is not enabled, an error is raised.

    Consistency requirements
    ------------------------
    The source scheme and this scheme must agree on:

      - observable array and ordering;
      - number of observables;
      - number of qubits;
      - commutativity_type;
      - is_overlapping;
      - k, if commutativity_type == "kc".

    Allocation options such as total_rounds, allocation_objective,
    rounding_strategy, and md_gap_tol_rel do not need to match the source scheme.

    Allocation objectives
    ---------------------
    Budget allocation is inherited from `_AllocationMixin`, which supports:

      - allocation_objective = "variance":

            sum_j |alpha_j|^2 / N_j

      - allocation_objective = "bernstein_l1":

            sum_j |alpha_j| / sqrt(N_j)
    """

    def __init__(self,observables,weights,source_scheme,epsilon: float = 0.1,
                 total_rounds: int = 0,is_overlapping: bool = True,
                 commutativity_type: str = "qwc", *,
                 informed_allocation: bool = True,
                 allocation_objective: str = "bernstein_l1",
                 attempt_truncation: bool = False,
                 rounding_strategy: str = "largest_fraction",
                 md_gap_tol_rel: float = 1e-4,
                 prior_counts=None,
                 k: int | None = None):
        
        #super().__init__(observables, weights, epsilon, save_scheme=False)
        super().__init__(observables, weights, epsilon)
        self.is_overlapping = bool(is_overlapping)
        self.is_sampling = False

        self.commutativity_type = str(commutativity_type).lower()
        if self.commutativity_type not in ("qwc", "fc", "kc"):
            raise ValueError("commutativity_type must be 'qwc', 'fc', or 'kc'.")

        self.total_rounds = int(total_rounds)
        if self.total_rounds < 0:
            raise ValueError("total_rounds must be >= 0.")

        # ------------------------------------------------------------------
        # kC metadata
        # ------------------------------------------------------------------

        if self.commutativity_type == "kc":
            if k is None:
                if not hasattr(source_scheme, "k"):
                    raise ValueError(
                        "commutativity_type='kc' requires either k to be provided "
                        "or source_scheme to have a 'k' attribute."
                    )
                self.k = int(source_scheme.k)
            else:
                self.k = int(k)

            if not (1 <= self.k <= self.num_qubits):
                raise ValueError(
                    f"k must satisfy 1 <= k <= num_qubits={self.num_qubits}. "
                    f"Got k={self.k}."
                )

            self.fc_blocks_dict = {}

        else:
            if k is not None:
                raise ValueError(
                    "The argument k should only be provided when "
                    "commutativity_type='kc'."
                )

        # ------------------------------------------------------------------
        # Allocation options used by _AllocationMixin
        # ------------------------------------------------------------------

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
                "overlapping allocation, i.e. is_overlapping=True."
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

        # Diagnostics populated by _AllocationMixin.
        self.allocation_info = {
            "attempted": False,
            "mode": None}

        self.truncation_info = {
            "attempted": False,
            "selected": False}

        self.imported_pool_info = {
            "source_class": source_scheme.__class__.__name__,
            "num_imported_settings_raw": 0,
            "num_imported_settings_unique": 0,
            "num_covered_observables": 0,
            "num_uncovered_observables": 0,
            "num_uncovered_positive_weight_observables": 0,
        }

        self.cliques_pool: list[np.ndarray] = []
        self.source_scheme = source_scheme

        self.get_groups(source_scheme)

        self.allocate_budget()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, updated_total_rounds=None, clear_prior_counts=False):
        """
        Reset the newly allocated measurement scheme and re-run budget allocation
        over the already imported group pool.

        This does not re-import groups from the source scheme. To refresh the
        pool, call `get_groups(source_scheme)` explicitly before reallocating.
        """
        if updated_total_rounds is not None:
            self.total_rounds = int(updated_total_rounds)
            if self.total_rounds < 0:
                raise ValueError("total_rounds must be >= 0.")

        if clear_prior_counts:
            self.prior_counts[:] = 0

        self.allocate_budget()

    # ------------------------------------------------------------------
    # Source validation
    # ------------------------------------------------------------------

    def _validate_source_scheme(self, source_scheme):
        """
        Validate that the source scheme is compatible with this scheme.
        """
        if not hasattr(source_scheme, "settings_dict"):
            raise ValueError("source_scheme must have a settings_dict attribute.")

        if len(source_scheme.settings_dict) == 0:
            raise ValueError(
                "source_scheme.settings_dict is empty. There are no settings "
                "from which to build a group pool."
            )

        if not hasattr(source_scheme, "obs"):
            raise ValueError("source_scheme must have an obs attribute.")

        if not hasattr(source_scheme, "num_obs"):
            raise ValueError("source_scheme must have a num_obs attribute.")

        if not hasattr(source_scheme, "num_qubits"):
            raise ValueError("source_scheme must have a num_qubits attribute.")

        if int(source_scheme.num_obs) != int(self.num_obs):
            raise ValueError(
                "source_scheme.num_obs does not match this scheme. "
                f"source_scheme.num_obs={source_scheme.num_obs}, "
                f"self.num_obs={self.num_obs}."
            )

        if int(source_scheme.num_qubits) != int(self.num_qubits):
            raise ValueError(
                "source_scheme.num_qubits does not match this scheme. "
                f"source_scheme.num_qubits={source_scheme.num_qubits}, "
                f"self.num_qubits={self.num_qubits}."
            )

        source_obs = np.asarray(source_scheme.obs)
        target_obs = np.asarray(self.obs)

        if source_obs.shape != target_obs.shape:
            raise ValueError(
                "source_scheme.obs has a different shape from self.obs. "
                f"source shape={source_obs.shape}, target shape={target_obs.shape}."
            )

        if not np.array_equal(source_obs, target_obs):
            raise ValueError(
                "source_scheme.obs and self.obs differ. Best_scheme_given_pool "
                "requires the same observable array and ordering, because "
                "settings_dict tokens encode observable indices."
            )

        if not hasattr(source_scheme, "commutativity_type"):
            raise ValueError("source_scheme must have a commutativity_type attribute.")

        source_comm = str(source_scheme.commutativity_type).lower()

        if source_comm != self.commutativity_type:
            raise ValueError(
                "source_scheme.commutativity_type does not match this scheme. "
                f"source={source_comm!r}, target={self.commutativity_type!r}."
            )

        if not hasattr(source_scheme, "is_overlapping"):
            raise ValueError("source_scheme must have an is_overlapping attribute.")

        source_overlap = bool(source_scheme.is_overlapping)

        if source_overlap != self.is_overlapping:
            raise ValueError(
                "source_scheme.is_overlapping does not match this scheme. "
                f"source={source_overlap}, target={self.is_overlapping}. "
                "The allocation solver uses is_overlapping to choose between "
                "non-overlapping and overlapping allocation."
            )

        if self.commutativity_type == "kc":
            if not hasattr(source_scheme, "k"):
                raise ValueError(
                    "source_scheme must have a k attribute when commutativity_type='kc'."
                )

            if int(source_scheme.k) != int(self.k):
                raise ValueError(
                    "source_scheme.k does not match this scheme. "
                    f"source={int(source_scheme.k)}, target={int(self.k)}."
                )

            if not hasattr(source_scheme, "fc_blocks_dict"):
                raise ValueError(
                    "source_scheme must have an fc_blocks_dict attribute when "
                    "commutativity_type='kc'."
                )

    # ------------------------------------------------------------------
    # Imported group post-processing
    # ------------------------------------------------------------------

    def _postprocess_imported_groups(self, token_group_pairs):
        """
        Canonicalize and deduplicate imported groups.

        Unlike Sorted_Insertion_OGM and Greedy_Clique_Cover, this method does
        not perform coverage repair by adding singleton groups. The imported
        pool is treated as fixed.

        Parameters
        ----------
        token_group_pairs : list[tuple[bytes, np.ndarray]]
            Pairs of source token and decoded observable-index group.

        Returns
        -------
        groups : list[np.ndarray]
            Canonical unique groups.

        canonical_to_source_token : dict[bytes, bytes]
            Map from canonical token to the first source token that produced it.
        """
        groups = []
        seen = set()
        canonical_to_source_token = {}

        for source_token, group in token_group_pairs:
            arr = np.asarray(group, dtype=np.int32).ravel()

            if arr.size == 0:
                continue

            if np.any(arr < 0) or np.any(arr >= self.num_obs):
                raise ValueError(
                    "A source setting contains observable indices outside "
                    f"the valid range [0, {self.num_obs})."
                )

            arr = np.unique(arr)
            arr.sort()

            canonical_token = encode_setting_token(arr)

            if canonical_token in seen:
                # Keep the first occurrence. For kC, this also means keeping
                # the first block metadata if duplicate observable groups occur.
                continue

            seen.add(canonical_token)
            groups.append(arr)
            canonical_to_source_token[canonical_token] = source_token

        if len(groups) == 0:
            raise ValueError(
                "No usable groups were decoded from source_scheme.settings_dict."
            )

        return groups, canonical_to_source_token

    def _validate_nonoverlapping_groups(self, groups):
        """
        Validate that the imported group list is a true partition.

        Required when is_overlapping=False, because the non-overlapping
        allocation solver assumes each observable appears exactly once.
        """
        counts = np.zeros(self.num_obs, dtype=np.int32)

        for arr in groups:
            counts[arr] += 1

        missing = np.flatnonzero(counts == 0)
        repeated = np.flatnonzero(counts > 1)

        if missing.size > 0 or repeated.size > 0:
            msg_parts = [
                "Best_scheme_given_pool received groups that are not a "
                "non-overlapping partition despite is_overlapping=False."
            ]

            if missing.size > 0:
                msg_parts.append(
                    f"Missing observables: {missing[:20].tolist()}"
                    + (" ..." if missing.size > 20 else "")
                )

            if repeated.size > 0:
                msg_parts.append(
                    f"Repeated observables: {repeated[:20].tolist()}"
                    + (" ..." if repeated.size > 20 else "")
                )

            msg_parts.append(
                "Use is_overlapping=True or provide a source scheme whose "
                "settings form a true partition."
            )

            raise ValueError(" ".join(msg_parts))

    def _validate_coverage_for_allocation(self, groups):
        """
        Validate coverage of positive-weight observables.

        Because this class imports a fixed pool, it does not add coverage-repair
        singleton groups. If positive-weight observables are uncovered and have
        no prior samples, allocation without truncation would silently ignore or
        leave them impossible to measure. Therefore, this is only allowed when
        attempt_truncation=True.
        """
        covered = np.zeros(self.num_obs, dtype=bool)

        for arr in groups:
            covered[arr] = True

        absw = np.abs(np.asarray(self.w, dtype=np.float64)).reshape(-1)
        prior = np.asarray(self.prior_counts, dtype=np.float64).reshape(-1)

        positive_uncovered = (~covered) & (absw > 0.0)
        positive_uncovered_no_prior = positive_uncovered & (prior <= 0.0)

        missing_positive_no_prior = np.flatnonzero(positive_uncovered_no_prior)

        if missing_positive_no_prior.size > 0 and not self.attempt_truncation:
            raise ValueError(
                "The imported setting pool does not cover all positive-weight "
                "observables with zero prior counts, and attempt_truncation=False. "
                "Uncovered positive-weight observables would have no samples and "
                "would not be accounted for by a truncation penalty. "
                f"First missing indices: {missing_positive_no_prior[:20].tolist()}"
                + (" ..." if missing_positive_no_prior.size > 20 else "")
            )

        self.imported_pool_info.update(
            {
                "num_covered_observables": int(np.count_nonzero(covered)),
                "num_uncovered_observables": int(np.count_nonzero(~covered)),
                "num_uncovered_positive_weight_observables": int(
                    np.count_nonzero(positive_uncovered)
                ),
                "num_uncovered_positive_weight_zero_prior_observables": int(
                    missing_positive_no_prior.size
                ),
            }
        )

    def _copy_kc_metadata(self, source_scheme, canonical_to_source_token):
        """
        Copy kC block metadata from the source scheme.

        The copied dictionary is keyed by canonical observable-index tokens,
        which are the same type of keys produced later by _AllocationMixin when
        committing the selected allocation.
        """
        if self.commutativity_type != "kc":
            return

        self.fc_blocks_dict = {}

        for canonical_token, source_token in canonical_to_source_token.items():
            if source_token not in source_scheme.fc_blocks_dict:
                raise ValueError(
                    "Missing kC block metadata in source_scheme.fc_blocks_dict "
                    "for an imported setting token."
                )

            self.fc_blocks_dict[canonical_token] = source_scheme.fc_blocks_dict[source_token]

    # ------------------------------------------------------------------
    # Group import
    # ------------------------------------------------------------------

    def get_groups(self, source_scheme):
        """
        Import the setting pool from source_scheme.settings_dict.

        Each key in source_scheme.settings_dict is decoded as a sorted list of
        observable indices. The resulting groups are canonicalized and
        deduplicated, but no coverage-repair groups are added.
        """
        self._validate_source_scheme(source_scheme)

        token_group_pairs = []

        for token in source_scheme.settings_dict.keys():
            if not isinstance(token, (bytes, bytearray, memoryview)):
                raise TypeError(
                    "settings_dict keys must be byte-like tokens produced by "
                    "encode_setting_token."
                )

            token_bytes = bytes(token)
            group = decode_setting_token(token_bytes).astype(np.int32, copy=True)

            token_group_pairs.append((token_bytes, group))

        self.imported_pool_info["num_imported_settings_raw"] = int(
            len(token_group_pairs))

        groups, canonical_to_source_token = self._postprocess_imported_groups(
            token_group_pairs)

        self.imported_pool_info["num_imported_settings_unique"] = int(len(groups))

        if not self.is_overlapping:
            self._validate_nonoverlapping_groups(groups)

        self._validate_coverage_for_allocation(groups)

        if self.commutativity_type == "kc":
            self._copy_kc_metadata(source_scheme, canonical_to_source_token)

        self.cliques_pool = groups

        return groups


class Shadow_Grouping_STD(Measurement_scheme):
    """Grouping method based on QWC (qubit-wise commutativity).
    
       The next measurement setting p is found as follows: it is initialized 
       as the identity operator. Next, we obtain an ordering of the observables 
       in terms of their respective weight_function. For each observable o in 
       the ordered list of observables in descending order, it checks 
       qubit-wise commutativity (QWC). If so, the qubits in p that fall in the 
       support of o are overwritten by those in o. Eventually, the list is either 
       exhausted or p does not contain identity operators anymore. The function 
       weight_function takes in the weights,epsilon and the current number of 
       N_hits and is supposed to return an numpy-array of length len(w).
       Instead, weight_function can also be set to None (this is useful for 
       instances where the function is actually never called).
    
       Returns p.
    """

    def __init__(self, observables, weights, epsilon, weight_function, 
                 save_scheme=False, handle_ties=True, compute_N_hits_pairs=True):
        super().__init__(observables, weights, epsilon)
        self.weight_function = weight_function
        if self.weight_function is not None:
            test = self.weight_function(self.w, self.eps, self.N_hits)
            assert len(test) == len(self.w), (
                "Weight function is supposed to return an array of shape {} (i.e. number of observables) "
                "but returned an array of shape {}"
            ).format(self.w.shape, np.shape(test))
        self.is_sampling = False
        self.commutativity_type = 'qwc'
        self.handle_ties = handle_ties
        self.compute_N_hits_pairs = compute_N_hits_pairs
        self.save_scheme = save_scheme

    def reset(self):
        self.N_hits        = np.zeros_like(self.N_hits)
        self.N_hits_pairs  = np.zeros_like(self.N_hits_pairs)
        self.seen_settings = set()
        self._is_hit_rows_used = 0
        self.settings_dict = {}
        self._hit_outer_cache = {}
        if self.save_scheme:
            self.all_settings_list      = []
            self.num_diff_settings_list = []
            self.diff_settings_counter  = 0

    def find_setting(self,forced_idx=None):
        """Find the next measurement setting (QWC), but tokenize by covered observables.
        Parameters
        ----------
        forced_idx : int or None
            If provided, this observable is forced to be considered first in the
            greedy construction of the setting. This guarantees that it is included
            in the final setting (since the setting starts empty).
        """
        weights = self.weight_function(self.w, self.eps, self.N_hits)

        if self.handle_ties:
            M = len(weights)
            rounded   = np.round(weights, decimals=12)
            primary   = -rounded
            secondary = -np.arange(M, dtype=np.int64)
            order_desc = np.lexsort((secondary, primary))
        else:
            order_desc = np.argsort(weights)[::-1]
        
        # If forced_idx is provided, place it at top the ranking
        order_desc = self._promote_forced_idx(order_desc, forced_idx)

        # Build QWC basis setting greedily
        setting = np.zeros(self.num_qubits, dtype=int)
        for idx in order_desc:
            o = self.obs[idx]
            if hit_by_numba(o, setting):
                non_id = (o != 0)
                setting[non_id] = o[non_id]
                if np.min(setting) > 0:
                    break

        selected_mask = hit_by_batch_numba(self.obs, setting).astype(bool)

        # Tokenize by the set of compatible observable indices
        setting_indices = np.nonzero(selected_mask)[0].astype(np.int32)
        setting_indices.sort()
        token = encode_setting_token(setting_indices)

        self.N_hits += selected_mask

        if self.compute_N_hits_pairs:
            # Cache (or retrieve) the canonical index list for this token
            idx = self._append_is_hit_hit_outer(token, setting_indices)
            # Apply the outer update without building an outer product
            if idx.size:
                self.N_hits_pairs[np.ix_(idx, idx)] += 1

        # Unique is_hit row + bookkeeping
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

        info = {}

        return setting_indices , info


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

        super().__init__(observables, weights, epsilon, save_scheme=False)

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
