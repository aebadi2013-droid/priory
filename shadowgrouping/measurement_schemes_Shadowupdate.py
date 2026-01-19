import numpy as np
import networkx as nx
from itertools import product
from time import time
import numbers
from numba import njit
from shadowgrouping_v2.helper_functions import (
    setting_to_str, char_to_int, hit_by_numba, hit_by_batch_numba, sample_obs_batch_from_setting_numba, prepare_settings_for_numba, setting_to_obs_form, sample_obs_batch_from_setting_batch_numba)
from shadowgrouping_v2.guarantees import (get_epsilon_Chebyshev_scalar_tighter_numba, get_epsilon_Chebyshev_scalar_numba, get_epsilon_Hoeffding_scalar_tighter_numba, get_epsilon_Hoeffding_scalar_numba, get_epsilon_Bernstein_scalar, get_epsilon_Bernstein_scalar_no_restricted_validity, get_epsilon_Bernstein_scalar_tighter_no_restricted_validity, Guaranteed_accuracy)

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
        self.scheme_params = {"eps": epsilon, "num_obs": M}
        self.N_hits        = np.zeros(M,dtype=int)
        self.is_adaptive   = False # useful default to be given to any child class
        
        return
        
    def find_setting(self):
        pass
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
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
            return np.infty
        w_abs  = np.abs(self.w)
        w_abs /= np.sqrt(self.N_hits)
        norm   = np.sum(w_abs)
        w_abs /= np.sqrt(self.N_hits)
        norm2  = np.sum(w_abs)
        epsilon = norm * np.sqrt(N_delta(delta)) #equation 29 of the supplementary
        if epsilon > 2*norm*(1+2*norm/norm2):
            print("Warning! Epsilon out of validity range.")
        return epsilon

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
            return np.infty
    
        w_abs = np.abs(self.w[mask]) / np.sqrt(self.N_hits[mask])
        sigma = 2 * np.sum(w_abs)  # Eq. (25)
    
        w_abs = np.abs(self.w[mask]) / self.N_hits[mask]
        B = 2 * np.sum(w_abs)  # Eq. (23)
    
        epsilon = sigma * (1 + np.sqrt(-2 * np.log(delta))) - (2 * B * np.log(delta) / 3)
        return epsilon


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
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
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
        print(np.sum(inconf))
        return np.sum(inconf)
    
    def get_Bernstein_bound(self):
        if np.min(self.N_hits) == 0:
            bound = -1
        else:
            bound = np.exp(-0.25*(self.eps/2/np.sum(np.abs(self.w)/np.sqrt(self.N_hits))-1)**2)
        return bound            
        
    def find_setting(self,verbose=True):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        # sort observable list by respective weight
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits,dtype=int)

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
        
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info

class Shadow_Grouping_Update(Measurement_scheme):
    """ Based on Bruno Proposal
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.selected_cliques = []  # Stores best cliques from each round
        self.clique_cache = {}  # maps observable tuple -> hit cliques
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        self.selected_cliques = []
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
        # sort observable list by respective weight
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits,dtype=int)
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
        #print("N_hits before update:", self.N_hits)
        tstart = time()
        if verbose:
            print("Checking list of observables.")
        
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
            if verbose:
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
            hit_cliques = find_cliques3(hit_graph, center_node)
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
        
        cliques_with_SC = []
        for clique in hit_cliques:
            sc = 0
            sw = 0

        # ---- Step 1: compute SC over the clique members ----
            for o in clique:
                o_arr = np.array(o)
                idx = np.where((self.obs == o_arr).all(axis=1))[0]
                if len(idx) > 0:
                    i = idx[0]
                    nhit = self.N_hits[i] 
                    sc += weights[i] / (nhit+1)

        # ---- Step 2: simulate building a setting from this clique ---- if self.N_hits[i] > 0 else 1e-6
            setting = np.zeros(self.num_qubits, dtype=int)
            for o in clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting):
                    non_id = o_arr != 0
                    setting[non_id] = o_arr[non_id]

        # ---- Step 3: check all observables against this setting ----
            for i, o in enumerate(self.obs):
                if hit_by(o, setting):
                    sw += weights[i] / (self.N_hits[i]+1)

        # ---- Store both scores along with the clique ----
        cliques_with_SC.append(((sc**2 + sw**2), clique))

        if not cliques_with_SC:
            raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")

        # Sort cliques by SC
        cliques_with_SC.sort(key=lambda x: x[0])
        _, best_clique = cliques_with_SC[0]
        self.selected_cliques.append(best_clique)
        
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


        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        tend = time()
        delta = 0.02
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["epsilon_Bernstein"] = self.get_epsilon_Bernstein(delta)
        info["run_time"] = tend - tstart
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        #print("info is",info)
        return setting, info



class Shadow_Grouping_Update2(Measurement_scheme):
    """ Based on overall weight of each clique
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.selected_cliques = []  # Stores best cliques from each round
        self.clique_cache = {}  # maps observable tuple -> hit cliques
        self.shadow_was_best_count = 0
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
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
        # sort observable list by respective weight
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits,dtype=int)
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
        if verbose:
            print("Checking list of observables.")
        tstart = time()

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

        # Get highest-weight observable
        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
        center_node = tuple(first_obs)  # Use tuple as dictionary key
        

        # Check cache
        if center_node in self.clique_cache:
            hit_cliques = self.clique_cache[center_node]
            #if verbose:
            #print(f"Using cached cliques for {center_node}")
        else:
            # Create setting from first_obs
            # Now check who is hit-by this setting
            non_id = first_obs != 0
            globalsetting[non_id] = first_obs[non_id]  # make it the setting
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
            #Now build the graph from hit list
            hit_graph = build_hit_graph(hit_list)
            print("hit_graph has", hit_graph.number_of_nodes(), "nodes and", hit_graph.number_of_edges(), "edges.")
            #find the cliques around the center node
            center_node = tuple(first_obs)  # This is the observable around which we’re building
            hit_cliques = find_cliques5(hit_graph)
            print("hit_cliques found:", len(hit_cliques))
            # Cache result
            self.clique_cache[center_node] = hit_cliques
        
        if not hit_cliques:
            hit_cliques=[[center_node]]
            #print("hit_cliques found:", len(hit_cliques))
            #for i, clique in enumerate(hit_cliques):
                #print(f"Clique {i+1}: {[list(node) for node in clique]}")

        hit_cliques.append(shadowclique)

        # Compute SC for each clique
        # Compute SC and SW for each clique
        cliques_with_SC = []
        for clique in hit_cliques:
            sw = 0
            setting = np.zeros(self.num_qubits, dtype=int)
            for o in clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting):
                    non_id = o_arr != 0
                    setting[non_id] = o_arr[non_id]
                if np.min(setting) != 0:
                    break
                if np.min(setting) == 0:
                    for idx in reversed(order):
                        o = self.obs[idx]
                        if hit_by(o,setting):
                            non_id = o!=0
                            setting[non_id] = o[non_id]
                        if np.min(setting) > 0:
                            break
            # ---- Step 3: check all observables against this setting ----
            for i, o in enumerate(self.obs):
                if hit_by(o, setting): #CHANGEHIT
                    sw += abs(weights[i])
            if len(clique) == len(shadowclique) and all(np.array_equal(a, b) for a, b in zip(clique, shadowclique)):
                shadowclique_weight = sw
            # ---- Store both scores along with the clique ----
            cliques_with_SC.append((sw, clique))

        if not cliques_with_SC:
            raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")

        # Sort cliques by SC
        cliques_with_SC.sort(key=lambda x: x[0],reverse=True)
        best_weight, best_clique = cliques_with_SC[0]
        self.selected_cliques.append(best_clique)

        if len(best_clique) == len(shadowclique) and all(np.array_equal(np.array(o1), np.array(o2)) for o1, o2 in zip(best_clique, shadowclique)):
            self.shadow_was_best_count += 1
            print("Shadow clique was selected", self.shadow_was_best_count, "times")
        else:
            print("best cliques has weight", best_weight,"while the shadow clique has weight", shadowclique_weight)
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
            print("setting is not complete")
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
                    print("but i completed it")
                    break
  

        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool) #CHANGEHIT
        self.N_hits += is_hit
        tend = time()
        delta = 0.02
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["epsilon_Bernstein"] = self.get_epsilon_Bernstein(delta)
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        #print("update 2 info is",info)
        return setting, info


class Shadow_Grouping_Update3(Measurement_scheme):
    """ First find all cliques using DomClique then select the heaviest in each round
    """
    
    def __init__(self, observables, weights, epsilon, weight_function):
        # Convert Pauli strings to arrays FIRST
        #observablesarray = [pauli_string_to_array(o) for o in observables]
    
        # Then pass converted observables into super().__init__()
        #super().__init__(observablesarray, weights, epsilon)
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.selected_cliques = []  # Stores best cliques from each round
        self._cached_graph = build_hit_graph(observables)
        #self._cached_cliques = list(nx.find_cliques(self._cached_graph))
        self._cached_cliques = list(DomClique1(self._cached_graph))
        
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

    #def build_graph_and_cliques(self):
        #weights = self.weight_function(self.w, self.eps, self.N_hits)
        #order = np.argsort(-weights)
        #sorted_obs = [self.obs[i] for i in order]
        # Build and cache the graph
        #self._cached_graph = build_hit_graph(sorted_obs)
        # Find and cache the cliques
        #self._cached_cliques = find_cliques2(self._cached_graph)
        #self._cached_cliques = nx.enumerate_all_cliques(self._cached_graph)
    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        if self._cached_graph is None or self._cached_cliques is None:
            self.build_graph_and_cliques()
        
        # sort observable list by respective weight
        weights = self.weight_function(self.w, self.eps, self.N_hits)
        tstart = time()
        # Compute SC for each clique
        # Compute SC and SW for each clique
        cliques_with_weights = []
        for clique in self._cached_cliques:
            sw = 0
            for o in clique:
                o_arr = np.array(o)
                # Find the index of the observable in self.obs
                idx = np.where((self.obs == o_arr).all(axis=1))[0]
                if len(idx) > 0:
                    sw += abs(weights[idx[0]])
            cliques_with_weights.append((sw, clique))

        if verbose:
            print("the cliques found are:",cliques_with_weights)
        if not cliques_with_weights:
            raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")

        # Sort cliques by SC
        cliques_with_weights.sort(key=lambda x: x[0],reverse=True)
        _, best_clique = cliques_with_weights[0]
        self.selected_cliques.append(best_clique)
        if verbose:
            print("best clique is:",best_clique)
        #setting construction
        setting = np.zeros(self.num_qubits, dtype=int)
        for o in best_clique:
            o = np.array(o)
            if verbose:
                print("Checking", o)
                print("current setting before update:",setting)
            if hit_by(o, setting):
                non_id = o != 0
                #print("non_id:",non_id)
                #print("o[non_id]:",o[non_id])
                setting[non_id] = o[non_id]
                if verbose:
                    print("p =", setting)
                if np.min(setting) > 0:
                    break

        tend = time()
        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        if not np.any(self.N_hits) == 0:
            delta = 0.02
            info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
            print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info


class Shadow_Grouping_Update4(Measurement_scheme):
    """ First find all cliques using Recursive Dominating Set then select the heaviest in each round
    """
    
    def __init__(self, observables, weights, epsilon, weight_function):
        # Convert Pauli strings to arrays FIRST
        #observablesarray = [pauli_string_to_array(o) for o in observables]
    
        # Then pass converted observables into super().__init__()
        #super().__init__(observablesarray, weights, epsilon)
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.selected_cliques = []  # Stores best cliques from each round
        self._cached_graph = build_hit_graph(observables)
        #self._cached_cliques = list(nx.find_cliques(self._cached_graph))
        self._cached_cliques = list(find_cliques3(self._cached_graph))
        self._cached_completesetting = []
        self._cached_settinghit = []
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

    #def build_graph_and_cliques(self):
        #weights = self.weight_function(self.w, self.eps, self.N_hits)
        #order = np.argsort(-weights)
        #sorted_obs = [self.obs[i] for i in order]
        # Build and cache the graph
        #self._cached_graph = build_hit_graph(sorted_obs)
        # Find and cache the cliques
        #self._cached_cliques = find_cliques2(self._cached_graph)
        #self._cached_cliques = nx.enumerate_all_cliques(self._cached_graph)
    
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        if self._cached_graph is None or self._cached_cliques is None:
            self.build_graph_and_cliques()
        
        # sort observable list by respective weight
        weights = self.weight_function(self.w, self.eps, self.N_hits)
        tstart = time()
        # Compute SC for each clique
        # Compute SC and SW for each clique
        cliques_with_weights = []
        for clique in self._cached_cliques:
            sw = 0
            for o in clique:
                o_arr = np.array(o)
                # Find the index of the observable in self.obs
                idx = np.where((self.obs == o_arr).all(axis=1))[0]
                if len(idx) > 0:
                    sw += abs(weights[idx[0]])
            cliques_with_weights.append((sw, clique))

        if verbose:
            print("the cliques found are:",cliques_with_weights)
        if not cliques_with_weights:
            raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")

        # Sort cliques by SC
        cliques_with_weights.sort(key=lambda x: x[0],reverse=True)
        _, best_clique = cliques_with_weights[0]
        self.selected_cliques.append(best_clique)
        if verbose:
            print("best clique is:",best_clique)
        #setting construction
        setting = np.zeros(self.num_qubits, dtype=int)
        for o in best_clique:
            o = np.array(o)
            if verbose:
                print("Checking", o)
                print("current setting before update:",setting)
            if hit_by(o, setting):
                non_id = o != 0
                #print("non_id:",non_id)
                #print("o[non_id]:",o[non_id])
                setting[non_id] = o[non_id]
                if verbose:
                    print("p =", setting)
                if np.min(setting) > 0:
                    break

        tend = time()
        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info


class Shadow_Grouping_Update5(Measurement_scheme):
    """ Based on overall weight of each clique-clique found by newest dominating set
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.selected_cliques = []  # Stores best cliques from each round
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return
    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
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
        # sort observable list by respective weight
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        setting = np.zeros(self.num_qubits,dtype=int)
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        
        if verbose:
            print("Checking list of observables.")
        tstart = time()


        # Get highest-weight observable
        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
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
                    
        


        # Now create the weight_map for each observable
        weight_map = {tuple(self.obs[i]): weights[i] for i in range(len(self.obs))}
        #print("hit_list has", len(hit_list), "observables.")

        # Build the hit graph and pass the weight_map
        hit_graph = build_hit_graph2(hit_list, weight_map=weight_map)
        #print("hit_graph has", hit_graph.number_of_nodes(), "nodes and", hit_graph.number_of_edges(), "edges.")
        
        #find the cliques around the center node
        center_node = tuple(first_obs)  # This is the observable around which we’re building
        hit_cliques = find_cliques4(hit_graph, max_depth=10, weight_map=weight_map)

        #print("hit_cliques found:", len(hit_cliques))
        #for i, clique in enumerate(hit_cliques):
            #print(f"Clique {i+1}: {[list(node) for node in clique]}")


        # Compute SC for each clique
        # Compute SC and SW for each clique
        cliques_with_SC = []
        for clique in hit_cliques:
            #sc = 0
            sw = 0

        # ---- Step 1: compute SC over the clique members ----
            #for o in clique:
                #o_arr = np.array(o)
                #idx = np.where((self.obs == o_arr).all(axis=1))[0]
                #if len(idx) > 0:
                    #i = idx[0]
                    #nhit = self.N_hits[i] if self.N_hits[i] > 0 else 1e-6
                    #sc += abs(weights[i])

        # ---- Step 2: simulate building a setting from this clique ----
            setting = np.zeros(self.num_qubits, dtype=int)
            for o in clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting):
                    non_id = o_arr != 0
                    setting[non_id] = o_arr[non_id]

        # ---- Step 3: check all observables against this setting ----
            for i, o in enumerate(self.obs):
                if hit_by(o, setting):
                    sw += abs(weights[i])

        # ---- Store both scores along with the clique ----
        cliques_with_SC.append((sw, clique))

        if not cliques_with_SC:
            raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")

        # Sort cliques by SC
        cliques_with_SC.sort(key=lambda x: x[0],reverse=True)
        _, best_clique = cliques_with_SC[0]
        self.selected_cliques.append(best_clique)
        
        #setting construction
        setting = np.zeros(self.num_qubits, dtype=int)
        for o in best_clique:
            if verbose:
                print("Checking", o)
            if hit_by(o, setting):
                non_id = o != 0
                setting[non_id] = o[non_id]
                if verbose:
                    print("p =", setting)
                if np.min(setting) > 0:
                    break

        tend = time()
        # update number of hits
        is_hit = np.array([hit_by(o,setting) for o in self.obs],dtype=bool)
        self.N_hits += is_hit
        
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info


class Priori(Measurement_scheme):
    """ First find all cliques using DomClique then select the heaviest in each round
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
        return

    
    def reset(self):
        self.N_hits = np.zeros_like(self.N_hits)
        #self.settings_dict = {} 2222222222222
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
            print("length of cliques with epsilon is", len(self.cliques_with_epsilon))
            _, best_clique = self.cliques_with_epsilon[0]
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
            else:
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


class Posteriori(Measurement_scheme):
    """ 
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        #self.settings_dict = {} 2222222222222
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
        #self.settings_dict = {} 22222222222
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


class Shadow_Grouping_Update8(Measurement_scheme):
    """ 
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
        self.processed_center_node = []
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

        # Get highest-weight observable
        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
        center_node = tuple(first_obs)  # Use tuple as dictionary key
        print("center node is", first_idx, "and its weight is", weights[first_idx])

        tstart = time()
        
        
        
        #print("shadow regime ended")
        #print(self.N_hits)
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
        #shadowclique = []
        #for o in self.obs:
        #    if hit_by(o, shadowcliquesetting):
        #        shadowclique.append(o)
        if verbose:
            print("Checking list of observables.")
            
        cliques_with_epsilon = []
        delta = 0.02
        removedcliques = 0
        completecliques = 0
        valid_settings = []
        valid_cliques = []
        if center_node not in self.processed_center_node:
            #non_id = first_obs != 0
            #globalsetting[non_id] = first_obs[non_id]  # make it the setting
            # Now check who is hit-by this setting
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
            #print("hit_graph has", hit_graph.number_of_nodes(), "nodes and", hit_graph.number_of_edges(), "edges.")
            if not hit_graph:
                raise RuntimeError("No hit graph found.")
            #find the cliques around the center node
            #center_node = tuple(first_obs)  # This is the observable around which we’re building
            #if not center_node:
            #    raise RuntimeError("No center node found.")
            hit_cliques = find_cliques5(hit_graph)
            #print("we fount these cliques",hit_cliques)
            # Cache result
            self.clique_cache[center_node] = hit_cliques
            #hit_cliques.append(shadowclique)
            if not hit_cliques:
                hit_cliques=[[center_node]]
            # Compute SC for each clique
            # Compute SC and SW for each clique
            #print("length of hit_cliques is",len(hit_cliques))
            for clique in hit_cliques:
                if center_node not in self.clean_setting_cache:
                    self.clean_setting_cache[center_node] = []
                    self.clean_clique_cache[center_node] = []
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
                    #removedcliques += 1
                    #print("Removed incomplete clique", removedcliques , "times")
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
                            break
                    completecliques += 1
                    self.clean_clique_cache[center_node].append(clique)
                    if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache[center_node]):
                        self.clean_setting_cache[center_node].append(setting_candidate)
                    #self.clean_setting_cache[center_node].append(setting_candidate)
                    #valid_cliques.append(clique)
                    #valid_settings.append(setting_candidate)
                    is_hit_candidate = []
                    is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                    #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                    #self.is_hit_clique_cache[key] = is_hit_candidate
                    #is_hit_candidate = np.array([hit_by(o,setting_candidate) for o in self.obs],dtype=bool) #CHANGEHIT
                    self.N_hits += is_hit_candidate
                    #print("complete cliques are", completecliques)
                    # ---- Store both scores along with the clique ----
                    #cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v3(delta), setting_candidate))
                    #cliques_with_epsilon.append((self.get_Bernstein_bound(), setting_candidate))
                    #cliques_with_epsilon.append((self.get_Bernstein_bound(), clique))
                    #cliques_with_epsilon.append((self.get_inconfidence_bound(), clique))
                    self.N_hits -= is_hit_candidate
                else:
                    completecliques += 1
                    self.clean_clique_cache[center_node].append(clique)
                    if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache[center_node]):
                        self.clean_setting_cache[center_node].append(setting_candidate)
                    #self.clean_setting_cache[center_node].append(setting_candidate)
                    #valid_cliques.append(clique)
                    #valid_settings.append(setting_candidate)
                    is_hit_candidate = []
                    is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                    #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                    #self.is_hit_clique_cache[key] = is_hit_candidate
                    #is_hit_candidate = np.array([hit_by(o,setting_candidate) for o in self.obs],dtype=bool) #CHANGEHIT
                    self.N_hits += is_hit_candidate
                    #print("complete cliques are", completecliques)
                    # ---- Store both scores along with the clique ----
                    cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v3(delta), setting_candidate)) 
                    #cliques_with_epsilon.append((self.get_Bernstein_bound(), setting_candidate))
                    #cliques_with_epsilon.append((self.get_Bernstein_bound(), clique))
                    #cliques_with_epsilon.append((self.get_inconfidence_bound(), clique))
                    self.N_hits -= is_hit_candidate
        else:
            #print("I already met this candidate")
            for setting_candidate in self.clean_setting_cache[center_node]:
                #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                #is_hit_candidate = self.is_hit_clique_cache[key]
                is_hit_candidate = []
                is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                self.N_hits += is_hit_candidate
                cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), setting_candidate))
                #cliques_with_epsilon.append((self.get_Bernstein_bound(), setting_candidate))
                self.N_hits -= is_hit_candidate
        is_hit_candidate = []
        if not any((existing == shadowcliquesetting).all() for existing in self.clean_setting_cache[center_node]):
                self.clean_setting_cache[center_node].append(shadowcliquesetting)
        #self.clean_setting_cache[center_node].append(shadowcliquesetting)
        is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
        self.N_hits += is_hit_candidate
        cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity_v2(delta), shadowcliquesetting))
        #cliques_with_epsilon.append((self.get_Bernstein_bound(), shadowcliquesetting))
        print("epsilon for shadow clique is",self.get_epsilon_Bernstein_no_restricted_validity_v2(delta))
        self.N_hits -= is_hit_candidate
        #if valid_cliques:
        #    self.clean_clique_cache[center_node] = valid_cliques
        #    self.clean_setting_cache[center_node] = valid_settings
        #print("length of clean setting cache", len(self.clean_setting_cache[center_node]))
        if not cliques_with_epsilon:
            raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
        print("length of cliques with epsilon is", len(cliques_with_epsilon))
        
        # Sort cliques by epsilon
        cliques_with_epsilon.sort(key=lambda x: x[0])
        _, best_clique = cliques_with_epsilon[0]
        #print("epsilon for best clique is",cliques_with_epsilon[0])
        self.selected_cliques.append(best_clique)
        #if len(best_clique) == len(shadowclique) and all(np.array_equal(np.array(o1), np.array(o2)) for o1, o2 in zip(best_clique, shadowclique)):
         #   self.shadow_was_best_count += 1
         #   print("Shadow clique was selected", self.shadow_was_best_count, "times")
        if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
            self.shadow_was_best_count += 1
            print("Shadow clique was selected", self.shadow_was_best_count, "times")
        else:
            print("epsilon for best clique is",cliques_with_epsilon[0])
        #print("best clique has",len(best_clique),"members")
        setting = best_clique
        
        
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
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info



class Shadow_Grouping_Update9(Measurement_scheme):
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
        shadowcliquesetting = np.zeros(self.num_qubits,dtype=int)
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        if verbose:
            print("Checking list of observables.")
        
        tstart = time()

        
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
            #if verbose:
            print(f"Using cached cliques for {center_node}")
        else:
            # Create setting from first_obs
            #non_id = first_obs != 0
            #globalsetting[non_id] = first_obs[non_id]  # make it the setting
            # Now check who is hit-by this setting
            hit_list = []
            for o in self.obs:
                if hit_by(o, first_obs):
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
            print("hit_graph has", hit_graph.number_of_nodes(), "nodes and", hit_graph.number_of_edges(), "edges.")
            if not hit_graph:
                raise RuntimeError("No hit graph found.")
        
            #find the cliques around the center node
            center_node = tuple(first_obs)  # This is the observable around which we’re building
            if not center_node:
                raise RuntimeError("No center node found.")
            hit_cliques = find_cliques3(hit_graph, center_node)
            #print(len(hit_cliques),"number of cliques were found")
                
            #print("we fount these cliques",hit_cliques)
            # Cache result
            self.clique_cache[center_node] = hit_cliques
             
            
        hit_cliques.append(shadowclique)
        #print("shadow clique added:",shadowclique)
        print("hit_cliques found:", len(hit_cliques))
        #for i, clique in enumerate(hit_cliques):
            #print(f"Clique {i+1}: {[list(node) for node in clique]}")
        #if not hit_cliques:
            #raise RuntimeError("No cliques found.")
        if not hit_cliques:
            hit_cliques=[[center_node]]
        # Compute SC for each clique
        # Compute SC and SW for each clique
        
        cliques_with_weights = []
        delta = 0.02
        for clique in hit_cliques:
        # ---- Step 2: simulate building a setting from this clique ---- if self.N_hits[i] > 0 else 1e-6
            sw = 0
            setting_candidate = np.zeros(self.num_qubits, dtype=int)
            for o in clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting_candidate):
                    non_id = o_arr != 0
                    setting_candidate[non_id] = o_arr[non_id]
                if np.min(setting_candidate) > 0:
                    break

            if np.min(setting_candidate) == 0:
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
                        break
            #if np.min(setting_candidate) != 0:
                #print("setting is now complete")
            # ---- Step 3: find all observables compatible with this setting_candidate ----
            #is_hit_candidate = np.array([sample_obs_from_setting(o,setting_candidate) for o in self.obs],dtype=bool) #CHANGEHIT
            # ---- Store both scores along with the clique ----
             
            #cliques_with_epsilon.append((self.get_epsilon_Bernstein(delta), clique))
            #cliques_with_epsilon.append((self.get_Bernstein_bound(), clique))
            #cliques_with_epsilon.append((self.get_inconfidence_bound(), clique))

        # ---- Step 3: check all observables against this setting ----
            for i, o in enumerate(self.obs):
                if sample_obs_from_setting(o, setting_candidate): #CHANGEHIT
                    sw += abs(weights[i])
            
            cliques_with_weights.append((sw, clique))
        
        if not cliques_with_weights:
            raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")

        # Sort cliques by epsilon
        cliques_with_weights.sort(key=lambda x: x[0])
        best_weight, best_clique = cliques_with_weights[0]
        self.selected_cliques.append(best_clique)

        if len(best_clique) == len(shadowclique) and all(np.array_equal(np.array(o1), np.array(o2)) for o1, o2 in zip(best_clique, shadowclique)):
            self.shadow_was_best_count += 1
            print("Shadow clique was selected", self.shadow_was_best_count, "times")

        
        print("best clique has",len(best_clique),"members with total weight", best_weight)

        
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
            print("setting is not complete")
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
        is_hit = np.array([sample_obs_from_setting(o,setting) for o in self.obs],dtype=bool) #CHANGEHIT
        self.N_hits += is_hit
        
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
        #print("epsilon_Bernstein_no_restricted_validity_v2:", info["epsilon_Bernstein_no_restricted_validity_v2"])
        if not np.any(self.N_hits) == 0:
            delta = 0.02
            info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
            print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info



class Shadow_Grouping_Update10(Measurement_scheme):
    """ 
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
        self.processed_center_node = []
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

        # Get highest-weight observable
        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
        center_node = tuple(first_obs)  # Use tuple as dictionary key
        print("center node is", first_idx)
        

        tstart = time()
        
        if np.any(self.N_hits == 0):
            #print("still in shadow regime")
            if center_node not in self.processed_center_node:
                self.processed_center_node.append(center_node)
                self.clean_setting_cache[center_node] = []
            for idx in reversed(order):
                o = self.obs[idx]
                if verbose:
                    print("Checking",o)
                if hit_by(o,setting):
                    non_id = o!=0
                    # overwrite those qubits that fall in the support of o
                    setting[non_id] = o[non_id]
                    # break sequence is case all identities in setting are exhausted
                    if np.min(setting) > 0:
                        print("p =",setting)
                        self.clean_setting_cache[center_node].append(setting)
                        break
        else:
            #print("shadow regime ended")
            #print(self.N_hits)
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
            #shadowclique = []
            #for o in self.obs:
            #    if hit_by(o, shadowcliquesetting):
            #        shadowclique.append(o)
            if verbose:
                print("Checking list of observables.")
                
            cliques_with_epsilon = []
            delta = 0.02
            removedcliques = 0
            completecliques = 0
            valid_settings = []
            valid_cliques = []
            if center_node not in self.processed_center_node:
                setting = shadowcliquesetting
                self.clean_setting_cache[center_node] = []
                self.processed_center_node.append(center_node)
                self.clean_setting_cache[center_node].append(setting)
            else:
                #print("I already met this candidate")
                if not any((existing == shadowcliquesetting).all() for existing in self.clean_setting_cache[center_node]):
                    self.clean_setting_cache[center_node].append(shadowcliquesetting)
                #self.clean_setting_cache[center_node].append(shadowcliquesetting)
                for cached_settings in self.clean_setting_cache.values():
                    for setting_candidate in cached_settings:
                        #for setting_candidate in self.clean_setting_cache[center_node]:
                        #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                        #is_hit_candidate = self.is_hit_clique_cache[key]
                        is_hit_candidate = []
                        is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                        self.N_hits += is_hit_candidate
                        cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), setting_candidate))
                        #cliques_with_epsilon.append((self.get_Bernstein_bound(), setting_candidate))
                        self.N_hits -= is_hit_candidate
                
                #for setting_candidate in self.clean_setting_cache[center_node]:
                    #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                    #is_hit_candidate = self.is_hit_clique_cache[key]
                 #   is_hit_candidate = []
                  #  is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                   # self.N_hits += is_hit_candidate
                    #cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), setting_candidate))
                    #self.N_hits -= is_hit_candidate
                # is_hit_candidate = []
                # is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
                # self.N_hits += is_hit_candidate
                # cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), shadowcliquesetting))
                # print("epsilon for shadow clique is",self.get_epsilon_Bernstein_no_restricted_validity(delta))
                # self.N_hits -= is_hit_candidate
                #if valid_cliques:
                #    self.clean_clique_cache[center_node] = valid_cliques
                #    self.clean_setting_cache[center_node] = valid_settings
                #print("length of clean setting cache", len(self.clean_setting_cache[center_node]))
                if not cliques_with_epsilon:
                    raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
                print("length of cliques with epsilon is", len(cliques_with_epsilon))
            
                # Sort cliques by epsilon
                cliques_with_epsilon.sort(key=lambda x: x[0])
                _, best_clique = cliques_with_epsilon[0]
                #print("epsilon for best clique is",cliques_with_epsilon[0])
                self.selected_cliques.append(best_clique)
                #if len(best_clique) == len(shadowclique) and all(np.array_equal(np.array(o1), np.array(o2)) for o1, o2 in zip(best_clique, shadowclique)):
                 #   self.shadow_was_best_count += 1
                 #   print("Shadow clique was selected", self.shadow_was_best_count, "times")
                if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                    self.shadow_was_best_count += 1
                    print("Shadow clique was selected", self.shadow_was_best_count, "times")
                else:
                    print("epsilon for best clique is",cliques_with_epsilon[0])
                #print("best clique has",len(best_clique),"members")
                setting = best_clique
                #self.clean_setting_cache[center_node].append(shadowcliquesetting)
            
        
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
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info

class Shadow_Grouping_Update11(Measurement_scheme):
    """ 
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.shadow_was_best_count = 0
        self.clique_cache = {}  # maps observable tuple -> hit cliques
        self.clean_clique_cache = {}  
        self.clean_setting_cache = []
        self.is_hit_clique_cache = {}
        self.processed_center_node = []
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

        # Get highest-weight observable
        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
        center_node = tuple(first_obs)  # Use tuple as dictionary key
        print("center node is", first_idx)

        tstart = time()
        
        if np.any(self.N_hits == 0):
            #print("still in shadow regime")
            for idx in reversed(order):
                o = self.obs[idx]
                if verbose:
                    print("Checking",o)
                if hit_by(o,setting):
                    non_id = o!=0
                    # overwrite those qubits that fall in the support of o
                    setting[non_id] = o[non_id]
                    # break sequence is case all identities in setting are exhausted
                    if np.min(setting) > 0:
                        if center_node not in self.clean_clique_cache:
                            self.clean_clique_cache[center_node] = []
                        if not any((existing == setting).all() for existing in self.clean_setting_cache):
                            self.clean_setting_cache.append(setting)
                        print("p =",setting)
                        break
        else:
            #print("shadow regime ended")
            #print(self.N_hits)
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
            #shadowclique = []
            #for o in self.obs:
            #    if hit_by(o, shadowcliquesetting):
            #        shadowclique.append(o)
            if verbose:
                print("Checking list of observables.")
                
            cliques_with_epsilon = []
            delta = 0.02
            removedcliques = 0
            completecliques = 0
            valid_settings = []
            valid_cliques = []
            if center_node not in self.processed_center_node:
                #non_id = first_obs != 0
                #globalsetting[non_id] = first_obs[non_id]  # make it the setting
                # Now check who is hit-by this setting
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
                #print("hit_graph has", hit_graph.number_of_nodes(), "nodes and", hit_graph.number_of_edges(), "edges.")
                if not hit_graph:
                    raise RuntimeError("No hit graph found.")
                #find the cliques around the center node
                #center_node = tuple(first_obs)  # This is the observable around which we’re building
                #if not center_node:
                #    raise RuntimeError("No center node found.")
                hit_cliques = find_cliques5(hit_graph)
                #print("we fount these cliques",hit_cliques)
                # Cache result
                self.clique_cache[center_node] = hit_cliques
                #hit_cliques.append(shadowclique)
                if not hit_cliques:
                    hit_cliques=[[center_node]]
                # Compute SC for each clique
                # Compute SC and SW for each clique
                #print("length of hit_cliques is",len(hit_cliques))
                for clique in hit_cliques:
                    if center_node not in self.clean_clique_cache:
                        self.clean_clique_cache[center_node] = []
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
                        #removedcliques += 1
                        #print("Removed incomplete clique", removedcliques , "times")
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
                                break
                        completecliques += 1
                        self.clean_clique_cache[center_node].append(clique)
                        if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache):
                            self.clean_setting_cache.append(setting_candidate)
                        #self.clean_setting_cache[center_node].append(setting_candidate)
                        #valid_cliques.append(clique)
                        #valid_settings.append(setting_candidate)
                        is_hit_candidate = []
                        is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                        #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                        #self.is_hit_clique_cache[key] = is_hit_candidate
                        #is_hit_candidate = np.array([hit_by(o,setting_candidate) for o in self.obs],dtype=bool) #CHANGEHIT
                        self.N_hits += is_hit_candidate
                        #print("complete cliques are", completecliques)
                        # ---- Store both scores along with the clique ----
                        cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), setting_candidate)) 
                        #cliques_with_epsilon.append((self.get_epsilon_Bernstein(delta), clique))
                        #cliques_with_epsilon.append((self.get_Bernstein_bound(), clique))
                        #cliques_with_epsilon.append((self.get_inconfidence_bound(), clique))
                        self.N_hits -= is_hit_candidate
                    else:
                        completecliques += 1
                        self.clean_clique_cache[center_node].append(clique)
                        if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache):
                            self.clean_setting_cache.append(setting_candidate)
                        #self.clean_setting_cache[center_node].append(setting_candidate)
                        #valid_cliques.append(clique)
                        #valid_settings.append(setting_candidate)
                        is_hit_candidate = []
                        is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                        #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                        #self.is_hit_clique_cache[key] = is_hit_candidate
                        #is_hit_candidate = np.array([hit_by(o,setting_candidate) for o in self.obs],dtype=bool) #CHANGEHIT
                        self.N_hits += is_hit_candidate
                        #print("complete cliques are", completecliques)
                        # ---- Store both scores along with the clique ----
                        cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), setting_candidate)) 
                        #cliques_with_epsilon.append((self.get_epsilon_Bernstein(delta), clique))
                        #cliques_with_epsilon.append((self.get_Bernstein_bound(), clique))
                        #cliques_with_epsilon.append((self.get_inconfidence_bound(), clique))
                        self.N_hits -= is_hit_candidate
            else:
                #print("I already met this candidate")
                for setting_candidate in self.clean_setting_cache:
                    #key = tuple(tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in clique)
                    #is_hit_candidate = self.is_hit_clique_cache[key]
                    is_hit_candidate = []
                    is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                    self.N_hits += is_hit_candidate
                    cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), setting_candidate))
                    self.N_hits -= is_hit_candidate
            is_hit_candidate = []
            if not any((existing == shadowcliquesetting).all() for existing in self.clean_setting_cache):
                    self.clean_setting_cache.append(shadowcliquesetting)
            #self.clean_setting_cache[center_node].append(shadowcliquesetting)
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            self.N_hits += is_hit_candidate
            cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), shadowcliquesetting))
            print("epsilon for shadow clique is",self.get_epsilon_Bernstein_no_restricted_validity(delta))
            self.N_hits -= is_hit_candidate
            #if valid_cliques:
            #    self.clean_clique_cache[center_node] = valid_cliques
            #    self.clean_setting_cache[center_node] = valid_settings
            #print("length of clean setting cache", len(self.clean_setting_cache[center_node]))
            if not cliques_with_epsilon:
                raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
            print("length of cliques with epsilon is", len(cliques_with_epsilon))
            
            # Sort cliques by epsilon
            cliques_with_epsilon.sort(key=lambda x: x[0])
            _, best_clique = cliques_with_epsilon[0]
            #print("epsilon for best clique is",cliques_with_epsilon[0])
            self.selected_cliques.append(best_clique)
            #if len(best_clique) == len(shadowclique) and all(np.array_equal(np.array(o1), np.array(o2)) for o1, o2 in zip(best_clique, shadowclique)):
             #   self.shadow_was_best_count += 1
             #   print("Shadow clique was selected", self.shadow_was_best_count, "times")
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
                print("Shadow clique was selected", self.shadow_was_best_count, "times")
            else:
                print("epsilon for best clique is",cliques_with_epsilon[0])
            #print("best clique has",len(best_clique),"members")
            setting = best_clique
        
        
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
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info

class Shadow_Grouping_Update12(Measurement_scheme):
    """ 
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.shadow_was_best_count = 0
        self.clique_cache = {}  # maps observable tuple -> hit cliques
        self.clean_clique_cache = {}  
        self.clean_setting_cache = self.find_all_settings()
        self.is_hit_clique_cache = {}
        self.processed_center_node = []
        if self.weight_function is not None:
            test = self.weight_function(self.w,self.eps,self.N_hits)
            assert len(test) == len(self.w), "Weight function is supposed to return an array of shape {} (i.e. number of observables) but returned an array of shape {}".format(self.w.shape,test.shape)
        self.is_sampling = False
        return

    def find_all_settings(self):
        unprocessed = self.obs
        valid_settings = []
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        for o in unprocessed:
            print("checking observable", o)
            hit_list = []
            hit_cliques = []
            globalsetting = o
            for o in self.obs:
                if hit_by(o, globalsetting):
                    hit_list.append(o)
                    #unprocessed = list(unprocessed)
                    #unprocessed.remove(o)
                    #unprocessed = unprocessed[unprocessed != o]
                    unprocessed = np.array([x for x in unprocessed if not np.array_equal(x, o)])
            if not hit_list:
                raise RuntimeError("No hit list found.")
            hit_graph = build_hit_graph(hit_list)
            if not hit_graph:
                #hit_cliques = o
                raise RuntimeError("No hit graph found.")
            hit_cliques = find_cliques5(hit_graph)
            if not hit_cliques:
                setting_candidate = np.zeros(self.num_qubits, dtype=int)
                o_arr = np.array(o)
                non_id = o_arr != 0
                setting_candidate[non_id] = o_arr[non_id]
                if np.min(setting_candidate) == 0:
                    print("Removed incomplete clique")
                else:        
                    if not any((existing == setting_candidate).all() for existing in valid_settings):
                        valid_settings.append(setting_candidate)
                        print("added complete clique")
            else:        
                for clique in hit_cliques:
                    setting_candidate = np.zeros(self.num_qubits, dtype=int)
                    for o in clique:
                        o_arr = np.array(o)
                        if hit_by(o_arr, setting_candidate):
                            non_id = o_arr != 0
                            setting_candidate[non_id] = o_arr[non_id]
                        if np.min(setting_candidate) > 0:
                            break
                    if np.min(setting_candidate) == 0:
                        print("Removed incomplete clique")
                    else:        
                        if not any((existing == setting_candidate).all() for existing in valid_settings):
                            valid_settings.append(setting_candidate)
                            print("added complete clique")
                        else:
                            print("Removed already existing clique")
        print("length of valid settings are",len(valid_settings))
        return valid_settings

            
    
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
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        order = np.argsort(weights)
        self.selected_cliques = []  # Stores best cliques from each round
        setting = np.zeros(self.num_qubits,dtype=int)
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        if verbose:
            print("Checking list of observables.")

        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
        center_node = tuple(first_obs)  # Use tuple as dictionary key
        print("center node is", first_idx)
        
        tstart = time()
        
        if np.any(self.N_hits == 0):
            for idx in reversed(order):
                o = self.obs[idx]
                if verbose:
                    print("Checking",o)
                if hit_by(o,setting):
                    non_id = o!=0
                    setting[non_id] = o[non_id]
                    if np.min(setting) > 0:
                        print("p =",setting)
                        break
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
            cliques_with_epsilon = []
            delta = 0.02
            removedcliques = 0
            completecliques = 0
            valid_settings = []
            valid_cliques = []
            for setting_candidate in self.clean_setting_cache:
                is_hit_candidate = []
                is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                self.N_hits += is_hit_candidate
                cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), setting_candidate))
                self.N_hits -= is_hit_candidate
            if not cliques_with_epsilon:
                raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
            is_hit_candidate = []
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            self.N_hits += is_hit_candidate
            cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), shadowcliquesetting))
            self.N_hits -= is_hit_candidate
            print("length of cliques with epsilon is", len(cliques_with_epsilon))
            cliques_with_epsilon.sort(key=lambda x: x[0])
            _, best_clique = cliques_with_epsilon[0]
            self.selected_cliques.append(best_clique)
            setting = best_clique
        
        
        tend = time()

        is_hit = []
        is_hit = hit_by_batch_numba(self.obs , setting)
        self.N_hits += is_hit
        delta = 0.02
            
        
        # further info for comparisons
        info = {}
        info["total_weight"] = np.sum(weights[is_hit])
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        if verbose:
            print("Finished assigning with total weight of",info["total_weight"])
        return setting, info


class Shadow_Grouping_Update13(Measurement_scheme):
    """ 
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon,weight_function):
        super().__init__(observables,weights,epsilon)
        self.N_hits = np.zeros_like(self.N_hits)
        self.weight_function = weight_function
        self.shadow_was_best_count = 0
        self.round_num = 0
        self.clique_cache = {}  # maps observable tuple -> hit cliques
        self.clean_clique_cache = {}  
        self.clean_setting_cache = {}
        self.is_hit_clique_cache = {}
        self.processed_center_node = []
        self.rounds = []
        self.eps_values_v3 = []
        self._cached_graph = build_hit_graph(observables)
        #self._cached_cliques = list(nx.find_cliques(self._cached_graph))
        self._cached_cliques = list(greedy_clique_cover(self._cached_graph))
        #self._cached_cliques = approximate_clique_cover(self._cached_graph)
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

    def total_hit_weight(self, weights, is_hit):
        weights = np.asarray(weights, dtype=float)
        is_hit = np.asarray(is_hit, dtype=bool)
        return (weights * is_hit).sum()
        
    def find_setting(self,verbose=False):
        """ Finds the next measurement setting. Can be verbosed to gain further information during the procedure. """
        # sort observable list by respective weight
        weights = self.weight_function(self.w,self.eps,self.N_hits)
        #print(f"alpha is = ", alpha)
        order = np.argsort(weights)
        self.selected_cliques = []  # Stores best cliques from each round
        setting = np.zeros(self.num_qubits,dtype=int)
        globalsetting = np.zeros(self.num_qubits,dtype=int)
        if verbose:
            print("Checking list of observables.")

        # Get highest-weight observable
        first_idx = order[-1]  # last one in ascending sort = highest weight
        first_obs = self.obs[first_idx]
        center_node = tuple(first_obs)  # Use tuple as dictionary key
        print("center node is", first_idx, "and its weight is", weights[first_idx])

        tstart = time()
        
        if np.any(self.N_hits == 0):
            print("length of cached cliques is", len(self._cached_cliques))
            first_clique = self._cached_cliques[0]
            print("size of first clique is", len(first_clique))
            setting_candidate = np.zeros(self.num_qubits, dtype=int)
            for o in first_clique:
                o_arr = np.array(o)
                if hit_by(o_arr, setting_candidate):
                    non_id = o_arr != 0
                    setting_candidate[non_id] = o_arr[non_id]
                    if np.min(setting_candidate) > 0:
                        break
            if np.min(setting_candidate) == 0:
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
                        break
            del self._cached_cliques[0]
            setting = setting_candidate
            if center_node not in self.clean_setting_cache:
                self.clean_setting_cache[center_node] = []
                self.clean_clique_cache[center_node] = []
            self.clean_setting_cache[center_node].append(setting_candidate)
        
        else:
            #print("shadow regime ended")
            #print(self.N_hits)
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

            if verbose:
                print("Checking list of observables.")
                
            cliques_with_epsilon = []
            delta = 0.02
            removedcliques = 0
            completecliques = 0
            valid_settings = []
            valid_cliques = []
            if center_node not in self.processed_center_node:
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
                #print("hit_graph has", hit_graph.number_of_nodes(), "nodes and", hit_graph.number_of_edges(), "edges.")
                if not hit_graph:
                    raise RuntimeError("No hit graph found.")
                hit_cliques = find_cliques5(hit_graph)
                self.clique_cache[center_node] = hit_cliques
                if not hit_cliques:
                    hit_cliques=[[center_node]]
                for clique in hit_cliques:
                    if center_node not in self.clean_setting_cache:
                        self.clean_setting_cache[center_node] = []
                        self.clean_clique_cache[center_node] = []
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
                                break
                        completecliques += 1
                        self.clean_clique_cache[center_node].append(clique)
                        if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache[center_node]):
                            self.clean_setting_cache[center_node].append(setting_candidate)

                    else:
                        completecliques += 1
                        self.clean_clique_cache[center_node].append(clique)
                        if not any((existing == setting_candidate).all() for existing in self.clean_setting_cache[center_node]):
                            self.clean_setting_cache[center_node].append(setting_candidate)
                for cached_settings in self.clean_setting_cache.values():
                    for setting_candidate in cached_settings:
                        is_hit_candidate = []
                        is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                        self.N_hits += is_hit_candidate
                        cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), setting_candidate))
                        #cliques_with_epsilon.append((self.get_Bernstein_bound(), setting_candidate))
                        self.N_hits -= is_hit_candidate
            else:
                for cached_settings in self.clean_setting_cache.values():
                    for setting_candidate in cached_settings:
                        is_hit_candidate = []
                        is_hit_candidate = hit_by_batch_numba(self.obs , setting_candidate)
                        self.N_hits += is_hit_candidate
                        cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), setting_candidate))
                        #cliques_with_epsilon.append((self.get_Bernstein_bound(), setting_candidate))
                        self.N_hits -= is_hit_candidate
            is_hit_candidate = []
            if not any((existing == shadowcliquesetting).all() for existing in self.clean_setting_cache[center_node]):
                    self.clean_setting_cache[center_node].append(shadowcliquesetting)
            #self.clean_setting_cache[center_node].append(shadowcliquesetting)
            is_hit_candidate = hit_by_batch_numba(self.obs , shadowcliquesetting)
            self.N_hits += is_hit_candidate
            cliques_with_epsilon.append((self.get_epsilon_Bernstein_no_restricted_validity(delta), shadowcliquesetting))
            #cliques_with_epsilon.append((self.get_Bernstein_bound(), shadowcliquesetting))
            print("epsilon for shadow clique is",self.get_epsilon_Bernstein_no_restricted_validity(delta))
            self.N_hits -= is_hit_candidate
            if not cliques_with_epsilon:
                raise RuntimeError("No cliques with valid observables were found. Likely due to representation mismatch or empty hit_cliques.")
            print("length of cliques with epsilon is", len(cliques_with_epsilon))
            
            # Sort cliques by epsilon
            cliques_with_epsilon.sort(key=lambda x: x[0])
            _, best_clique = cliques_with_epsilon[0]
            #print("epsilon for best clique is",cliques_with_epsilon[0])
            self.selected_cliques.append(best_clique)
            if (len(best_clique) == len(shadowcliquesetting) and all(np.array_equal(a, b) for a, b in zip(best_clique, shadowcliquesetting))):
                self.shadow_was_best_count += 1
                print("Shadow clique was selected", self.shadow_was_best_count, "times")
            else:
                print("epsilon for best clique is",cliques_with_epsilon[0])
            #print("best clique has",len(best_clique),"members")
            setting = best_clique
        
        
        tend = time()

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
        info["run_time"] = tend - tstart
        info["epsilon_Bernstein_no_restricted_validity"] = self.get_epsilon_Bernstein_no_restricted_validity(delta)
        #info["epsilon_Bernstein_no_restricted_validity_v2"] = self.get_epsilon_Bernstein_no_restricted_validity_v2(delta)
        info["epsilon_Bernstein_no_restricted_validity_v3"] = self.get_epsilon_Bernstein_no_restricted_validity_v3(delta)
        self.eps_values_v3.append(info["epsilon_Bernstein_no_restricted_validity_v3"])
        print("epsilon_Bernstein_no_restricted_validity:", info["epsilon_Bernstein_no_restricted_validity"])
        #print("epsilon_Bernstein_no_restricted_validity_v2:", info["epsilon_Bernstein_no_restricted_validity_v2"])
        print("epsilon_Bernstein_no_restricted_validity_v3:", info["epsilon_Bernstein_no_restricted_validity_v3"])
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
        best_setting, best_weight = [], np.infty
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

class AdaptiveShadows(Shadow_Grouping):
    """ Comparison class to Shadow_Grouping, based on https://github.com/charleshadfield/adaptiveshadows/.
        Starts-off as classical shadows (uniformly at random) but biases the distribution
        the more the Pauli bases have been set. Does not require any hyperparameters.
        epsilon (optional): parameter solely used for comparison with other methods. Defaults to 0.1.
        
        Returns p and a dictionary info holding further details on the matching procedure.
    """
    
    def __init__(self,observables,weights,epsilon=0.1):
        super().__init__(observables,weights,epsilon,None)
        self.is_sampling = True
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
        
        info = {"inconfidence_bound": self.get_inconfidence_bound(),
                "Bernstein bound":    self.get_Bernstein_bound(),
                "run_time":           tend - tstart
               }
            
        return np.array(setting), info
            
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
    
class Derandomization(Shadow_Grouping):

    """ Finds the next measurement setting following the derandomization procedure.
        Optionally, a parameter delta in [0,1] can be provided to vary the degree of randomness (delta == 1 fully random, delta == 0 as proposed).
        If num_measurements is provided, the corresponding inconfidence bound is adapted to that.
        If use_one_norm, implements a 1-norm weighting to the bound as proposed in the paper.
    """

    def __init__(self,observables,weights,epsilon,delta=0,num_measurements=None,use_one_norm=False):
        super().__init__(observables,weights,epsilon,None)
        
        self.num_measurements = num_measurements
        # (n x M) integer array with entries in {0,1,2,3} == {E,X,Y,Z}
        self.localities = np.zeros((self.num_qubits+1,self.num_obs),dtype=int) # keep the last zero as the support of an empty Pauli string
        self.localities[:-1,:] = np.array([np.sum(observables[:,i:]!=0,axis=1) for i in range(self.num_qubits)])
        self.N_hits = np.zeros(self.num_obs,dtype=int)
        self.eps_greedy = delta
        self.scheme_params["eps_greedy"] = delta
        self.scheme_params["use_one_norm"] = use_one_norm
        
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
        info["inconfidence_bound"] = self.get_inconfidence_bound()
        info["Bernstein bound"] = self.get_Bernstein_bound()
        info["run_time"] = tend - tstart
        #if verbose:
            #print("Finished assigning with total weight of",info["total_weight"])
        
        return np.array(self.last_assignment), info



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

    


