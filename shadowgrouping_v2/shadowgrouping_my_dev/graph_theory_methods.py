import numpy as np, networkx as nx, time, warnings
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

def block_is_clique(block, conn):
    """
    Check if `block` is a clique in the graph with adjacency matrix `conn`.

    Parameters
    ----------
    block : iterable of int
        Qubit indices in the FC block.
    conn : 2D bool array, shape (num_qubits, num_qubits)
        conn[i,j] is True iff qubits i and j are directly connected
        by a native 2-qubit gate.

    Returns
    -------
    bool
        True iff for every unordered pair {i, j} in block with i != j
        we have conn[i, j] == True.
    """
    b = list(block)
    L = len(b)
    if L <= 1:
        # Trivially a clique
        return True

    for i_idx in range(L):
        qi = b[i_idx]
        row = conn[qi]
        for j_idx in range(i_idx + 1, L):
            qj = b[j_idx]
            if not row[qj]:
                return False
    return True

def block_is_connected_subgraph(block, conn):
    """
    Check if the induced subgraph on `block` is connected.

    Parameters
    ----------
    block : iterable of int
        Qubit indices in the FC block.
    conn : 2D bool array, shape (num_qubits, num_qubits)
        conn[i,j] is True iff there is a native 2-qubit gate between i and j.

    Returns
    -------
    bool
        True iff the undirected graph induced by `block` is connected.
    """
    b = list(block)
    L = len(b)
    if L <= 1:
        return True

    block_set = set(b)
    visited = set()
    stack = [b[0]]

    while stack:
        q = stack.pop()
        if q in visited:
            continue
        visited.add(q)

        # neighbors inside the block
        for r in block_set:
            if r not in visited and conn[q, r]:
                stack.append(r)

    return len(visited) == L

def clique_measurement_basis(pauli_array, clique, which_format="integer", complete_basis=False):
    """
    Determine the measurement basis (tensor product of single-qubit bases)
    corresponding to a given clique of qubit-wise commuting Pauli observables.

    Parameters
    ----------
    pauli_array : ndarray, shape (M, n)
        Pauli observables with entries {0,1,2,3} for {I,X,Y,Z}.
    clique : list of int
        Indices of observables forming one clique.
    which_format : {"integer", "string"}, optional
        - "integer" (default) returns a list of ints {0,1,2,3}
          where 0 means "no measurement"/identity.
        - "string" returns a string like "XYZ" or "IXZ".
    complete_basis : bool, optional
        - False (default): keep identities (0) as "unmeasured" qubits.
        - True : replace identities (0) with Z (3), i.e. measure every qubit in
          X/Y/Z.

    Returns
    -------
    basis : list[int] or str
        Measurement basis in requested format.
    """
    n = pauli_array.shape[1]

    # Pick observable in clique with maximum support to initialize
    supports = [np.count_nonzero(pauli_array[idx]) for idx in clique]
    start_idx = clique[np.argmax(supports)]
    basis_choice = pauli_array[start_idx].copy()

    # Refine basis with other observables in clique
    for idx in clique:
        for q in range(n):
            if basis_choice[q] == 0 and pauli_array[idx, q] != 0:
                basis_choice[q] = pauli_array[idx, q]

        # Break early if no identities left
        if np.all(basis_choice != 0):
            break

    if which_format == "integer":
        if complete_basis:
            # Replace identities (0) with Z (3)
            basis_choice = basis_choice.copy()
            basis_choice[basis_choice == 0] = 3
        # Otherwise keep 0's as "no measurement"
        return basis_choice.tolist()

    elif which_format == "string":
        if complete_basis:
            # I → Z by convention (we measure every qubit)
            mapping = {0: 'Z', 1: 'X', 2: 'Y', 3: 'Z'}
        else:
            # Keep explicit identities
            mapping = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
        return "".join(mapping[int(x)] for x in basis_choice)

    else:
        raise ValueError("which_format must be either 'integer' or 'string'")

def compat_matrix_to_graph(C: np.ndarray) -> nx.Graph:
    """
    Build an undirected graph from a symmetric boolean compatibility matrix C.
    C[i,j]=True means edge (i,j) exists; diagonal is ignored.
    """
    n = C.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # add edges for i<j only
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if C[i, j]]
    G.add_edges_from(edges)
    return G

def greedy_clique_cover_by_coloring(G: nx.Graph, strategy: str = "saturation_largest_first",
                                    overlap: bool = True, *, 
                                    n_random: int = 10, 
                                    seed: Optional[int] = None,
                                    min_num_cliques_per_node: Optional[int] = None,
                                    augment_limit: int = 5000) -> List[List[int]]:
    """
    Greedy clique cover via greedy coloring of the complement graph.
    
    If overlap=True and min_num_cliques_per_node is set, this function ensures 
    redundancy by augmenting the clique pool for specific 'deficit' nodes.
    
    Parameters
    ----------
    augment_limit : int
        Safety cap on the number of maximal cliques generated during the 
        augmentation phase for a single node. Prevents infinite hangs on dense graphs.
    """
    if G.number_of_nodes() == 0:
        return []

    H = nx.complement(G)

    def _coloring_to_cliques(coloring: Dict[int, int]) -> List[List[int]]:
        buckets = defaultdict(list)
        for v, c in coloring.items():
            buckets[c].append(v)
        return [sorted(bucket) for bucket in buckets.values()]

    # Case 1: Non-overlapping (Single Partition)
    if not overlap:
        color = nx.algorithms.coloring.greedy_color(H, strategy=strategy)
        return _coloring_to_cliques(color)

    # Case 2: Overlapping Pool Generation
    
    # We maintain a deduplicated pool using a dictionary
    # Key: tuple(clique), Value: None
    seen_cliques_map: Dict[Tuple[int, ...], None] = {} 
    
    def _add_cliques(cliques_list: List[List[int]]):
        for clq in cliques_list:
            key = tuple(clq)
            if key not in seen_cliques_map:
                seen_cliques_map[key] = None

    # 2.1 Run Deterministic Strategies
    base_strategies = [
        "saturation_largest_first", "largest_first", "smallest_last",
        "independent_set", "connected_sequential_bfs", "connected_sequential_dfs"
    ]
    for strat in base_strategies:
        color = nx.algorithms.coloring.greedy_color(H, strategy=strat)
        _add_cliques(_coloring_to_cliques(color))

    # 2.2 Run Random Restarts
    if n_random > 0:
        rng = np.random.default_rng(seed)

        def numpy_random_strategy(G, colors):
            nodes = list(G)
            rng.shuffle(nodes)
            return iter(nodes)

        for _ in range(int(n_random)):
            color = nx.algorithms.coloring.greedy_color(H, strategy=numpy_random_strategy)
            _add_cliques(_coloring_to_cliques(color))

    # Convert to initial pool
    full_pool = [list(c) for c in seen_cliques_map.keys()]

    # If no filtering is requested, return all unique cliques found
    if min_num_cliques_per_node is None:
        return full_pool

    # Case 3: Augmentation & Multi-Cover Selection
    
    target_k = int(min_num_cliques_per_node)
    nodes = list(G.nodes())
    
    # Map: Node -> List of compatible clique INDICES in full_pool
    # (We rebuild this map dynamically if we augment the pool)
    node_to_clique_indices = defaultdict(list)
    for idx, clq in enumerate(full_pool):
        for node in clq:
            node_to_clique_indices[node].append(idx)

    # 3.1 AUGMENTATION PHASE
    # Check for nodes that don't have enough options in the current pool
    deficit_nodes = []
    for v in nodes:
        if len(node_to_clique_indices[v]) < target_k:
            deficit_nodes.append(v)
    
    if deficit_nodes:
        # We need to add more cliques for these specific nodes.
        # We use nx.find_cliques(G, nodes=[v]) to find maximal cliques containing v.
        
        new_cliques_found = 0
        
        for v in deficit_nodes:           
            # Generator for maximal cliques containing v
            clique_gen = nx.find_cliques(G, nodes=[v])
            
            count_for_v = 0
            for clq in clique_gen:
                # Safety break: Don't spend forever on one dense node
                if count_for_v >= augment_limit:
                    break
                    
                clq_sorted = sorted(clq)
                key = tuple(clq_sorted)
                
                # If this is a NEW clique, add it
                if key not in seen_cliques_map:
                    seen_cliques_map[key] = None
                    # Append to full_pool and update indices immediately
                    new_idx = len(full_pool)
                    full_pool.append(clq_sorted)
                    for u in clq_sorted:
                        node_to_clique_indices[u].append(new_idx)
                    
                    new_cliques_found += 1
                
                count_for_v += 1
                
                # Heuristic break: If we have found enough NEW cliques for this node, 
                # we can stop early to save time. 
                # We check the updated length of valid indices for v.
                if len(node_to_clique_indices[v]) >= target_k + 5: # +5 buffer
                    break

    # 3.2 SELECTION PHASE (Greedy Multi-Cover)
    
    picked_indices = set()
    coverage_count = defaultdict(int) 
    
    # Loop layers from k=1 to target_k
    for k in range(1, target_k + 1):
        
        # Rank nodes by average size of *available* cliques
        node_scores = []
        for v in nodes:
            available_indices = [i for i in node_to_clique_indices[v] if i not in picked_indices]
            
            if not available_indices:
                avg_size = 0.0
            else:
                total_size = sum(len(full_pool[i]) for i in available_indices)
                avg_size = total_size / len(available_indices)
            
            node_scores.append((v, avg_size))
        
        # Sort descending
        node_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_nodes = [x[0] for x in node_scores]

        # Greedy Selection
        for v in ranked_nodes:
            if coverage_count[v] >= k:
                continue
            
            candidates = [i for i in node_to_clique_indices[v] if i not in picked_indices]
            
            if not candidates:
                continue

            # Pick largest candidate
            best_idx = max(candidates, key=lambda i: len(full_pool[i]))
            
            picked_indices.add(best_idx)
            for u in full_pool[best_idx]:
                coverage_count[u] += 1

    # 3.3 FINAL VALIDATION & WARNING
    # Check if any node fell short of the target
    failed_nodes = []
    for v in nodes:
        if coverage_count[v] < target_k:
            failed_nodes.append(v)
            
    if failed_nodes:
        # We cap the number of nodes printed in the warning to avoid spam
        example_nodes = failed_nodes[:5] 
        msg = (f"Warning: Could not find {target_k} compatible cliques for {len(failed_nodes)} "
               f"nodes (Examples: {example_nodes}). The graph structure may not allow "
               f"this many unique maximal cliques for these nodes.")
        warnings.warn(msg)

    return [full_pool[i] for i in picked_indices]

def greedy_clique_cover_by_setcover(
    G: nx.Graph, 
    overlap: bool = True, 
    min_num_cliques_per_node: int = 1, 
    augment_limit: int = 5000
) -> List[List[int]]:
    """
    Greedy set-cover over a clique pool.

    Parameters
    ----------
    G : nx.Graph
        The compatibility graph.
    overlap : bool
        If True, allows overlapping cliques to ensure redundancy.
        If False, returns a strict partition (min_num_cliques_per_node is ignored/treated as 1).
    min_num_cliques_per_node : int
        Target redundancy. The algorithm attempts to find at least this many 
        compatible cliques for every node.
    augment_limit : int
        A universal complexity cap that limits:
        1. The graph size for whole-graph clique enumeration (enhance_n_max).
        2. The max cliques added during enumeration (enum_max_cliques).
        3. The max cliques searched per node during deficit augmentation.

    Internal Defaults (Auto-Scaled)
    -------------------------------
    - pool_size: Scaled to roughly (2 * N * min_num_cliques).
    - n_restarts: Fixed at 5 to prioritize heavy random sampling over restart overhead.
    - random_starts_per_restart: Scaled to fill the calculated pool_size.
    - enhance: Enabled if N <= augment_limit.
    """
    n = G.number_of_nodes()
    if n == 0:
        return []
    if n == 1:
        v = next(iter(G.nodes()))
        return [[v]]

    # 1. AUTO-SCALING HYPERPARAMETERS
    rng = np.random.default_rng(None) 
    
    target_k = int(min_num_cliques_per_node) if overlap else 1

    # Heuristic: We need a pool large enough to hold 'k' distinct cliques for 'n' nodes.
    # We add a safety factor of 2.0 to account for overlaps and inefficient generation.
    # Minimum baseline of 2000 to ensure good mixing on small graphs.
    pool_size = int(max(2000, 2.0 * n * target_k))

    # We fix restarts to a small number and rely on massive random sampling per restart
    n_restarts = 5
    
    # Calculate how many random shots we need per restart to fill that pool
    # (Subtracting 1 for the deterministic pass per restart)
    random_starts_per_restart = int(pool_size / n_restarts)

    # 2. POOL GENERATION
    deg = dict(G.degree())
    nodes = list(G.nodes())
    
    _nbr_cache = {}
    def nbrs(v):
        s = _nbr_cache.get(v)
        if s is None:
            s = set(G.neighbors(v))
            _nbr_cache[v] = s
        return s

    def grow_clique(start, *, pick_mode: str = "degree") -> frozenset:
        clique = {start}
        candidates = nbrs(start).copy()
        while candidates:
            if pick_mode == "random":
                nxt = rng.choice(tuple(candidates))
            else:
                nxt = max(candidates, key=lambda u: deg.get(u, 0))
            clique.add(nxt)
            candidates &= nbrs(nxt)
        return frozenset(clique)

    pool = set()
    nodes_by_deg = sorted(nodes, key=lambda v: deg.get(v, 0), reverse=True)

    # 2.1 Heuristic Growth Loop
    for r in range(n_restarts):
        uncovered_local = set(nodes)
        pick_mode = "degree" if (r % 2 == 0) else "degree"

        # Deterministic Pass (Target Uncovered)
        while uncovered_local and len(pool) < pool_size:
            start = None
            for v in nodes_by_deg:
                if v in uncovered_local:
                    start = v
                    break
            if start is None:
                start = next(iter(uncovered_local))

            C = grow_clique(start, pick_mode=pick_mode)
            pool.add(C)
            uncovered_local -= set(C)

        if len(pool) >= pool_size: break

        # Random Pass (Diversity)
        for _ in range(random_starts_per_restart):
            if len(pool) >= pool_size: break
            start = rng.choice(nodes)
            mode = "random" if (rng.random() < 0.25) else "degree"
            pool.add(grow_clique(start, pick_mode=mode))

        if len(pool) >= pool_size: break

    # 2.2 Whole-Graph Enhancement (Optional)
    # Controlled by augment_limit acting as 'enhance_n_max'
    m = G.number_of_edges()
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0
    
    # Defaults for enhancement logic
    small_n_thresh = 400
    density_thresh = 0.03
    enum_time_limit_s = 0.75
    
    # Run if graph is small/sparse enough AND below the user's hard limit
    do_enhance = ((n <= small_n_thresh or density <= density_thresh) and 
                  (n <= augment_limit))

    if do_enhance:
        t0 = time.perf_counter()
        added = 0
        # For simplicity, we skip relabeling strategies here to minimize params
        for clique in nx.find_cliques(G):
            if (time.perf_counter() - t0) > enum_time_limit_s: break
            if added >= augment_limit: break # augment_limit acting as enum_max_cliques
            if not clique: continue
            
            pool.add(frozenset(clique))
            added += 1

    # 3. AUGMENTATION PHASE (Deficit Nodes)
    # Ensure every node has 'target_k' candidates in the pool
    
    full_pool = list(pool)
    node_to_clique_indices = defaultdict(list)
    for idx, clq in enumerate(full_pool):
        for node in clq:
            node_to_clique_indices[node].append(idx)
            
    if overlap and target_k > 1:
        deficit_nodes = [v for v in nodes if len(node_to_clique_indices[v]) < target_k]
        
        if deficit_nodes:
            known_cliques_hashes = {C for C in pool} 
            
            for v in deficit_nodes:
                clique_gen = nx.find_cliques(G, nodes=[v])
                
                count_for_v = 0
                for clq in clique_gen:
                    # Safety Break using augment_limit
                    if count_for_v >= augment_limit: break
                    
                    c_frozenset = frozenset(clq)
                    if c_frozenset not in known_cliques_hashes:
                        known_cliques_hashes.add(c_frozenset)
                        new_idx = len(full_pool)
                        full_pool.append(c_frozenset)
                        for u in clq:
                            node_to_clique_indices[u].append(new_idx)
                    
                    count_for_v += 1
                    # Stop if we found enough options for this node (+buffer)
                    if len(node_to_clique_indices[v]) >= target_k + 5:
                        break

    # 4. SELECTION PHASE
    
    picked_indices = set()
    coverage_count = defaultdict(int)
    
    # Helper to format output
    def build_result():
        return [sorted([int(x) for x in full_pool[i]]) for i in picked_indices]

    # PHASE 4A: BASE COVER (Minimum Clique Cover)
    # Goal: Cover every node at least once (k=1) maximizing Gain
        
    uncovered = set(nodes)
    pool_sets = [set(C) for C in full_pool]
    
    # We will build the final list directly if overlap=False (Partition Mode)
    # If overlap=True, we continue using indices to support Phase 4B.
    partition_cover = [] 
    
    while uncovered:
        best_idx = None
        best_gain = -1
        best_size = -1
        
        # Global Greedy Scan
        for i, C_set in enumerate(pool_sets):
            if i in picked_indices: continue
            
            gain = len(C_set & uncovered)
            if gain <= 0: continue
            
            if gain > best_gain or (gain == best_gain and len(C_set) > best_size):
                best_gain = gain
                best_size = len(C_set)
                best_idx = i
        
        # Fallback for unconnected/isolated nodes
        if best_idx is None:
            v = next(iter(uncovered))
            if node_to_clique_indices[v]:
                best_idx = node_to_clique_indices[v][0] 
            else:
                # Create singleton
                singleton = frozenset([v])
                full_pool.append(singleton)
                best_idx = len(full_pool) - 1
                pool_sets.append({v})
                node_to_clique_indices[v].append(best_idx)
                
        picked_indices.add(best_idx)
        chosen_set = pool_sets[best_idx]
        
        if overlap:
            # Standard Mode: Mark covered, but keep full clique structure
            uncovered -= chosen_set
            for u in chosen_set:
                coverage_count[u] += 1
        else:
            # Partition Mode: "Trim" the clique to fit the hole
            # We only keep the part of the clique that intersects with 'uncovered'
            trimmed_clique = chosen_set & uncovered
            partition_cover.append(sorted([int(x) for x in trimmed_clique]))
            
            # Remove these specific nodes from uncovered
            uncovered -= trimmed_clique
            
    if not overlap:
        # Return the trimmed list directly
        return partition_cover

    # PHASE 4B: REDUNDANCY (Largest First)
    # Goal: Ensure coverage_count[v] >= target_k using Largest Available
    
    for k in range(2, target_k + 1):
        
        # Rank nodes by average size of available cliques
        node_scores = []
        for v in nodes:
            if coverage_count[v] >= k: continue
            
            avail = [i for i in node_to_clique_indices[v] if i not in picked_indices]
            if not avail:
                avg_size = 0.0
            else:
                avg_size = sum(len(full_pool[i]) for i in avail) / len(avail)
            node_scores.append((v, avg_size))
            
        node_scores.sort(key=lambda x: x[1], reverse=True)
        ranked_nodes = [x[0] for x in node_scores]
        
        for v in ranked_nodes:
            if coverage_count[v] >= k:
                continue
                
            candidates = [i for i in node_to_clique_indices[v] if i not in picked_indices]
            if not candidates: continue
            
            # Pick Largest Available
            best_idx = max(candidates, key=lambda i: len(full_pool[i]))
            
            picked_indices.add(best_idx)
            for u in full_pool[best_idx]:
                coverage_count[u] += 1

    # Warning Check
    failed_nodes = [v for v in nodes if coverage_count[v] < target_k]
    if failed_nodes:
        example_nodes = failed_nodes[:5] 
        msg = (f"Warning: Could not find {target_k} compatible cliques for {len(failed_nodes)} "
               f"nodes (Examples: {example_nodes}).")
        warnings.warn(msg)

    return build_result()