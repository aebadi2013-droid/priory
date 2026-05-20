import numpy as np, random, os
from .helper_functions import int_to_bitlist, bitlist_to_int
from .hamiltonian import Hamiltonian
from typing import Callable, Optional, List

# Functions to apply readout noise models to sampling outcomes

def apply_stochastic_bit_flips(samples, single_qubit_assignment_matrices,
                               pairwise_assignment_matrices=None,
                               measured_qubits=None, seed=None, bit_input=False):
    """
    Applies uncorrelated and correlated readout noise as a post-processing step.

    Parameters:
    ----------
    samples : ndarray
        Array of shape (Nshots, nqubits) with values in {0,1} or {-1,1}.
    
    single_qubit_assignment_matrices : dict
        Dictionary {q: A_q} of 2×2 assignment matrices for each qubit.

    pairwise_assignment_matrices : dict or None
        Optional dictionary {(i,j): A_ij} of 4×4 assignment matrices for correlated flips.
        
    measured_qubits : list or None
        List of indices of qubits (from 0 to nqubits-1) that are measured.
        If None, all qubits are assumed measured.
    
    seed : int or None
        Random seed for reproducibility.

    bit_input : bool
        If True, assumes inputs are bits {0,1}; otherwise signs {-1,+1} (default).

    Returns:
    -------
    noisy : ndarray
        Same shape as samples, in same bit/sign format.
    """
    rng = np.random.default_rng(seed)
    noisy = samples.copy()
    nshots, nqubits = samples.shape

    # Convert signs to bits if needed
    if not bit_input:
        noisy = ((1 - noisy) // 2).astype(int)  # {-1,+1} → {1,0}

    if measured_qubits is None:
        measured_qubits = list(range(nqubits))
    assert len(measured_qubits) <= nqubits

    correlated_mode = pairwise_assignment_matrices is not None and len(pairwise_assignment_matrices) > 0

    if correlated_mode:
        all_pairs = list(pairwise_assignment_matrices.keys())
        for s in range(nshots):
            col = noisy[s].copy()

            # Choose one correlated pair uniformly
            chosen_pair = all_pairs[rng.integers(len(all_pairs))]
            i, j = chosen_pair

            if i in measured_qubits and j in measured_qubits:
                col_entry = 2 * col[i] + col[j]
                probs = pairwise_assignment_matrices[chosen_pair][:, col_entry]
                probs /= probs.sum()
                sampled_row_entry = rng.choice(4, p=probs)
                b_i = (sampled_row_entry >> 1) & 1
                b_j = sampled_row_entry & 1
                noisy[s, i] = b_i
                noisy[s, j] = b_j

            applied_pair = {i, j}

            # Apply single-qubit flips to remaining measured qubits
            for q in range(nqubits):
                if q in applied_pair or q not in measured_qubits:
                    continue
                if q not in single_qubit_assignment_matrices:
                    continue
                A_q = single_qubit_assignment_matrices[q]
                col_bit = col[q]
                probs = A_q[:, col_bit]
                probs /= probs.sum()
                noisy[s, q] = rng.choice(2, p=probs)
    else:
        # Fast path for uncorrelated flips only
        for q, A_q in single_qubit_assignment_matrices.items():
            if q not in measured_qubits:
                continue
            p0to1 = A_q[1, 0]
            p1to0 = A_q[0, 1]
            flips = rng.random(nshots)
            mask0 = (noisy[:, q] == 0) & (flips < p0to1)
            mask1 = (noisy[:, q] == 1) & (flips < p1to0)
            noisy[mask0, q] = 1
            noisy[mask1, q] = 0

    # Convert bits back to signs if needed
    if not bit_input:
        noisy = 1 - 2 * noisy  # {0,1} → {+1,-1}

    return noisy

def apply_assignment_matrix(
    samples: np.ndarray,
    A_row_col: Callable[[List[int], List[int]], float],
    measured_qubits: Optional[List[int]] = None,
    bit_format: bool = False,
    lsb_first: bool = False,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply general readout assignment matrix to bitstring samples via post-processing.

    Parameters:
    -----------
    samples : ndarray (N, n)
        Input noiseless samples in bits {0,1} or signs {-1,+1}.
    A_row_col : Callable
        Function that returns A[row, col], with row and col given as bitlists.
    measured_qubits : list or None
        List of indices of measured qubits. Used to skip unmeasured qubits.
    bit_format : bool
        If False (default), input is in {-1, +1} and output will be as well.
        If True, inputs/outputs are in {0, 1}.
    lsb_first : bool
        Whether to use LSB-to-MSB ordering (default False).
    seed : int
        RNG seed for reproducibility.

    Returns:
    --------
    noisy_samples : ndarray (N, n)
        Noisy bitstrings in the same format as input.
    """
    rng = np.random.default_rng(seed)
    nshots, nqubits = samples.shape
    
    if measured_qubits is None:
        measured_qubits = list(range(nqubits)) # All qubits measured

    # Convert signs {-1, +1} to bits {0, 1} if needed
    if not bit_format:
        samples = ((1 - samples) // 2).astype(int)

    noisy_samples = np.zeros_like(samples)

    for i in range(nshots):
        col = samples[i].tolist()

        # Build probability distribution for each row
        probs = []
        for idx in range(2**nqubits):
            row = int_to_bitlist(idx, width=nqubits, lsb_first=lsb_first)
            # Only evaluate measured qubits; leave others untouched
            prob = A_row_col(row, col)
            probs.append(prob)
        probs = np.array(probs)
        probs /= probs.sum()

        # Sample new row index
        sampled_idx = rng.choice(2**nqubits, p=probs)
        sampled_row = int_to_bitlist(sampled_idx, width=nqubits, lsb_first=lsb_first)
        noisy_samples[i] = sampled_row

    # For unmeasured qubits, reset values to 1 or 0
    for i in range(nqubits):
        if i not in measured_qubits:
            noisy_samples[:, i] = 0 if bit_format else 1

    # Convert back to {-1, +1} if needed
    if not bit_format:
        noisy_samples = 1 - 2 * noisy_samples  # map {0,1} to {+1,-1}

    return noisy_samples

# Functions to generate assignment matrix

def generate_n_qubit_assignment_matrix_randomly(num_qubits, p, seed=None):
    """
    Generate a random column-stochastic 2^num_qubits × 2^num_qubits 
    assignment matrix A, with average diagonal value = p.
    
    Parameters:
    - num_qubits: number of qubits
    - p: target mean of diagonal entries (e.g., 0.95)
    
    Returns:
    - A: numpy array of shape (2^num_qubits, 2^num_qubits)
    """
    rng = np.random.default_rng(seed)
    d = 2**num_qubits
    A = np.zeros((d, d))

    # Step 1: Set diagonals to average around p
    diag = np.clip(rng.normal(loc=p, scale=0.01, size=d), 0.0, 1.0)

    for j in range(d):
        A[j, j] = diag[j]
        remaining = 1.0 - A[j, j]

        # Sample the remaining (d-1) entries for column j
        if d > 1:
            off_diag_indices = [i for i in range(d) if i != j]
            # Random point on (d-2)-simplex
            noise = rng.random(len(off_diag_indices))
            noise /= noise.sum()
            A[off_diag_indices, j] = remaining * noise

    return A

def A_row_col_T(row, col, S_matrix_dict):
    """
    Compute the matrix element A[row, col] of the n-qubit assignment matrix A
    that is the tensor product of single-qubit 2×2 assignment matrices, without
    explicitly forming the full matrix.

    Parameters:
    ----------
    row : list of int
        Noisy measurement outcome bitstring (list of 0s and 1s), length n.
        Corresponds to the row index of the full assignment matrix A.

    col : list of int
        Ideal (noiseless) measurement outcome bitstring (list of 0s and 1s), length n.
        Corresponds to the column index of A.

    S_matrix_dict : dict
        Dictionary mapping qubit indices (0 to n-1) to 2×2 single-qubit assignment matrices.
        Each entry S_matrix_dict[k] is a numpy array of shape (2, 2) representing the 
        conditional probability P(measured = i | true = j) for qubit k.

    Returns:
    -------
    float
        The matrix element A[row, col] = ∏_k S_k[row_k, col_k], where S_k is the
        2×2 assignment matrix for qubit k.
    """
    assert len(row) == len(col), "row and col must be bitstrings of equal length"
    n = len(row)
    assert set(S_matrix_dict.keys()) == set(range(n)), \
        f"S_matrix_dict must contain all qubit indices from 0 to {n-1}"

    matrix_element = 1.0
    for k, S in S_matrix_dict.items():
        matrix_element *= S[row[k], col[k]]
    return matrix_element

def make_A_row_col_T(S_matrix_dict):
    """Return a callable A_row_col that computes A[row, col] using S dictionary."""

    def A_row_col(row, col):
        matrix_element = 1.0
        for k in S_matrix_dict.keys():
            matrix_element *= S_matrix_dict[k][row[k], col[k]]
        return matrix_element

    return A_row_col

def A_row_col_C(row, col, S_matrix_dict, C_matrix_dict):
    """
    Computes the matrix element A[row, col] of an approximate n-qubit assignment matrix A
    that includes pairwise correlated readout errors. The final result is normalized to ensure
    the matrix is column-stochastic (i.e., columns sum to 1).

    The model assumes:
    - For each qubit pair (i, j), we have a 4×4 correlated assignment matrix C_ij.
    - All other qubits (not in the pair) are treated independently using 2×2 matrices S_k.
    - The total matrix element is the average over all such pairwise contributions.

    Parameters
    ----------
    row : list[int]
        Noisy bitstring (measurement outcome), length n.
    
    col : list[int]
        Noiseless bitstring (ideal outcome), length n.
    
    S_matrix_dict : dict[int -> np.ndarray]
        Dictionary mapping each qubit index k to a 2×2 single-qubit assignment matrix S_k.

    C_matrix_dict : dict[tuple[int, int] -> np.ndarray]
        Dictionary mapping each pair of qubit indices (i < j) to a 4×4 correlated
        assignment matrix C_ij. All pairs should be unique.

    Returns
    -------
    float
        Approximate matrix element A[row, col], properly normalized to form a valid
        assignment matrix.
    """
    n = len(row)
    assert len(col) == n, "Row and column bitstrings must be of equal length."
    assert set(S_matrix_dict.keys()) == set(range(n)), "S_matrix_dict must contain all qubit indices."

    matrix_element = 0.0

    for pair, C in C_matrix_dict.items():
        i, j = pair
        assert i != j, "Correlated pair must contain two distinct qubit indices."
        assert 0 <= i < n and 0 <= j < n, "Invalid qubit index in pair."

        row_ij = 2 * row[i] + row[j]
        col_ij = 2 * col[i] + col[j]
        term = C[row_ij, col_ij]

        for k in range(n):
            if k not in pair:
                term *= S_matrix_dict[k][row[k], col[k]]

        matrix_element += term

    num_pairs = len(C_matrix_dict)

    return matrix_element / num_pairs

def make_A_row_col_C(S_matrix_dict, C_matrix_dict):
    """Return a callable A_row_col that computes A[row, col] using S and C dictionaries."""
    from math import comb

    def A_row_col(row, col):
        matrix_element = 0
        pairs = list(C_matrix_dict.keys())
        for pair in pairs:
            k, l = pair
            row_entry = row[k]*2 + row[l]
            col_entry = col[k]*2 + col[l]
            element = C_matrix_dict[pair][row_entry, col_entry]

            for m in range(len(row)):
                if m not in pair:
                    element *= S_matrix_dict[m][row[m], col[m]]
            matrix_element += element

        if len(pairs) > 0:
            matrix_element /= comb(len(row), 2)

        return matrix_element

    return A_row_col

def A_row_col_F(row, col, A_matrix, lsb_first=False):
    """
    Compute the matrix element A[row, col] of the full n-qubit assignment matrix A.

    Parameters:
    ----------
    row : list of int
        Noisy measurement outcome bitstring (length n).
    col : list of int
        Ideal (noiseless) measurement outcome bitstring (length n).
    A_matrix : numpy.ndarray
        Full 2^n x 2^n assignment matrix.
    lsb_first : bool
        If True, interpret bitstrings as LSB-to-MSB order; else MSB-to-LSB.

    Returns:
    -------
    float
        The matrix element A[row, col].
    """
    assert isinstance(A_matrix, np.ndarray), "A_matrix must be a numpy array."
    assert len(row) == len(col), "row and col must be bitstrings of equal length"
    n = len(row)
    assert A_matrix.shape == (2**n, 2**n), "A_matrix must be 2^n x 2^n"

    row_idx = bitlist_to_int(row, lsb_first=lsb_first)
    col_idx = bitlist_to_int(col, lsb_first=lsb_first)
    return A_matrix[row_idx, col_idx]

def make_A_row_col_F(A, lsb_first=False):
    """Return a callable A_row_col that evaluates entries of full A matrix."""
    def A_row_col_F(row, col):
        i = bitlist_to_int(row, lsb_first=lsb_first)
        j = bitlist_to_int(col, lsb_first=lsb_first)
        return A[i, j]
    return A_row_col_F

# Functions to apply local and global depolarizing noise models at the end of the circuit only

def apply_local_depolarizing_noise_end_of_circuit(samples, p_array, measured_qubits=None):
    """
    Applies local depolarizing noise to measurement outcomes via post-processing.
    
    Parameters:
    - samples: array of shape (nshots, nqubits) with entries in {-1, +1}
    - p_array: list or array of depolarizing noise probabilities, one per qubit
    - measured_qubits: List of indices of measured qubits. If None, all measured.
    
    Returns:
    - noisy_samples: modified version of samples
    """
    nshots, nqubits = samples.shape
    noisy = samples.copy()
    
    for q in range(nqubits):
        p = p_array[q]

        # Effective flip probability depends on basis and channel
        if q not in measured_qubits:
            continue  # skip unmeasured qubits
        else:
            # Only X, Y flips affect Z measurement → total flip prob = 2p/3
            # Only Y, Z flips affect X measurement → total flip prob = 2p/3
            # Only X, Z flips affect Y measurement → total flip prob = 2p/3
            flip_prob = 2*p/3

        flip_mask = np.random.rand(nshots) < flip_prob
        noisy[flip_mask, q] *= -1

    return noisy

def apply_global_depolarizing_noise_end_of_circuit(samples, p):
    """
    Applies global depolarizing noise to measurement outcomes via post-processing.
    
    Parameters:
    - samples: array of shape (nshots, nqubits) with entries in {-1, +1}
    - p: global depolarizing noise probability
    
    Returns:
    - noisy_samples: modified version of samples
    """
    
    noisy = samples.copy()
    
    for i in range(noisy.shape[0]):
        if random.random() < p:
            # Replace with a random bitstring in {-1,1}
            for j in range(noisy.shape[1]):
                noisy[i,j] = -2*random.randint(0, 1)+1
    
    return noisy

# Functions to compute effect of local and global depolarizing noise models 
# at the end of the circuit only on the energy estimation analytically

def energy_GDN_bias(energy_unbiased, p_GDN, offset):
    energy_GDN = (1-p_GDN)*(energy_unbiased - offset) + offset
    return energy_GDN

def energy_LDN_bias(p_array, molecule_name, basis_set, mapping_name, folder_Hamiltonians, state):
    len_name = len(molecule_name) + len(basis_set) + 1

    # open folder where the Hamiltonians of various encodings are stored
    available_folders = os.listdir(folder_Hamiltonians)
    folder_name = None
    for folder in available_folders:
        if folder[:len_name] == molecule_name + "_" + basis_set:
            folder_name = folder
    
    # open file where the Hamiltonian of the specified encoding is stored
    available_files = os.listdir(folder_Hamiltonians + folder_name)
    file_name = None
    file_energy = None
    for file in available_files:
        if file[:2] == mapping_name[:2].lower() and file.find("grouped") == -1:
            file_name = file
        elif file == "ExactEnergy.txt":
            file_energy = file
    
    # extract Pauli list from file
    full_file_name = os.path.join(folder_Hamiltonians,folder_name,file_name)
    data = np.loadtxt(full_file_name,dtype=object)
    paulis, weights = data[::2].astype(str), data[1::2].astype(complex).real
    
    # Generate Hamiltonian class
    H = Hamiltonian(weights, paulis)
    
    LDN_factors_array = H.LDN_factor_calculator(p_array)
    
    H_LDN = H.SummedOp_LDN(LDN_factors_array).to_matrix(sparse=True)
    state = np.array(state, dtype=complex).reshape((-1,))
    energy_LDN = np.vdot(state, H_LDN.dot(state))

    return energy_LDN
