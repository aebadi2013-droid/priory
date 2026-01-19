import numpy as np, random, os
from shadowgrouping_v2.helper_functions import int_to_bitlist
from shadowgrouping_v2.hamiltonian import Hamiltonian

# Functions to apply readout noise models to sampling outcomes

def apply_stochastic_bit_flips(samples, p0to1, p1to0, correlated_groups=None, p_corr=None, seed=None):
    """
    samples: (Nshots, nqubits) array of 0/1
    p0to1, p1to0: probabilities for uncorrelated bit flips, one for each qubit
    correlated_groups: list of lists of qubit indices to flip together (groups can be of varying size)
    p_corr: probabilities of flipping groups, one for each group
    """
    rng = np.random.default_rng(seed)
    noisy = samples.copy()

    # Uncorrelated flips
    flips = rng.random(noisy.shape)
    mask01 = (noisy == 0) & (flips < p0to1)
    mask10 = (noisy == 1) & (flips < p1to0)
    noisy[mask01] = 1
    noisy[mask10] = 0

    # Correlated flips
    if correlated_groups is not None:
        assert p_corr is not None, "Probabilities of correlated bit-flips mist be provided."
        for g, group in enumerate(correlated_groups):
            flip_mask = rng.random(noisy.shape[0]) < p_corr[g]  # one mask per shot
            for i, do_flip in enumerate(flip_mask):
                if do_flip:
                    for q in group:
                        noisy[i, q] ^= 1  # flip bit

    return noisy

def apply_assignment_matrix(samples, S_matrix_dict, C_matrix_dict, A_row_col, seed=None):
    """
    Application of assignment matrix (A-matrix) to noiseless samples to produce noisy ones.
    
    Parameters:
    - samples: (Nshots, nqubits) array of 0/1 noiseless outcomes
    - S_matrix_array: dictionary of single-qubit (2x2) assignment matrices only for the
                      measured qubit, with keys corresponding to indices of columns of
                      samples matrix associated with the same qubit
    - C_matrix_array: dictionary of two-qubit (4x4) assignment matrices only for pairs of
                      measured qubit, with keys corresponding to tuples of indices of
                      columns of samples matrix associated with the two qubits
    - A_row_col: function that returns matrix element A[row, col] of assignment matrix
    - seed: RNG seed (optional)
    
    Returns:
    - noisy_samples: (Nshots, nqubits)
    """
    num_qubits = samples.shape[1]
    rng = np.random.default_rng(seed)
    Nshots = samples.shape[0]
    
    noisy_indices = np.zeros(Nshots, dtype=int)
    
    for i in range(Nshots):
        col = samples[i,:]
        probs = []
        for row in range(2**num_qubits):
            row = int_to_bitlist(row, width=num_qubits)
            probs.append(A_row_col(row,col,S_matrix_dict,C_matrix_dict))
        cdf = np.cumsum(np.array(probs))
        r = rng.random()
        noisy_index = np.searchsorted(cdf, r)
        noisy_indices[i] = noisy_index
    
    # Convert indices to bitstrings
    noisy_samples = ((noisy_indices[:, None] & (1 << np.arange(num_qubits)[::-1])) > 0).astype(int)
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

def A_row_col_T(row, col, S_matrix_dict, C_matrix_dict=None):
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

# Functions to apply local and global depolarizing noise models at the end of the circuit only

def apply_local_depolarizing_noise_end_of_circuit(samples, meas_basis, p_array):
    """
    Applies local depolarizing noise to measurement outcomes via post-processing.
    
    Parameters:
    - samples: array of shape (nshots, nqubits) with entries in {-1, +1}
    - meas_basis: Pauli string of length nqubits (e.g., "XZIY")
    - p_array: list or array of depolarizing noise probabilities, one per qubit
    
    Returns:
    - noisy_samples: modified version of samples
    """
    nshots, nqubits = samples.shape
    noisy = samples.copy()
    
    for q in range(nqubits):
        basis = meas_basis[q]
        p = p_array[q]

        # Effective flip probability depends on basis and channel
        if basis == "I":
            continue  # skip unmeasured qubits
        elif basis == "Z":
            # Only X, Y flips affect Z measurement → total flip prob = 2p/3
            flip_prob = 2*p/3
        elif basis == "X":
            # Only Y, Z flips affect X measurement
            flip_prob = 2*p/3
        elif basis == "Y":
            # Only X, Z flips affect Y measurement
            flip_prob = 2*p/3
        else:
            raise ValueError(f"Invalid Pauli basis: {basis}")

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
    H.p_array = p_array
    # print(H.p_array)
    
    H.LDN_factor_calculator()
    # print(H.LDN_factors)
    
    H_LDN = H.SummedOp_LDN().to_spmatrix()
    state = np.array(state, dtype=complex).reshape((-1,))
    energy_LDN = np.vdot(state, H_LDN.dot(state))

    return energy_LDN
