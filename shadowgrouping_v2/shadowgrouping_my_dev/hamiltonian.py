import numpy as np, os
from scipy.sparse.linalg import eigsh
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp

char_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

def apply_Hamiltonian_to_state(state, molecule_name, basis_set, mapping_name, folder_Hamiltonians):
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
    for file in available_files:
        if file[:2] == mapping_name[:2].lower() and file.find("grouped") == -1:
            file_name = file
    
    # extract Pauli list from file
    full_file_name = os.path.join(folder_Hamiltonians,folder_name,file_name)
    data = np.loadtxt(full_file_name,dtype=object)
    paulis, weights = data[::2].astype(str), data[1::2].astype(complex).real
    
    # Generate Hamiltonian class
    H = Hamiltonian(weights, paulis)
        
    H_sparse = H.SummedOp().to_matrix(sparse=True)
    state = np.array(state, dtype=complex).reshape((-1,))
    H_dot_state = H_sparse.dot(state)

    return H_dot_state

def generate_arbitrary_linear_combination_given_basis(basis, weight_on_first_state=None):
    """
    Generate a random linear combination of basis states (columns of `basis`).

    Parameters
    ----------
    basis : np.ndarray, shape (dim, k)
        Orthonormal basis vectors as columns (e.g. eigenstates).
    weight_on_first_state : float or None
        If None, pick a random superposition (uniform on complex unit sphere).
        If float in [0,1], fix the magnitude of the coefficient for the first
        basis state to this value, and distribute the remaining weight randomly
        across the other states.

    Returns
    -------
    final_state : np.ndarray, shape (dim,)
        Normalized state vector.
    """
    num_basis_states = basis.shape[1]

    if weight_on_first_state is None:
        # Uniformly random state on the complex unit sphere
        coeffs = np.random.randn(num_basis_states) + 1j*np.random.randn(num_basis_states)
        coeffs /= np.linalg.norm(coeffs)
    else:
        if not (0 <= weight_on_first_state <= 1):
            raise ValueError("weight_on_first_state must be in [0,1]")
        coeffs = np.zeros(num_basis_states, dtype=complex)
        coeffs[0] = np.sqrt(weight_on_first_state)  # real, positive weight on first state
        other_coeffs = np.random.randn(num_basis_states-1) + 1j*np.random.randn(num_basis_states-1)
        other_coeffs /= np.linalg.norm(other_coeffs)
        other_coeffs *= np.sqrt(1 - weight_on_first_state)
        coeffs[1:] = other_coeffs

    final_state = basis @ coeffs
    # Final normalization (just in case of rounding error)
    final_state /= np.linalg.norm(final_state)
    return final_state

def generate_random_state_possibly_in_right_symmetry_subspace(n, N_up_down=None, seed=None):
    """
    Generate a random normalized statevector of size 2^n.

    Parameters
    ----------
    n : int
        Number of qubits.
    N_up_down : tuple (N_up, N_down) or None
        If None, return a Haar-random state in the full Hilbert space.
        If tuple, restrict amplitudes to basis states with Hamming weight
        N_up in the first n/2 qubits and N_down in the last n/2 qubits.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    state : np.ndarray, shape (2^n,)
        Normalized random complex statevector.
    """
    rng = np.random.default_rng(seed)
    dim = 2**n
    state = np.zeros(dim, dtype=np.complex128)

    if N_up_down is None:
        # Full Haar-random state (Gaussian then normalize)
        vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        state = vec / np.linalg.norm(vec)
    else:
        N_up, N_down = N_up_down
        if n % 2 != 0:
            raise ValueError("n must be even when using N_up_down constraints.")
        half = n // 2

        support_indices = []

        # All ways to place N_up ones in first half
        for ups in combinations(range(half), N_up):
            # All ways to place N_down ones in second half
            for downs in combinations(range(half, n), N_down):
                ones = set(ups) | set(downs)
                # Build integer index from bitstring
                idx = sum(1 << (n - 1 - q) for q in ones)
                support_indices.append(idx)

        if not support_indices:
            raise ValueError("No basis states satisfy the given (N_up, N_down).")

        vec = rng.normal(size=len(support_indices)) + 1j * rng.normal(size=len(support_indices))
        vec /= np.linalg.norm(vec)
        state[support_indices] = vec

    return state

def get_energy_given_state(state, molecule_name, basis_set, mapping_name, folder_Hamiltonians):
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
    for file in available_files:
        if file[:2] == mapping_name[:2].lower() and file.find("grouped") == -1:
            file_name = file
    
    # extract Pauli list from file
    full_file_name = os.path.join(folder_Hamiltonians,folder_name,file_name)
    data = np.loadtxt(full_file_name,dtype=object)
    paulis, weights = data[::2].astype(str), data[1::2].astype(complex).real
    
    # Generate Hamiltonian class
    H = Hamiltonian(weights, paulis)
        
    H_sparse = H.SummedOp().to_matrix(sparse=True)
    state = np.array(state, dtype=complex).reshape((-1,))
    energy = np.real(np.vdot(state, H_sparse.dot(state)))

    return energy

class Hamiltonian:
    """ Helper class to turn a list of Pauli operators with accompanying weights 
        into a (sparse) Hamiltonian and diagonalize it.
        Updated for Qiskit 1.0+ compatibility.
        observables must be provided as a list of strings 
        (e.g., ['ZIII', 'XXII', 'YYZZ', ...]), so its format
        must be changed with respect to the numpy.array of integers
        provided by load_pauli_list.
    """

    def __init__(self, weights, observables):
        self.weights = weights
        self.observables = observables

    def _build_operator(self, weights):
        """Helper to construct the SparsePauliOp efficiently."""
        return SparsePauliOp(self.observables, coeffs=weights)

    def SummedOp(self):
        """Returns the Hamiltonian as a SparsePauliOp."""
        return self._build_operator(self.weights)
    
    def LDN_factor_calculator(self, p_array):
        attenuation = [1 - 4/3 * p_array[i] for i in range(len(self.observables[0]))]
        LDN_factors_array = []
        
        for i in range(len(self.observables)):
            factors = []
            for j in range(len(self.observables[0])):
                pauli_char = self.observables[i][j]
                factors.append(attenuation[j] if pauli_char != 'I' else 1)
            LDN_factor = np.prod(factors)
            LDN_factors_array.append(LDN_factor)
        
        return LDN_factors_array
    
    def SummedOp_LDN(self, LDN_factors):             
        # Create new weights: original_weight * LDN_factor
        # We use np.multiply for element-wise multiplication if inputs are arrays,
        # otherwise a list comprehension works safely for lists.
        new_weights = [w * ldn for w, ldn in zip(self.weights, LDN_factors)]
        return self._build_operator(new_weights)

    def ground(self, sparse=True):
        operator = self.SummedOp()
        
        if not sparse:
            # to_matrix() returns a dense numpy array
            mat = operator.to_matrix()
            evalues, evectors = np.linalg.eigh(mat)
        else:
            # to_matrix(sparse=True) returns a scipy csr_matrix
            # replacing the old .to_spmatrix() method
            mat = operator.to_matrix(sparse=True)
            evalues, evectors = eigsh(mat, k=1, which='SA')
            
        index = np.argmin(evalues)
        ground_energy = evalues[index]
        ground_state = evectors[:, index]
        return ground_energy, ground_state
    
    def k_low_lying(self, k=1, sparse=True, shift_trick_value=None):
        operator = self.SummedOp()
        
        if not sparse:
            mat = operator.to_matrix()
            evalues, evectors = np.linalg.eigh(mat)
            w_low = evalues[:k]
            v_low = evectors[:, :k]
        else:
            mat = operator.to_matrix(sparse=True)
            if shift_trick_value is None:
                w_low, v_low = eigsh(mat, k=k, which='SA')
            else:
                w_low, v_low = eigsh(mat, k=k, sigma=shift_trick_value, which='LM')
            
            # Sort eigenvalues as eigsh does not guarantee order
            idx = w_low.argsort()
            w_low = w_low[idx]
            v_low = v_low[:, idx]

        return w_low, v_low
    
def load_k_low_lying_states(folder_hamiltonian, molecule_name,basis_name, encoding, 
                            k=1, shift_trick_value=None, sparse=True, check_energy=False):
    """
    Compute the k lowest-energy eigenstates and eigenvalues.
     
    Parameters
    ----------
    k : int
    Number of lowest-energy eigenstates to calculate (default=1).
    sparse : bool
    If False, use dense diagonalization (np.linalg.eigh).
    If True, use sparse methods (scipy.sparse.linalg.eigsh).
    shift_trick_value : float or None
    If None, use ARPACK 'SA' to find algebraically smallest eigenvalues.
    If set, use shift-invert with sigma=shift_trick_value to target eigenvalues near sigma.
     
    Returns
    -------
    w_low : np.ndarray, shape (k,)
    The k lowest eigenvalues.
    v_low : np.ndarray, shape (n, k)
    The corresponding eigenvectors (columns).
    """

    # ensure the folder exists
    assert os.path.isdir(folder_hamiltonian), f"Path '{folder_hamiltonian}' does not exist or is not a directory."

    # find folder matching molecule + basis name
    available_folders = os.listdir(folder_hamiltonian)
    prefix = f"{molecule_name}_{basis_name}"
    folder_name = None
    for folder in available_folders:
        if folder.startswith(prefix):
            folder_name = folder
            break

    assert folder_name is not None, f"File not found for molecule '{molecule_name}' and basis set '{basis_name}'."
    full_folder_path = os.path.join(folder_hamiltonian, folder_name)

    # look for encoding and energy file
    available_files = os.listdir(full_folder_path)
    file_name = None
    for file in available_files:
        if file.lower().startswith(encoding[:2].lower()) and "grouped" not in file:
            file_name = file

    assert file_name is not None, f"File not found for encoding '{encoding}'."
    
    # extract Pauli list from file
    full_file_name = os.path.join(folder_hamiltonian,folder_name,file_name)
    data = np.loadtxt(full_file_name,dtype=object)
    paulis, weights = data[::2].astype(str), data[1::2].astype(complex).real
    
    H = Hamiltonian(weights,paulis)
    w_low, v_low = H.k_low_lying(k=k, sparse=sparse, shift_trick_value=shift_trick_value)

    return w_low, v_low

def load_pauli_list(folder_hamiltonian,molecule_name,basis_name,encoding,
                    verbose=False,sparse=False,diagonalize=True,check_energy=False):
    """ Loads the Pauli operators and the corresponding ground-state energy from the files of
        https://github.com/charleshadfield/adaptiveshadows
        Requires the name of the folder where all the Hamiltonians are stored together with the selection of the
        molecule, basis set and encoding. If verbose is set to True, some elements of the Pauli list are printed to console.
        If sparse is set to True, carries out the numerical diagonalization on a sparse form of the Hamiltonian.
        If diagonalize is set to False, only returns the Pauli decomposition from file and sets all other return values to None.
        
        Returns the observables, their respective weight, the offset energy and the exact ground-state energy.
    """

    # ensure the folder exists
    assert os.path.isdir(folder_hamiltonian), f"Path '{folder_hamiltonian}' does not exist or is not a directory."

    # find folder matching molecule + basis name
    available_folders = os.listdir(folder_hamiltonian)
    prefix = f"{molecule_name}_{basis_name}"
    folder_name = None
    for folder in available_folders:
        if folder.startswith(prefix):
            folder_name = folder
            break

    assert folder_name is not None, f"File not found for molecule '{molecule_name}' and basis set '{basis_name}'."
    full_folder_path = os.path.join(folder_hamiltonian, folder_name)

    # look for encoding and energy file
    available_files = os.listdir(full_folder_path)
    file_name = None
    file_energy = None
    for file in available_files:
        if file.lower().startswith(encoding[:2].lower()) and "grouped" not in file:
            file_name = file
        elif file == "ExactEnergy.txt":
            file_energy = file

    assert file_name is not None, f"File not found for encoding '{encoding}'."
    if check_energy:
        assert file_energy is not None, "File not found for ground-state energy."
    
    if diagonalize and check_energy:
        # read ground-state energy from file
        full_file_name = os.path.join(folder_hamiltonian,folder_name,file_energy)
        with open(full_file_name,"r") as f:
            E_GS = float(f.readline().strip().split()[-1])
    else:
        E_numerics = None
        state = None
    
    # extract Pauli list from file
    full_file_name = os.path.join(folder_hamiltonian,folder_name,file_name)
    data = np.loadtxt(full_file_name,dtype=object)
    paulis, weights = data[::2].astype(str), data[1::2].astype(complex).real
    
    if diagonalize:
        # use Pauli list to create Hamiltonian and diagonalize it afterwards to obtain ground-state
        H = Hamiltonian(weights,paulis)
        E_numerics, state = H.ground(sparse=sparse)
        if check_energy:
            if abs(E_GS-E_numerics) >= 1e-6:
                print("Warning: Recorded value for the energy deviates significantly from numerical estimate!")
                print("Recorded:",E_GS)
                print("Calculated:",E_numerics)
    
    # Pauli item "III...II" in list should correspond to energy offset
    ind = -1
    identity = "I"*len(paulis[0])
    for i,p in enumerate(paulis):
        if p == identity:
            ind = i
            break
    if ind == -1:
        offset = 0
        obs = paulis
        w = weights
    else:
        offset = weights[ind]
        # erase the corresponding entry in paulis and weights
        obs = np.delete(paulis,ind)
        w = np.delete(weights,ind)
        assert len(obs) == len(paulis) - 1, "Error in line eraser."
        assert len(obs) == len(w), "Both arrays are not of equal length anymore."
    
    # print some to console
    if verbose:
        print("Offset","\t\t",offset)
        for i, (p, we) in enumerate(zip(obs, w)):
            print(p,"\t",we)
            if i == 9:
                print("\t","...")
                break
    
    # convert string characters to integers
    observables = np.array([[char_to_int[c] for c in o] for o in obs],dtype=int)
    
    return observables, w, offset, E_numerics, state

def load_thermal_state(beta,folder_hamiltonian,molecule_name,basis_name,encoding,verbose=False):
    """ Calculates the thermal state at a given inverse temperature <beta> for the electronic structure problem.
        Loads the Pauli operators from the files of https://github.com/charleshadfield/adaptiveshadows
        Requires the name of the folder where all the Hamiltonians are stored together with the selection of the
        molecule, basis set and encoding. If verbose is set to True, some elements of the Pauli list are printed to console.
        If diagonalize is set to False, only returns the Pauli decomposition from file and sets all other return values to None.
        
        Returns the observables, their respective weight, the offset energy and the thermal energy with its corresponding density matrix.
    """
    # match basis set naming scheme to saved files
    basis_matcher = {"sto3g": "STO3g", "6-31g": "6-31G"}
    basis_name = basis_matcher[basis_name]
    
    len_name = len(molecule_name) + len(basis_name) + 1 # for underscore char in naming scheme
    
    # open folder where the Hamiltonians of various encodings are stored
    available_folders = os.listdir(folder_hamiltonian)
    folder_name = None
    for folder in available_folders:
        if folder[:len_name] == molecule_name + "_" + basis_name:
            folder_name = folder
    assert folder_name is not None, "File not found for molecule {} and basis set {}".format(molecule_name,basis_name)
    
    # open file where the Hamiltonian of the specified encoding is stored
    available_files = os.listdir(folder_hamiltonian + folder_name)
    file_name = None
    for file in available_files:
        if file[:2] == encoding[:2].lower() and file.find("grouped") == -1:
            file_name = file
    assert file_name   is not None, "File not found for encoding {}".format(encoding)
    
    # extract Pauli list from file
    full_file_name = os.path.join(folder_hamiltonian,folder_name,file_name)
    data = np.loadtxt(full_file_name,dtype=object)
    paulis, weights = data[::2].astype(str), data[1::2].astype(complex).real
    
    # use Pauli list to create Hamiltonian and diagonalize it afterwards to eigenstates and energies
    # we can set the offset to zero for this because it does not affect the thermal state at all
    inds = paulis != "I"*len(paulis[0])
    H = Hamiltonian(weights[inds],paulis[inds])
    mat = H.SummedOp().to_matrix()
    vals, states =  np.linalg.eigh(mat)
    states = states.real # the eigenstates are real-valued because the Hamiltonian is as well
    beta *= -1
    probs = np.exp(beta*vals - beta*vals[0])
    # adding a constant to all exponents does not alter the probabilities
    # because the argument is negative, we substract the smallest value to make the calculation more stable
    probs /= np.sum(probs)
    E_numerics = np.sum(probs*vals)
    rho = np.einsum("i,ji,ki",probs,states,states)
    assert abs(E_numerics - np.trace(mat@rho)) < 1e-3, "wrong einstein-summation"
    
    # Pauli item "III...II" in list should correspond to energy offset
    ind = -1
    identity = "I"*len(paulis[0])
    for i,p in enumerate(paulis):
        if p == identity:
            ind = i
            break
    if ind == -1:
        offset = 0
        obs = paulis
        w = weights
    else:
        offset = weights[ind]
        # erase the corresponding entry in paulis and weights
        obs = np.delete(paulis,ind)
        w = np.delete(weights,ind)
        assert len(obs) == len(paulis) - 1, "Error in line eraser."
        assert len(obs) == len(w), "Both arrays are not of equal length anymore."
    
    # print some to console
    if verbose:
        print("Offset","\t\t",offset)
        for i, (p, we) in enumerate(zip(obs, w)):
            print(p,"\t",we)
            if i == 9:
                print("\t","...")
                break
    
    # convert string characters to integers
    observables = np.array([[char_to_int[c] for c in o] for o in obs],dtype=int)
    
    return observables, w, offset, E_numerics, rho
