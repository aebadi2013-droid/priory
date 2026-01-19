import numpy as np
import numbers
from numba import njit

char_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
int_to_char = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

def hit_by(O,P):
    """ Returns whether o is hit by p """
    for o,p in zip(O,P):
        if not (o==0 or p==0 or o==p):
            return False
    return True

@njit
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

def hit_by_standalone(O, P, pedantic=True):
    """
    Returns whether O is hit by P, i.e., whether they commute qubitwise.
    
    O and P should be sequences (list or array) of integers in {0, 1, 2, 3},
    representing I, X, Y, Z respectively. Returns True if for every qubit:
    O[i] == 0 or P[i] == 0 or O[i] == P[i].

    Parameters:
    - O, P: Lists or arrays of integers in [0, 3]
    - pedantic (bool): If True, performs full input validation.

    Examples:
        hit_by([0, 2, 0], [1, 2, 3]) == True
        hit_by([1, 2, 3], [3, 1, 2]) == False
    """
    if pedantic:
        if not isinstance(O, (list, np.ndarray)) or not isinstance(P, (list, np.ndarray)):
            raise TypeError("Both O and P must be lists or numpy arrays.")
        if len(O) != len(P):
            raise ValueError("O and P must have the same length.")
        for i, (o, p) in enumerate(zip(O, P)):
            if not (isinstance(o, numbers.Integral) and 0 <= o <= 3):
                raise ValueError(f"O[{i}] = {o} is not an integer in [0, 3].")
            if not (isinstance(p, numbers.Integral) and 0 <= p <= 3):
                raise ValueError(f"P[{i}] = {p} is not an integer in [0, 3].")
                
    for o,p in zip(O,P):
        if not (o==0 or p==0 or o==p):
            return False
    return True

def hit_by_batch(O_batch, P):
    O_batch = np.asarray(O_batch)
    P = np.asarray(P)
    
    # Preallocate result array
    result = np.empty(O_batch.shape[0], dtype=bool)

    for i, O in enumerate(O_batch):
        for o, p in zip(O, P):
            if not (o == 0 or p == 0 or o == p):
                result[i] = False
                break
        else:
            result[i] = True

    return result

@njit
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

def hit_by_batch_standalone(O_batch, P, *, pedantic=True):
    """
    Checks which observables in O_batch are hit by the setting P.
    Uses fast row-wise exit (loop over rows, vectorized inside each row).

    Parameters:
        O_batch : array-like of shape (n_obs, n_qubits)
        P       : array-like of shape (n_qubits,)
        pedantic : bool, default=False
            If True, perform strict validation checks.

    Returns:
        np.ndarray of bool of shape (n_obs,)
    """
    if not isinstance(O_batch, (list, np.ndarray)):
        raise TypeError("O_batch must be a list or numpy array.")
    if not isinstance(P, (list, np.ndarray)):
        raise TypeError("P must be a list or numpy array.")

    O_batch = np.asarray(O_batch)
    P = np.asarray(P)

    if pedantic:
        if O_batch.ndim != 2 or P.ndim != 1:
            raise ValueError("O_batch must be 2D and P must be 1D.")
        if O_batch.shape[1] != P.shape[0]:
            raise ValueError("Mismatch in number of qubits.")
        if not (np.issubdtype(O_batch.dtype, np.integer) and np.issubdtype(P.dtype, np.integer)):
            raise ValueError("All values must be integers.")
        if not ((0 <= O_batch).all() and (O_batch <= 3).all()):
            raise ValueError("All entries in O_batch must be integers in [0, 3].")
        if not ((0 <= P).all() and (P <= 3).all()):
            raise ValueError("All entries in P must be integers in [0, 3].")

    # Preallocate result array
    result = np.empty(O_batch.shape[0], dtype=bool)

    for i, O in enumerate(O_batch):
        for o, p in zip(O, P):
            if not (o == 0 or p == 0 or o == p):
                result[i] = False
                break
        else:
            result[i] = True

    return result

def hit_by_batch_batch(O_batch, P_batch):
    """
    Checks pairwise qubitwise compatibility of each observable with each setting.
    Saves time by skipping full computation when a mismatch is found early.
    
    Parameters:
        O_batch: shape (n_obs, n_qubits)
        P_batch: shape (n_settings, n_qubits)
        
    Returns:
        result: shape (n_obs, n_settings) boolean array
    """
    O_batch = np.asarray(O_batch)
    P_batch = np.asarray(P_batch)

    n_obs, n_qubits = O_batch.shape
    n_settings = P_batch.shape[0]

    # Start with all True (assume all are compatible)
    result = np.ones((n_obs, n_settings), dtype=bool)

    for q in range(n_qubits):
        Oq = O_batch[:, q][:, np.newaxis]  # shape (n_obs, 1)
        Pq = P_batch[:, q][np.newaxis, :]  # shape (1, n_settings)

        # valid if O == 0 or P == 0 or O == P
        is_valid = (Oq == 0) | (Pq == 0) | (Oq == Pq)
        result &= is_valid  # Accumulate

        # Early exit: if any row is already all False, we skip further updates for those
        if np.all(result == False):
            break

    return result

@njit
def hit_by_batch_batch_numba(O_batch, P_batch):
    """
    Numba-accelerated version of hit_by_batch_batch with early exit.
    O_batch: (n_obs, n_qubits)
    P_batch: (n_settings, n_qubits)
    Returns: (n_obs, n_settings) boolean array
    """
    n_obs, n_qubits = O_batch.shape
    n_settings = P_batch.shape[0]
    result = np.empty((n_obs, n_settings), dtype=np.bool_)

    for i in range(n_obs):
        for j in range(n_settings):
            compatible = True
            for q in range(n_qubits):
                o = O_batch[i, q]
                p = P_batch[j, q]
                if not (o == 0 or p == 0 or o == p):
                    compatible = False
                    break
            result[i, j] = compatible

    return result

def hit_by_batch_batch_standalone(O_batch, P_batch, pedantic=True):
    """
    Checks pairwise qubitwise compatibility of each observable with each setting.
    Standalone version with input validation.

    Parameters:
        O_batch: list or np.ndarray of shape (n_obs, n_qubits)
        P_batch: list or np.ndarray of shape (n_settings, n_qubits)

    Returns:
        result: np.ndarray of bools, shape (n_obs, n_settings)
    """
    if pedantic:
        if not isinstance(O_batch, (list, np.ndarray)):
            raise TypeError("O_batch must be a list or numpy array.")
        if not isinstance(P_batch, (list, np.ndarray)):
            raise TypeError("P_batch must be a list or numpy array.")

    O_batch = np.asarray(O_batch)
    P_batch = np.asarray(P_batch)

    if pedantic:
        if O_batch.ndim != 2 or P_batch.ndim != 2:
            raise ValueError("O_batch and P_batch must be 2D arrays.")
        if O_batch.shape[1] != P_batch.shape[1]:
            raise ValueError("Mismatch in number of qubits between O_batch and P_batch.")
        if not np.issubdtype(O_batch.dtype, np.integer) or not np.issubdtype(P_batch.dtype, np.integer):
            raise ValueError("All entries must be integers.")
        if not ((0 <= O_batch).all() and (O_batch <= 3).all() and (0 <= P_batch).all() and (P_batch <= 3).all()):
            raise ValueError("All values must be integers in [0, 3].")

    n_obs, n_qubits = O_batch.shape
    n_settings = P_batch.shape[0]
    result = np.ones((n_obs, n_settings), dtype=bool)

    for q in range(n_qubits):
        Oq = O_batch[:, q][:, np.newaxis]  # shape (n_obs, 1)
        Pq = P_batch[:, q][np.newaxis, :]  # shape (1, n_settings)

        is_valid = (Oq == 0) | (Pq == 0) | (Oq == Pq)
        result &= is_valid

        if not result.any():
            break

    return result

def index_to_string(index_list):
    index_array = np.asarray(index_list, dtype=int)
    pauli_chars = np.array(['I', 'X', 'Y', 'Z'])
    return ''.join(pauli_chars[index_array])

def index_to_string_standalone(index_list, pedantic=True):
    """
    Converts a list of Pauli indices (0 -> I, 1 -> X, 2 -> Y, 3 -> Z)
    into the corresponding Pauli string.
    """
    if pedantic:
        if not np.issubdtype(np.array(index_list).dtype, np.integer):
            raise ValueError("All entries must be integers.")
    
    index_array = np.asarray(index_list, dtype=int)

    if pedantic:
        if index_array.ndim != 1:
            raise ValueError("index_list must be 1-dimensional.")
        if np.any((index_array < 0) | (index_array > 3)):
            raise ValueError("All elements must be in {0,1,2,3}.")

    # Vectorized mapping using numpy array and list comprehension
    pauli_chars = np.array(['I', 'X', 'Y', 'Z'])
    return ''.join(pauli_chars[index_array])

def int_to_bitlist(n, width=None):
    """
    Converts integer n to a list of bits (MSB to LSB).
    
    Parameters:
    - n: integer
    - width: total number of bits to pad to (optional)
    
    Returns:
    - List of bits [b_{MSB}, ..., b_{LSB}]
    """
    if width is None:
        width = n.bit_length() or 1  # handle n = 0
    return [(n >> i) & 1 for i in reversed(range(width))]

def prepare_settings_for_numba(settings_dict):
    """
    Convert a settings_dict into settings_int and settings_reps for Numba-compatible processing.

    Parameters:
    - settings_dict: dict
        Dictionary where keys are Pauli strings (e.g., 'IXYZ') and values are repetition counts.

    Returns:
    - settings_int: list of numpy arrays
        Each array is a sequence of integers in {0,1,2,3}, representing a Pauli string.
    - settings_reps: numpy array
        Array of integers representing the number of times each setting appears.
    """
    char_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

    settings_int = []
    settings_reps = []

    for pauli_str, count in settings_dict.items():
        setting = np.array([char_to_int[c] for c in pauli_str], dtype=np.int64)
        settings_int.append(setting)
        settings_reps.append(count)

    return np.array(settings_int), np.array(settings_reps, dtype=np.int64)

def sample_obs_from_setting(O, P):
    for o, p in zip(O, P):
        if o != 0 and o != p:
            return False
    return True

@njit
def sample_obs_from_setting_numba(O, P):
    """
    Numba-accelerated version of sample_obs_from_setting.
    Returns True if observable O can be sampled from setting P.
    Assumes O and P are 1D NumPy arrays of the same length with integer values in [0, 3].
    No input validation is performed.
    """
    for i in range(O.shape[0]):
        if O[i] != 0 and O[i] != P[i]:
            return False
    return True

def sample_obs_from_setting_standalone(O, P, pedantic=True):
    """
    Returns whether observable O can be sampled from setting P.
    Sampling is allowed if every nonzero entry in O matches the corresponding entry in P.

    Parameters:
        O : list or np.ndarray of ints in {0,1,2,3}
        P : list or np.ndarray of ints in {0,1,2,3}
        pedantic : bool (default: False)
            If True, performs input validation.

    Returns:
        bool : True if O can be sampled from P, False otherwise.
    """
    if pedantic:
        if not isinstance(O, (list, np.ndarray)) or not isinstance(P, (list, np.ndarray)):
            raise TypeError("Both O and P must be lists or numpy arrays.")
        if len(O) != len(P):
            raise ValueError("O and P must be of the same length.")
        for i, (o, p) in enumerate(zip(O, P)):
            if not (isinstance(o, int) and isinstance(p, int)):
                raise ValueError("All entries must be integers.")
            if not (0 <= o <= 3 and 0 <= p <= 3):
                raise ValueError(f"O[{i}] = {o}, P[{i}] = {p} must be in [0, 3].")

    for o, p in zip(O, P):
        if o != 0 and o != p:
            return False
    return True

def sample_obs_batch_from_setting(O_batch, P):
    O_batch = np.asarray(O_batch)
    P = np.asarray(P)

    n_obs = O_batch.shape[0]
    result = np.empty(n_obs, dtype=bool)

    for i in range(n_obs):
        compatible = True
        for o, p in zip(O_batch[i], P):
            if o != 0 and o != p:
                compatible = False
                break
        result[i] = compatible

    return result

@njit
def sample_obs_batch_from_setting_numba(O_batch, P):
    """
    Numba-accelerated version of sample_obs_batch_from_setting with early exit.
    
    Parameters:
        O_batch : np.ndarray, shape (n_obs, n_qubits)
        P       : np.ndarray, shape (n_qubits,)
        
    Returns:
        result  : np.ndarray of bool, shape (n_obs,)
                  result[i] is True if O_batch[i] is sampleable from P
    """
    n_obs, n_qubits = O_batch.shape
    result = np.empty(n_obs, dtype=np.bool_)

    for i in range(n_obs):
        compatible = True
        for q in range(n_qubits):
            o = O_batch[i, q]
            p = P[q]
            if not (o == 0 or o == p):
                compatible = False
                break
        result[i] = compatible

    return result

def sample_obs_batch_from_setting_standalone(O_batch, P, pedantic=True):
    """
    Efficient early-exit version of sample_obs_batch_from_setting.

    Parameters:
        O_batch : array-like of shape (n_obs, n_qubits)
        P       : array-like of shape (n_qubits,)
        pedantic : bool, optional (default=False)
            If True, perform input validation.

    Returns:
        result : np.ndarray of bool of shape (n_obs,)
                 True if observable is compatible with P, False otherwise.
    """
    O_batch = np.asarray(O_batch)
    P = np.asarray(P)

    if pedantic:
        if O_batch.ndim != 2 or P.ndim != 1:
            raise ValueError("O_batch must be 2D and P must be 1D.")
        if O_batch.shape[1] != P.shape[0]:
            raise ValueError("Mismatch between observable and setting dimensions.")
        if not (np.issubdtype(O_batch.dtype, np.integer) and np.issubdtype(P.dtype, np.integer)):
            raise ValueError("O_batch and P must contain only integers.")
        if not np.all((0 <= O_batch) & (O_batch <= 3)) or not np.all((0 <= P) & (P <= 3)):
            raise ValueError("All entries must be integers in [0, 3].")

    n_obs = O_batch.shape[0]
    result = np.empty(n_obs, dtype=bool)

    for i in range(n_obs):
        compatible = True
        for o, p in zip(O_batch[i], P):
            if o != 0 and o != p:
                compatible = False
                break
        result[i] = compatible

    return result

def sample_obs_batch_from_setting_batch(O_batch, P_batch):
    O_batch = np.asarray(O_batch)
    P_batch = np.asarray(P_batch)

    n_obs, n_qubits = O_batch.shape
    n_set = P_batch.shape[0]
    result = np.ones((n_obs, n_set), dtype=bool)

    for i in range(n_obs):
        for j in range(n_set):
            for q in range(n_qubits):
                o = O_batch[i, q]
                p = P_batch[j, q]
                if o != 0 and o != p:
                    result[i, j] = False
                    break  # Early exit for this (i, j) pair

    return result

@njit
def sample_obs_batch_from_setting_batch_numba(O_batch, P_batch):
    """
    Numba-accelerated version of sample_obs_batch_from_setting_batch.
    Returns True if each observable O[i] can be sampled from each setting P[j].

    Parameters:
        O_batch: np.ndarray of shape (n_obs, n_qubits), values in [0, 3]
        P_batch: np.ndarray of shape (n_set, n_qubits), values in [0, 3]

    Returns:
        result: np.ndarray of bool of shape (n_obs, n_set)
    """
    n_obs, n_qubits = O_batch.shape
    n_set = P_batch.shape[0]
    result = np.ones((n_obs, n_set), dtype=np.bool_)

    for i in range(n_obs):
        for j in range(n_set):
            for q in range(n_qubits):
                o = O_batch[i, q]
                p = P_batch[j, q]
                if o != 0 and o != p:
                    result[i, j] = False
                    break  # early exit for this pair
    return result

def sample_obs_batch_from_setting_batch_standalone(O_batch, P_batch, pedantic=True):
    """
    Optimized version of sample_obs_batch_from_setting_batch with early exit on mismatch.
    
    Parameters:
        O_batch: array-like of shape (n_obs, n_qubits)
        P_batch: array-like of shape (n_set, n_qubits)
        pedantic: if True, perform thorough input checks

    Returns:
        result: np.ndarray of bool with shape (n_obs, n_set)
    """
    if pedantic:
        if not isinstance(O_batch, (list, np.ndarray)):
            raise TypeError("O_batch must be a list or numpy array.")
        if not isinstance(P_batch, (list, np.ndarray)):
            raise TypeError("P_batch must be a list or numpy array.")

    O_batch = np.asarray(O_batch)
    P_batch = np.asarray(P_batch)

    if pedantic:
        if O_batch.ndim != 2 or P_batch.ndim != 2:
            raise ValueError("O_batch and P_batch must be 2D arrays.")
        if O_batch.shape[1] != P_batch.shape[1]:
            raise ValueError("Number of qubits must match.")
        if not (np.issubdtype(O_batch.dtype, np.integer) and np.issubdtype(P_batch.dtype, np.integer)):
            raise ValueError("All entries must be integers.")
        if not ((0 <= O_batch).all() and (O_batch <= 3).all() and
                (0 <= P_batch).all() and (P_batch <= 3).all()):
            raise ValueError("All entries must be integers in [0, 3].")

    n_obs, n_qubits = O_batch.shape
    n_set = P_batch.shape[0]
    result = np.ones((n_obs, n_set), dtype=bool)

    for i in range(n_obs):
        for j in range(n_set):
            for q in range(n_qubits):
                o = O_batch[i, q]
                p = P_batch[j, q]
                if o != 0 and o != p:
                    result[i, j] = False
                    break  # Early exit for this (i, j) pair

    return result
            
def settings_to_dict(settings, settings_dict, settings_buffer, is_adaptive=False, order=None):
    settings = np.asarray(settings)

    unique_settings, inverse_indices, counts = np.unique(
        settings, axis=0, return_inverse=True, return_counts=True
    )

    # Convert to strings once
    paulistrings = [''.join('IXYZ'[x] for x in setting) for setting in unique_settings]

    if is_adaptive:
        if order is None:
            raise ValueError("Since scheme is adaptive, order must be provided as input.")
        for idx, pauli_str in enumerate(paulistrings):
            order[pauli_str] = np.where(inverse_indices == idx)[0]

    for pauli_str, count in zip(paulistrings, counts):
        for diction in (settings_dict, settings_buffer):
            diction[pauli_str] = diction.get(pauli_str, 0) + count
            
def setting_to_obs_form(setting_str):
    return [char_to_int[char] for char in setting_str]

def setting_to_obs_form_standalone(setting_str, pedantic=True):
    """
    Converts a Pauli string into its integer representation.

    'I' -> 0, 'X' -> 1, 'Y' -> 2, 'Z' -> 3

    Parameters:
    - setting_str (str): The input Pauli string (e.g., "XYZ").
    - pedantic (bool): If True, performs strict type and value validation.

    Returns:
    - List[int]: List of integers representing the Pauli operators.
    """

    if pedantic:
        if not isinstance(setting_str, str):
            raise TypeError("Input must be a string.")
        valid_chars = {'I', 'X', 'Y', 'Z'}
        for i, char in enumerate(setting_str):
            if char not in valid_chars:
                raise ValueError(
                    f"Invalid character '{char}' at position {i}. "
                    "Allowed characters are 'I', 'X', 'Y', 'Z'."
                )

    try:
        return [char_to_int[char] for char in setting_str]
    except KeyError as e:
        if pedantic:
            raise
        else:
            raise ValueError(f"Invalid character '{e.args[0]}' in setting string.")
            
def setting_to_str(P):
    return ''.join(int_to_char[p] for p in P)

def setting_to_str_standalone(P, pedantic=True):
    """
    Converts a list of integers representing Pauli operators to a string.

    Mapping:
        0 -> 'I'
        1 -> 'X'
        2 -> 'Y'
        3 -> 'Z'

    Parameters:
    - P (list or np.ndarray): Sequence of integers in {0, 1, 2, 3}.
    - pedantic (bool): If True, perform full type and range validation.

    Returns:
    - str: Pauli string (e.g., [1,2,3] -> 'XYZ').
    """
    int_to_char = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

    if pedantic:
        if not isinstance(P, (list, np.ndarray)):
            raise TypeError("P must be a list or numpy array.")

        for i, p in enumerate(P):
            if not isinstance(p, numbers.Integral) or not (0 <= p <= 3):
                raise ValueError(f"P[{i}] = '{p}' is not an integer in [0, 3].")

    try:
        return ''.join(int_to_char[p] for p in P)
    except KeyError as e:
        if pedantic:
            raise
        else:
            raise ValueError(f"Invalid integer '{e.args[0]}' in input sequence.")