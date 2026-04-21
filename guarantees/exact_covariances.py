import numpy as np, numba
from joblib import Parallel, delayed

@numba.njit
def apply_pauli_string_numba(state, pauli_str):
    """
    Apply an n-qubit Pauli string to a statevector.

    pauli_str: 1D array of length n with entries in {0,1,2,3} = {I,X,Y,Z}
    state: complex128 array of length 2^n
    """
    n = len(pauli_str)
    size = state.shape[0]
    out = np.zeros_like(state)
    for idx in range(size):   # plain range (avoids parfor rewrite issues)
        amp = state[idx]
        new_idx = idx
        phase = 1.0 + 0j
        for q in range(n):
            p = pauli_str[n - 1 - q]  # pauli_str[0] acts on leftmost qubit
            bit = (idx >> q) & 1
            if p == 1:  # X
                new_idx ^= (1 << q)
            elif p == 2:  # Y
                new_idx ^= (1 << q)
                phase *= (1j if bit == 0 else -1j)
            elif p == 3:  # Z
                phase *= (1 if bit == 0 else -1)
        out[new_idx] += phase * amp
    return out

@numba.njit
def expectation_numba(state, pauli_str):
    """Compute ⟨ψ|P|ψ⟩ for a single Pauli string P."""
    phi = apply_pauli_string_numba(state, pauli_str)
    val = 0.0 + 0j
    for k in range(state.shape[0]):
        val += np.conj(state[k]) * phi[k]
    return val

@numba.njit
def multiply_pauli_strings(p1, p2):
    """
    Multiply two Pauli strings (arrays of {0,1,2,3} for {I,X,Y,Z}).
    Returns (resulting_pauli, phase) with phase in {1, -1, 1j, -1j}.
    """
    n = len(p1)
    out = np.empty(n, dtype=np.int8)
    phase = 1 + 0j
    for i in range(n):
        a, b = p1[i], p2[i]
        if a == 0:
            out[i] = b
        elif b == 0:
            out[i] = a
        elif a == b:
            # X*X = I, Y*Y = I, Z*Z = I  (no extra phase)
            out[i] = 0
            # phase unchanged
        else:
            # distinct non-identity Pauli multiplications
            if a == 1 and b == 2:   # X*Y = iZ
                out[i] = 3; phase *= 1j
            elif a == 2 and b == 1: # Y*X = -iZ
                out[i] = 3; phase *= -1j
            elif a == 1 and b == 3: # X*Z = -iY
                out[i] = 2; phase *= -1j
            elif a == 3 and b == 1: # Z*X = iY
                out[i] = 2; phase *= 1j
            elif a == 2 and b == 3: # Y*Z = iX
                out[i] = 1; phase *= 1j
            elif a == 3 and b == 2: # Z*Y = -iX
                out[i] = 1; phase *= -1j
            else:
                out[i] = 0
    return out, phase

def covariance_matrix(state, paulis, n_jobs=-1, verbose=0):
    """
    Compute the exact (generally complex) covariance matrix C of Pauli observables,
    keeping operator order:
        C_ij = ⟨O_i O_j⟩ - ⟨O_i⟩ ⟨O_j⟩
    so that C is Hermitian with C_ji = conj(C_ij).

    Parameters
    ----------
    state : np.ndarray, shape (2^n,), dtype=complex128
        Statevector |ψ⟩.
    paulis : np.ndarray, shape (M,n), dtype=int8/int
        Each row is an n-qubit Pauli string with entries {0,1,2,3} = {I,X,Y,Z}.
    n_jobs : int
        Number of parallel workers for joblib (use -1 for all cores).
    verbose : int
        Joblib verbosity.

    Returns
    -------
    C : np.ndarray, shape (M,M), dtype=complex128
        Hermitian covariance matrix.
    """
    M, _ = paulis.shape
    C = np.zeros((M, M), dtype=np.complex128)

    # Cache ⟨O_i⟩ in parallel
    exp_cache = np.array(
        Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(expectation_numba)(state, paulis[i]) for i in range(M)
        ),
        dtype=np.complex128
    )

    def compute_upper_entry(i, j):
        # Compute ⟨O_i O_j⟩
        p_ij, phase_ij = multiply_pauli_strings(paulis[i], paulis[j])
        exp_ij = expectation_numba(state, p_ij)
        cov_ij = phase_ij * exp_ij - exp_cache[i] * exp_cache[j]
        return i, j, cov_ij

    # Fill upper triangle (including diagonal)
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_upper_entry)(i, j) for i in range(M) for j in range(i, M)
    )

    for i, j, val in results:
        C[i, j] = val
        if j != i:
            C[j, i] = np.conj(val)  # Hermitian completion

    return C
