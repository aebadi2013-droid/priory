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

def Pauli_expectation_values(
    state,
    paulis,
    n_jobs=-1,
    verbose=0,
    imag_tolerance=1e-12,
):
    """
    Compute the exact expectation value of every Pauli observable:

        exp_values[i] = <psi|O_i|psi>.

    Since each O_i is a Hermitian Pauli string, its expectation value is real.
    Small imaginary parts caused by floating-point roundoff are discarded after
    checking that they lie below the specified tolerance.

    Parameters
    ----------
    state : np.ndarray, shape (2^n,), dtype=complex128
        Statevector |psi>.

    paulis : np.ndarray, shape (M, n), dtype=int8/int
        Each row is an n-qubit Pauli string with entries

            {0, 1, 2, 3} = {I, X, Y, Z}.

        The first column acts on the leftmost qubit, consistently with
        `apply_pauli_string_numba`.

    n_jobs : int, optional
        Number of parallel joblib workers. Use -1 to use all available cores.

    verbose : int, optional
        Joblib verbosity level.

    imag_tolerance : float, optional
        Maximum tolerated absolute imaginary part of an expectation value.
        Imaginary components below this threshold are treated as numerical
        roundoff.

    Returns
    -------
    exp_values : np.ndarray, shape (M,), dtype=float64
        Exact expectation values of the Pauli strings with respect to `state`.
    """
    state = np.asarray(state, dtype=np.complex128)
    paulis = np.asarray(paulis)

    if state.ndim != 1:
        raise ValueError("state must be a one-dimensional statevector")

    if paulis.ndim != 2:
        raise ValueError("paulis must be a two-dimensional array of shape (M, n)")

    M, n = paulis.shape

    if state.shape[0] != 2**n:
        raise ValueError(
            f"The statevector has length {state.shape[0]}, but {n} qubits "
            f"require a statevector of length {2**n}."
        )

    if np.any((paulis < 0) | (paulis > 3)):
        raise ValueError(
            "All entries of paulis must belong to {0, 1, 2, 3} = {I, X, Y, Z}."
        )

    # A contiguous int8 representation is convenient for the Numba routine.
    paulis = np.ascontiguousarray(paulis, dtype=np.int8)
    state = np.ascontiguousarray(state, dtype=np.complex128)

    exp_values_complex = np.asarray(
        Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(expectation_numba)(state, paulis[i])
            for i in range(M)
        ),
        dtype=np.complex128,
    )

    max_imaginary_part = (
        np.max(np.abs(exp_values_complex.imag)) if M > 0 else 0.0
    )

    if max_imaginary_part > imag_tolerance:
        raise ValueError(
            "A Pauli expectation value has a non-negligible imaginary part: "
            f"maximum |Im(<P>)| = {max_imaginary_part:.3e}. "
            "This may indicate an invalid Pauli string or an implementation issue."
        )

    return np.ascontiguousarray(exp_values_complex.real, dtype=np.float64)