import numpy as np, numbers, hashlib, warnings
from typing import List, Tuple
from qibo import models
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford
from qiskit.synthesis import synth_clifford_full

char_to_int = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}
int_to_char = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}

def bitlist_to_int(bitlist, lsb_first=False):
    """
    Converts a list of bits into an integer.

    Parameters:
    - bitlist: list of bits (0 or 1)
    - lsb_first: if True, interpret bitlist as [LSB, ..., MSB];
                 otherwise as [MSB, ..., LSB] (default)

    Returns:
    - Integer corresponding to the bitlist
    """
    if type(bitlist) == str:
        raise TypeError("bitlist must be a list of bits, not a string.")
    if not is_bitlist(bitlist):
        raise ValueError("bitlist must be a list of bits.")
    if not lsb_first:
        bitlist = bitlist[::-1]
    return sum(b << i for i, b in enumerate(bitlist))

def bootstrap_rmse(energy_array, E_GS, n_boot=10000, ci=0.67):
    """
    Bootstrap the RMSE of energy estimates around a fixed true value E_GS.

    Parameters
    ----------
    energy_array : array-like of shape (n_samples,)
        Empirical energy estimates.
    E_GS : float
        True/target energy (held fixed during resampling).
    n_boot : int, default=10000
        Number of bootstrap resamples.
    ci : float in (0,1), default=0.67
        Two-sided percentile confidence level for RMSE.

    Returns
    -------
    rmse : float
        Point estimate RMSE on the original data.
    se : float
        Bootstrap standard error of RMSE.
    ci_interval : (float, float)
        Percentile confidence interval for RMSE at level `ci`.
    boot_values : np.ndarray
        Array of bootstrap RMSE values of shape (n_boot,).
    """
    
    x = np.asarray(energy_array, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("energy_array must be non-empty.")

    # Point estimate
    errs = x - E_GS
    rmse = np.sqrt(np.mean(errs**2))

    # Helper to compute bootstrap RMSEs for a given number of replicates
    def _boot_chunk(m):
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=(m, n))
        samples = x[idx]
        e = samples - E_GS
        return np.sqrt(np.mean(e**2, axis=1))

    boot_values = _boot_chunk(n_boot)

    se = boot_values.std(ddof=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_values, [alpha, 1 - alpha])

    return rmse, se, (lo, hi), boot_values

def calculate_circuit_metrics(circuit: models.Circuit) -> dict:
    """
    Calculates three metrics for a given Qibo circuit:
    1. Total CNOT Count: Explicit CNOTs + 3 * SWAPs. (CZ counted as 1 CNOT).
    2. Total Circuit Depth: Layers of all gates (1q gates=1 step, CNOT/CZ=1 step, SWAP=3 steps).
    3. CNOT Depth: Layers of only 2-qubit gates (1q gates=0 steps, CNOT/CZ=1 step, SWAP=3 steps).
    
    Args:
        circuit: The Qibo circuit to analyze.
        
    Returns:
        dict: {"cnot_count": int, "total_depth": int, "cnot_depth": int}
    """
    # Initialize counters
    total_cnot_count = 0
    
    # Clocks to track the "finish time" of the last gate on each qubit
    # qubit_clock_total: Tracks depth including single-qubit gates
    qubit_clock_total = [0] * circuit.nqubits
    
    # qubit_clock_cnot: Tracks depth considering only two-qubit interactions
    # (Single qubit gates are transparent/instantaneous in this metric)
    qubit_clock_cnot = [0] * circuit.nqubits
    
    for gate in circuit.queue:
        name = gate.name.lower()
        qubits = gate.qubits
                
        if name in ('m', 'measure', 'barrier'):
            continue
            
        elif name in ('swap',):
            total_cnot_count += 3
            
            q0, q1 = qubits
            
            # Update Total Depth
            # The SWAP starts after the latest of the two qubits is free
            start_total = max(qubit_clock_total[q0], qubit_clock_total[q1])
            finish_total = start_total + 3
            qubit_clock_total[q0] = finish_total
            qubit_clock_total[q1] = finish_total
            
            # Update CNOT Depth
            start_cnot = max(qubit_clock_cnot[q0], qubit_clock_cnot[q1])
            finish_cnot = start_cnot + 3
            qubit_clock_cnot[q0] = finish_cnot
            qubit_clock_cnot[q1] = finish_cnot
            
        elif name in ('cn', 'cnot', 'cx', 'cz'):
            total_cnot_count += 1
            
            q0, q1 = qubits
            
            # Update Total Depth
            start_total = max(qubit_clock_total[q0], qubit_clock_total[q1])
            finish_total = start_total + 1
            qubit_clock_total[q0] = finish_total
            qubit_clock_total[q1] = finish_total
            
            # Update CNOT Depth
            start_cnot = max(qubit_clock_cnot[q0], qubit_clock_cnot[q1])
            finish_cnot = start_cnot + 1
            qubit_clock_cnot[q0] = finish_cnot
            qubit_clock_cnot[q1] = finish_cnot
            
        else:
            # Single Qubit Gates (H, S, X, Y, Z, U3, etc.)
            # Count: 0 CNOTs
            # Total Depth: +1
            # CNOT Depth: +0 (Transparent)
            
            for q in qubits:
                # Update Total Depth
                qubit_clock_total[q] += 1
                
                # CNOT Depth unchanged
                pass

    return {
        "cnot_count": total_cnot_count,
        "total_depth": max(qubit_clock_total) if qubit_clock_total else 0,
        "cnot_depth": max(qubit_clock_cnot) if qubit_clock_cnot else 0
    }

def combine_seed(seed: int, setting_token) -> int:
    h = stable_hash(setting_token, digest_size=8)  # 64-bit
    # fold into uint32 (or use 2**31-1 if a library requires signed)
    return (int(seed) + (h & 0xFFFFFFFF)) & 0xFFFFFFFF

def commute_blockwise(paulis):
    # paulis are strings of equal length
    for i in range(len(paulis)):
        for j in range(i+1, len(paulis)):
            # mismatch parity restricted to this block-length string
            parity = 0
            for a, b in zip(paulis[i], paulis[j]):
                if a != 'I' and b != 'I' and a != b:
                    parity ^= 1
            if parity:
                return False, (paulis[i], paulis[j])
    return True, None

def decode_setting_token(token: bytes) -> np.ndarray:
    """
    Decode a token produced by encode_setting_token.

    In the current Energy_estimator workflow, tokens encode the sorted
    indices of Hamiltonian Pauli strings measured in one round, for QWC,
    FC, and kC alike.

    The physical measurement basis/circuit is reconstructed later from
    these observable indices and the estimator's compat_type.
    """
    return np.frombuffer(token, dtype=np.int32)

def decompose_dense_clifford_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Scans the circuit for 'dense' or 'custom' gates (u, u2, u3, unitary, etc.) 
    and decomposes them into strict discrete Cliffords {H, S, Sdg, X, Y, Z, CX, CZ, SWAP}.
    
    Removes BARRIERS to ensure maximum simplification.
    """
    # 1. Define the Strict Allowed Basis
    # Any gate in this list is preserved as-is.
    allowed_gates = {'h', 's', 'sdg', 'x', 'y', 'z', 'cx', 'cz', 'swap', 'id'}
    
    # 2. Define Directives to KEEP (usually just measurements)
    # We explicitly exclude 'barrier' from here so they get dropped.
    keep_directives = {'measure', 'reset', 'snapshot'}

    new_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    
    for instruction in qc.data:
        gate_name = instruction.operation.name.lower()
        
        # A. If it's a barrier, SKIP it entirely to allow better optimization later
        if gate_name == 'barrier':
            continue
            
        # B. If it's a clean gate or a necessary directive, keep it
        elif gate_name in allowed_gates or gate_name in keep_directives:
            new_qc.append(instruction)
            
        # C. If it's a 'dirty' gate (u, u2, unitary), decompose it
        else:
            try:
                # 1. Extract the mathematical Clifford operator
                cliff = Clifford(instruction.operation)
                
                # 2. Re-synthesize into standard gates
                sub_circ_canonical = synth_clifford_full(cliff)
                
                # 3. Transpile the sub-circuit to ensure strict compliance
                # Note: We use optimization_level=1 to merge simple gates (like H-H)
                # that might result from the synthesis.
                sub_circ_discrete = transpile(
                    sub_circ_canonical, 
                    basis_gates=['h', 's', 'sdg', 'x', 'y', 'z', 'cx', 'cz', 'swap', 'id'], 
                    optimization_level=1
                )
                
                # 4. Append the clean decomposition
                new_qc.compose(sub_circ_discrete, qubits=instruction.qubits, inplace=True)
                
            except Exception as e:
                # If truly non-Clifford, warn and keep.
                print(f"Warning: Could not decompose '{gate_name}' to Discrete Clifford: {e}")
                new_qc.append(instruction)
            
    return new_qc

def draw_circuit(circuit, qc_type='qibo', draw_type='mpl'):
    """
    Draws a quantum circuit using Qiskit's visualizer. Supports both Qiskit and 
    Qibo circuit objects by translating Qibo circuits to Qiskit on the fly.

    Args:
        circuit: The quantum circuit object (qiskit.QuantumCircuit or qibo.models.Circuit).
        qc_type (str): Type of the input circuit. Options: 'qibo', 'qiskit'. 
                       Default is 'qibo'.
        draw_type (str): The drawing style. Options: 'mpl' (matplotlib), 'text', 'latex'. 
                         Default is 'mpl'.

    Returns:
        The drawing object (matplotlib Figure or TextDrawing), which Jupyter 
        will automatically render.
    """
    if qc_type == 'qiskit':
        # For Qiskit, simply return the draw method's output
        return circuit.draw(draw_type)
    
    elif qc_type == 'qibo':
        # Create a new Qiskit circuit with the same number of qubits
        # Note: Qibo's nqubits is an integer; Qiskit needs (n_qubits, n_clbits) 
        # or just n_qubits. We add clbits if measurements exist.
        qiskit_circuit = QuantumCircuit(circuit.nqubits, circuit.nqubits)
        
        # Map Qibo gates to Qiskit gates
        for gate in circuit.queue:
            name = gate.__class__.__name__
            qubits = gate.qubits
            params = gate.parameters
            
            # --- Standard Clifford Gates ---
            if name == "H":
                qiskit_circuit.h(qubits[0])
            elif name == "X":
                qiskit_circuit.x(qubits[0])
            elif name == "Y":
                qiskit_circuit.y(qubits[0])
            elif name == "Z":
                qiskit_circuit.z(qubits[0])
            elif name == "S":
                qiskit_circuit.s(qubits[0])
            elif name == "SDG":  # Essential: Added missing SDG gate
                qiskit_circuit.sdg(qubits[0])
            elif name == "I" or name == "ID": # Essential: Added missing Identity
                qiskit_circuit.id(qubits[0])
                
            # --- Parametric Gates ---
            elif name == "RX":
                qiskit_circuit.rx(params[0], qubits[0])
            elif name == "RY":
                qiskit_circuit.ry(params[0], qubits[0])
            elif name == "RZ":
                qiskit_circuit.rz(params[0], qubits[0])
                
            # --- Two-Qubit Gates ---
            elif name == "CNOT":
                qiskit_circuit.cx(qubits[0], qubits[1])
            elif name == "CZ":
                qiskit_circuit.cz(qubits[0], qubits[1])
            elif name == "SWAP":
                qiskit_circuit.swap(qubits[0], qubits[1])
                
            # --- Measurements ---
            elif name == "M":
                for q in qubits:
                    qiskit_circuit.measure(q, q)
            else:
                raise NotImplementedError(f"Gate {name} not implemented in Qiskit translation.")
        
        # Return the figure object so Jupyter displays it
        return qiskit_circuit.draw(draw_type)
    
    else:
        raise ValueError("Unknown circuit type. Must be either 'qiskit' or 'qibo'.")
        
def encode_setting_token(setting: np.ndarray) -> bytes:
    """
    Return a canonical bytes token for a setting.
    Works for both:
      - QWC setting       :  shape (num_qubits,), entries in {0,1,2,3}
      - FC and kC settings:  shape (k,), entries are observable indices
    """
    arr = np.asarray(setting, dtype=np.int32)
    return arr.tobytes(order="C")

def extract_gates_from_qibo(qibo_circuit: models.Circuit) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    Parses a Qibo circuit and converts it back to the simple 
    (name, args) tuple format used for Pauli tracking.
    """
    gate_list = []
    
    for gate in qibo_circuit.queue:
        name = gate.name.lower()
        qubits = gate.qubits
        
        # Map Qibo names to our internal schema
        if name == 'h':
            gate_list.append(('H', qubits))
        elif name == 's':
            gate_list.append(('S', qubits))
        elif name == 'sdg':
            gate_list.append(('Sdg', qubits))
        elif name == 'cn' or name == 'cnot' or name == 'cx':
            gate_list.append(('CNOT', qubits))
        elif name == 'cz':
            gate_list.append(('CZ', qubits))
        elif name == 'swap':
            gate_list.append(('SWAP', qubits))
        elif name == 'x':
            gate_list.append(('X', qubits))
        elif name == 'y':
            gate_list.append(('Y', qubits))
        elif name == 'z':
            gate_list.append(('Z', qubits))
        elif name == 'id':
            gate_list.append(('ID', qubits))
        else:
            raise ValueError(f"Unsupported gate type found in optimized circuit: {name}")
            
    return gate_list

def gates_to_qiskit_circuit(n_qubits: int, gates_list: list) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for name, args in gates_list:
        if name == 'H':
            qc.h(args[0])
        elif name == 'S':
            qc.s(args[0])
        elif name == 'Sdg':
            qc.sdg(args[0])
        elif name == 'X':
            qc.x(args[0])
        elif name == 'Y':
            qc.y(args[0])
        elif name == 'Z':
            qc.z(args[0])
        elif name == 'CNOT':
            qc.cx(args[0], args[1])
        elif name == 'CZ':
            qc.cz(args[0], args[1])
        elif name == 'SWAP':
            qc.swap(args[0], args[1])
        else:
            raise ValueError(f"Unsupported gate for Qiskit conversion: {name}")
    return qc

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

def int_to_bitlist(n, width=None, lsb_first=False):
    """
    Converts integer n to a list of bits.

    Parameters:
    - n: integer
    - width: total number of bits to pad to (optional)
    - lsb_first: if True, return bits from LSB to MSB; otherwise, MSB to LSB

    Returns:
    - List of bits in specified order
    """
    if not isinstance(n, numbers.Integral):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative.")
    if width is None:
        width = n.bit_length() or 1  # handle n = 0

    bits = [(n >> i) & 1 for i in range(width)]
    return bits if lsb_first else bits[::-1]

def is_bitlist(bitlist):
    return all(b in (0, 1) for b in bitlist)

def _pack_token_count_dict(d: dict[bytes, int]):
    """Convert {bytes_token: count} -> (hex_tokens[str], counts[int64]) arrays."""
    if not d:
        return np.array([], dtype=str), np.array([], dtype=np.int64)
    tokens_hex = np.array([tok.hex() for tok in d.keys()], dtype=str)
    counts = np.array([int(c) for c in d.values()], dtype=np.int64)
    return tokens_hex, counts

def paired_mse_difference(energy_estimates_1, energy_estimates_2, E_GS,
                          n_boot=10000, ci=0.67, seed=0):
    """
    Paired (common-random-numbers) estimate of the MSE difference:
        ΔMSE = MSE_1 - MSE_2
    where MSE_k = E[(Ehat_k - E_GS)^2].

    Assumes energy_estimates_1[i] and energy_estimates_2[i] come from the
    same underlying randomness/seed (paired runs).

    Parameters
    ----------
    energy_estimates_1, energy_estimates_2 : array-like, shape (n_runs,)
        Paired empirical energy estimates for protocol 1 and 2.
    E_GS : float
        True/target energy.
    n_boot : int, default=10000
        Number of paired bootstrap resamples (resample indices with replacement).
    ci : float in (0,1), default=0.67
        Two-sided percentile confidence level for ΔMSE.
    seed : int, default=0
        RNG seed for bootstrapping.

    Returns
    -------
    delta_mse : float
        Point estimate of ΔMSE = MSE_1 - MSE_2 on the original paired data.
    se : float
        Bootstrap standard error of ΔMSE.
    ci_interval : (float, float)
        Percentile confidence interval for ΔMSE at level `ci`.
    boot_values : np.ndarray
        Bootstrap ΔMSE values, shape (n_boot,).

    Notes
    -----
    - Because this is paired, it typically has much lower variance than
      separately estimating MSE_1 and MSE_2 and subtracting.
    - Positive ΔMSE means protocol 2 is better (smaller MSE).
    """
    x1 = np.asarray(energy_estimates_1, dtype=float)
    x2 = np.asarray(energy_estimates_2, dtype=float)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("energy_estimates_1 and energy_estimates_2 must be 1D arrays.")
    if x1.size == 0:
        raise ValueError("Inputs must be non-empty.")
    if x1.size != x2.size:
        raise ValueError("Paired inputs must have the same length.")

    n = x1.size
    e1 = x1 - E_GS
    e2 = x2 - E_GS

    # Per-run squared-error difference (paired)
    d = e1**2 - e2**2

    # Point estimate: mean paired difference in squared error
    delta_mse = float(np.mean(d))

    # Paired bootstrap over runs
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_values = np.mean(d[idx], axis=1)

    se = float(boot_values.std(ddof=1))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_values, [alpha, 1.0 - alpha])

    return delta_mse, se, (float(lo), float(hi)), boot_values

def random_pauli_strings(M, n, rng=None):
    """
    Generate M random n-qubit Pauli strings, excluding the all-identity string.

    Parameters
    ----------
    M : int
        Number of Pauli strings to generate.
    n : int
        Number of qubits.
    return_int : bool, optional
        If True, returns an (M, n) array with ints in {0=I,1=X,2=Y,3=Z}.
        If False, returns a list of strings of length n, with chars in {"I","X","Y","Z"}.
    rng : np.random.Generator or None
        Random number generator. If None, uses default RNG.

    Returns
    -------
    list of str or np.ndarray
        Random Pauli strings in the requested format.
    """
    if rng is None:
        rng = np.random.default_rng()

    codes = np.empty((M, n), dtype=np.int64)
    for i in range(M):
        while True:
            cand = rng.integers(0, 4, size=n)
            if np.any(cand != 0):  # exclude all-identity
                codes[i] = cand
                break

    lookup = np.array(["I", "X", "Y", "Z"])
    strings = ["".join(lookup[codes[i]]) for i in range(M)]

    return strings, codes

def settings_to_dict(settings_rounds,
                     settings_dict,
                     settings_buffer,
                     order=None):
    """
    Convert a list of settings (each a list/array of observable indices)
    into token-count dictionaries.

    Parameters
    ----------
    settings_rounds : list of 1D integer arrays / lists
        Each entry is a (variable-length) list/array of observable indices
        selected in that round. This is the unified format for qwc/fc/kc.

    settings_dict : dict
        Global cumulative counts {token (bytes) : total_count_so_far}.

    settings_buffer : dict
        Session/buffer counts {token (bytes) : count_since_last_reset_or_read}.

    order : dict or None
        If provided, maps each token (bytes) to the list of local round indices
        (0..len(settings_rounds)-1) that produced that token in this call.

    Notes
    -----
    - We canonicalize each setting by sorting indices and casting to int32 before encoding.
    - Empty settings are allowed but usually indicate a bug upstream; we keep them.
    """

    batch_counts = {}
    batch_rounds = {}
    
    for r, setting_indices in enumerate(settings_rounds):
        arr = np.asarray(setting_indices, dtype=np.int32)

        if arr.size > 1:
            arr = np.sort(arr)

        token = encode_setting_token(arr)

        batch_counts[token] = batch_counts.get(token, 0) + 1

        if order is not None:
            batch_rounds.setdefault(token, []).append(r)

    for token, c in batch_counts.items():
        c = int(c)
        settings_dict[token] = settings_dict.get(token, 0) + c
        settings_buffer[token] = settings_buffer.get(token, 0) + c

        if order is not None:
            order[token] = batch_rounds[token]
            
    return
            
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
            
def stable_hash(setting_token, *, digest_size=8) -> int:
    """
    Deterministic hash -> nonnegative Python int.

    digest_size=8 gives a 64-bit hash (good enough for seeding).
    """
    # Normalize token to raw bytes deterministically
    if isinstance(setting_token, (bytes, bytearray, memoryview)):
        b = bytes(setting_token)
    elif isinstance(setting_token, np.bytes_):
        b = bytes(setting_token)  # numpy scalar bytes -> python bytes
    elif isinstance(setting_token, str):
        b = setting_token.encode("utf-8")
    else:
        # last resort: convert array-like to bytes deterministically
        b = np.asarray(setting_token).tobytes()

    digest = hashlib.blake2b(b, digest_size=digest_size).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)

def token_to_pauli_list(token: bytes, compat_type: str, obs: np.ndarray) -> list[str]:
    """
    Convert a stored 'setting token' back into human / circuit friendly Pauli strings.

    Parameters
    ----------
    token : bytes
        The encoded setting (output of encode_setting_token).
    compat_type : str
        "qwc" or "fc" or "kc"
        - "qwc":
            token decodes to shape (num_qubits,), entries in {0,1,2,3}.
            This is a single global measurement basis across qubits.
            We return a *list with one string*, e.g. ["XZYI..."], because
            that setting means "measure all qubits in this joint basis."
        - "fc" or "kc":
            token decodes to a list of observable indices, e.g. [5, 17, 22].
            We look up each obs[i] (which is a length-num_qubits array of {0,1,2,3})
            and convert those rows to separate Pauli strings,
            e.g. ["XZII", "ZZZX", "IIZZ", ...].
            We return that list of strings.

    obs : np.ndarray, shape (num_obs, num_qubits), dtype=int
        Each row is a Pauli string encoded as 0=I,1=X,2=Y,3=Z.

    Returns
    -------
    list[str]
        For "qwc"       : ["XYZI..."]  (single element list)
        For "fc" or "kc":  ["XZII", "ZZZX", ...] (one per observable in the commuting group)
    """
    idx_arr = decode_setting_token(token)  # np.int32 1D array
    pauli_map = np.array(['I','X','Y','Z'], dtype='<U1')

    if compat_type == "qwc":
        # idx_arr is length num_qubits, entries in {0,1,2,3}
        pauli_str = ''.join(pauli_map[idx_arr])
        return [pauli_str]

    elif compat_type == "fc" or compat_type == 'kc':
        pauli_list = []
        for obs_idx in idx_arr:
            # sanity check: observable index must be valid
            if obs_idx < 0 or obs_idx >= obs.shape[0]:
                raise IndexError(
                    f"token_to_pauli_list: observable index {obs_idx} is out of bounds "
                    f"for obs with shape {obs.shape}"
                )
            row = obs[obs_idx]  # shape (num_qubits,), entries in {0,1,2,3}
            word = ''.join(pauli_map[row])
            pauli_list.append(word)
        return pauli_list

    else:
        raise ValueError(f"Unknown compat_type '{compat_type}', expected 'qwc' or 'fc' or 'kc'.")
        
def _unpack_token_count_dict(tokens_hex: np.ndarray, counts: np.ndarray):
    """Convert (hex_tokens[str], counts[int]) -> {bytes_token: count}."""
    out = {}
    for h, c in zip(tokens_hex.tolist(), counts.tolist()):
        out[bytes.fromhex(str(h))] = int(c)
    return out