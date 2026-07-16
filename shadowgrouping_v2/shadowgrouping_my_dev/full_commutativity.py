import numpy as np, numba
from numba import njit, prange
from typing import List, Tuple, Dict, Optional, Set
from qibo import models, gates
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import LinearFunction
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Clifford
from qiskit.synthesis import synth_clifford_full
from qiskit.synthesis import synth_cnot_depth_line_kms
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CollectLinearFunctions

from shadowgrouping_v2.shadowgrouping_my_dev.helper_functions import decompose_dense_clifford_gates, gates_to_qiskit_circuit

Gate = Tuple[str, Tuple[int, ...]]  # ('H',(q,)), ('S',(q,)), ('CNOT',(a,b)), ('CZ',(a,b)), ('SWAP',(a,b))

class _GF2LinearBasis:
    """
    Maintains an XOR-basis for bitstrings of length `max_bits`.
    Supports:
      - add(v): True iff v is independent (and gets added)
      - contains(v): True iff v is in the span of the current basis
      - copy(): returns an independent copy of the current basis
    """
    __slots__ = ("max_bits", "basis", "rank")

    def __init__(self, max_bits: int):
        self.max_bits = int(max_bits)
        self.basis = [0] * self.max_bits
        self.rank = 0

    def copy(self):
        other = _GF2LinearBasis(self.max_bits)
        other.basis = self.basis.copy()
        other.rank = self.rank
        return other

    def add(self, v: int) -> bool:
        v = int(v)
        if v == 0:
            return False
        for b in range(self.max_bits - 1, -1, -1):
            if (v >> b) & 1:
                if self.basis[b]:
                    v ^= self.basis[b]
                else:
                    self.basis[b] = v
                    self.rank += 1
                    return True
        return False

    def contains(self, v: int) -> bool:
        v = int(v)
        if v == 0:
            return True
        for b in range(self.max_bits - 1, -1, -1):
            if ((v >> b) & 1) and self.basis[b]:
                v ^= self.basis[b]
        return v == 0
    
@numba.njit(cache=True)
def _popcount_u64(x):
    # Hamming weight for 64-bit integers (works in nopython)
    x = x - ((x >> 1) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> 2) & np.uint64(0x3333333333333333))
    x = (x + (x >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x + (x >> 8)
    x = x + (x >> 16)
    x = x + (x >> 32)
    return np.uint64(x & np.uint64(0x7F))


@numba.njit(cache=True)
def fc_compat_row_numba(x_all_u64, z_all_u64, xi_u64, zi_u64):
    """
    Return row[j] = True iff Pauli i commutes with Pauli j (full commutativity).
    x_all_u64, z_all_u64: (M,) uint64 arrays
    xi_u64, zi_u64: scalars for the generator i
    """
    M = x_all_u64.size
    out = np.empty(M, dtype=np.bool_)
    for j in range(M):
        parity = (_popcount_u64(xi_u64 & z_all_u64[j]) + _popcount_u64(zi_u64 & x_all_u64[j])) & np.uint64(1)
        out[j] = (parity == 0)
    return out


def _export_basis_compact(basis: "_GF2LinearBasis"):
    """
    Convert Python-list pivoted basis into compact arrays for Numba.
    Returns:
      basis_rows: (r,) uint64
      pivot_bits: (r,) uint8   (pivot indices, descending)
    """
    pivots = [b for b in range(basis.max_bits - 1, -1, -1) if basis.basis[b] != 0]
    pivot_bits = np.asarray(pivots, dtype=np.uint8)
    basis_rows = np.asarray([basis.basis[b] for b in pivots], dtype=np.uint64)
    return basis_rows, pivot_bits

def apply_CNOT_tab(Z: np.ndarray, X: np.ndarray, a: int, b: int):
    X[:, b] ^= X[:, a]
    Z[:, a] ^= Z[:, b]

def apply_CZ_tab(Z: np.ndarray, X: np.ndarray, a: int, b: int):
    # X_a -> X_a Z_b ; X_b -> Z_a X_b (implemented as Z updates)
    Z[:, b] ^= X[:, a]
    Z[:, a] ^= X[:, b]

def apply_H_tab(Z: np.ndarray, X: np.ndarray, q: int):
    Z[:, q], X[:, q] = X[:, q].copy(), Z[:, q].copy()

def apply_S_tab(Z: np.ndarray, X: np.ndarray, q: int):
    Z[:, q] ^= X[:, q]
    
def apply_Steiner_synthesis(qc: QuantumCircuit, connectivity: List[Tuple[int, int]]) -> QuantumCircuit:
    """
    Identifies blocks of linear gates (CNOT/SWAP) and re-synthesizes them 
    using the Kutin-Moulton-Smith (KMS) algorithm, which is optimized for 
    linear nearest-neighbor (LNN) connectivity.
    """
    # 1. Collect pure CNOT/SWAP blocks into LinearFunctions
    pm_collect = PassManager(CollectLinearFunctions())
    qc_linearized = pm_collect.run(qc)
    
    # 2. Iterate over the circuit and replace LinearFunctions with KMS circuits
    new_dag = circuit_to_dag(qc_linearized)
    
    for node in new_dag.op_nodes():
        if isinstance(node.op, LinearFunction):
            linear_matrix = node.op.linear
            steiner_circuit = synth_cnot_depth_line_kms(linear_matrix)
            
            # Replace the original block with the connectivity-aware block
            new_dag.substitute_node_with_dag(node, circuit_to_dag(steiner_circuit))
            
    return dag_to_circuit(new_dag)
    
def check_commuting(Z: np.ndarray, X: np.ndarray) -> bool:
    m, _ = Z.shape
    for i in range(m):
        for j in range(i+1, m):
            sip = (int(np.dot(Z[i], X[j]) + np.dot(X[i], Z[j])) % 2)
            if sip != 0:
                return False
    return True

@njit(parallel=True)
def compute_exact_compat_degrees_qwc(obs_int8):
    M, N = obs_int8.shape
    degrees = np.zeros(M, dtype=np.float32)
    for i in prange(M):
        deg = 0
        for j in range(M):
            commutes = True
            for k in range(N):
                a = obs_int8[i, k]
                b = obs_int8[j, k]
                if a != 0 and b != 0 and a != b:
                    commutes = False
                    break
            if commutes:
                deg += 1
        degrees[i] = deg
    return degrees

@njit(parallel=True)
def compute_exact_compat_degrees_fc(obs_int8):
    M, N = obs_int8.shape
    degrees = np.zeros(M, dtype=np.float32)
    for i in prange(M):
        deg = 0
        for j in range(M):
            anti_commutes = 0
            for k in range(N):
                a = obs_int8[i, k]
                b = obs_int8[j, k]
                if a != 0 and b != 0 and a != b:
                    anti_commutes += 1
            if anti_commutes % 2 == 0:
                deg += 1
        degrees[i] = deg
    return degrees

@njit(parallel=True)
def compute_approx_compat_degrees_qwc(obs_int8, n_samples=1000):
    M, N = obs_int8.shape
    degrees_est = np.zeros(M, dtype=np.float32)
    k = min(n_samples, M)
    for i in prange(M):
        deg_count = 0
        for step in range(k):
            j = (i * 73 + step * 101) % M
            commutes = True
            for c in range(N):
                a = obs_int8[i, c]
                b = obs_int8[j, c]
                if a != 0 and b != 0 and a != b:
                    commutes = False
                    break
            if commutes:
                deg_count += 1
        degrees_est[i] = (deg_count / k) * M
    return degrees_est

@njit(parallel=True)
def compute_approx_compat_degrees_fc(obs_int8, n_samples=1000):
    M, N = obs_int8.shape
    degrees_est = np.zeros(M, dtype=np.float32)
    k = min(n_samples, M)
    for i in prange(M):
        deg_count = 0
        for step in range(k):
            j = (i * 73 + step * 101) % M
            anti_commutes = 0
            for c in range(N):
                a = obs_int8[i, c]
                b = obs_int8[j, c]
                if a != 0 and b != 0 and a != b:
                    anti_commutes += 1
            if anti_commutes % 2 == 0:
                deg_count += 1
        degrees_est[i] = (deg_count / k) * M
    return degrees_est

def cnot_depth_qiskit(qc, swap_cost: int = 3) -> int:
    """
    Two-qubit depth proxy measured in 'CNOT layers':
      - CX/CNOT/CZ cost 1
      - SWAP cost `swap_cost` (default 3)
    Single-qubit gates ignored.
    """
    n = qc.num_qubits
    clock = [0] * n

    for inst in qc.data:
        op = inst.operation
        qargs = inst.qubits
        if len(qargs) != 2:
            continue

        name = op.name.lower()
        if name in ("cx", "cnot", "cz"):
            cost = 1
        elif name == "swap":
            cost = swap_cost
        else:
            # Conservative fallback: count any unknown 2q gate as 1 entangling layer
            cost = 1

        q0 = qc.find_bit(qargs[0]).index
        q1 = qc.find_bit(qargs[1]).index
        start = clock[q0] if clock[q0] > clock[q1] else clock[q1]
        finish = start + cost
        clock[q0] = finish
        clock[q1] = finish

    return max(clock) if clock else 0

def conjugate_pauli_by_circuit(pauli: str, gates: List[Gate]) -> Tuple[str, int]:
    s = pauli
    sign = 1
    for g in gates:
        s, sgn = conjugate_pauli_by_gate(s, g)
        sign *= sgn
    return s, sign

def conjugate_pauli_by_gate(pauli: str, gate: Gate) -> Tuple[str, int]:
    """
    Conjugate a Pauli string by a single Clifford gate g, returning (new_string, ±1 sign).
    Supports conjugation with H, S, CNOT, CZ, SWAP, Sdg, X, Y, Z, ID.
    
    Operation: Computes g P g^dagger.
    """
    g_name, args = gate
    p = list(pauli)
    sign = 1

    # --- Single Qubit Gates ---
    if g_name == 'H':
        (q,) = args
        ch = p[q]
        if ch == 'X':
            p[q] = 'Z'
        elif ch == 'Z':
            p[q] = 'X'
        elif ch == 'Y':
            p[q] = 'Y'; sign *= -1  # H Y H = -Y

    elif g_name == 'S':
        (q,) = args
        ch = p[q]
        if ch == 'X':
            p[q] = 'Y'              # S X S† = Y
        elif ch == 'Y':
            p[q] = 'X'; sign *= -1  # S Y S† = -X
        # Z and I unchanged

    elif g_name == 'Sdg':  # S†
        (q,) = args
        ch = p[q]
        if ch == 'X':
            p[q] = 'Y'; sign *= -1  # S† X S = -Y
        elif ch == 'Y':
            p[q] = 'X'              # S† Y S = X
        # Z and I unchanged

    elif g_name == 'X':
        (q,) = args
        ch = p[q]
        if ch == 'Y' or ch == 'Z':
            sign *= -1              # X anticommutes with Y and Z

    elif g_name == 'Y':
        (q,) = args
        ch = p[q]
        if ch == 'X' or ch == 'Z':
            sign *= -1              # Y anticommutes with X and Z

    elif g_name == 'Z':
        (q,) = args
        ch = p[q]
        if ch == 'X' or ch == 'Y':
            sign *= -1              # Z anticommutes with X and Y

    elif g_name == 'ID':
        pass

    # --- Two Qubit Gates ---
    elif g_name == 'CNOT':
        a, b = args
        pair = (p[a], p[b])
        
        # Table for CNOT (Control a, Target b)
        # Mapping: XI->XX, XZ->YY(-), YI->YX, YZ->XY, ZI->ZI, ZX->ZX, IX->IX, IZ->ZZ
        cnot_map = {
            ("I","I"): ("II", +1), ("I","X"): ("IX", +1), ("I","Y"): ("ZY", +1), ("I","Z"): ("ZZ", +1),
            ("X","I"): ("XX", +1), ("X","X"): ("XI", +1), ("X","Y"): ("YZ", +1), ("X","Z"): ("YY", -1),
            ("Y","I"): ("YX", +1), ("Y","X"): ("YI", +1), ("Y","Y"): ("XZ", -1), ("Y","Z"): ("XY", +1),
            ("Z","I"): ("ZI", +1), ("Z","X"): ("ZX", +1), ("Z","Y"): ("IY", +1), ("Z","Z"): ("IZ", +1),
        }
        
        if pair in cnot_map:
            new_pair, sgn = cnot_map[pair]
            p[a], p[b] = new_pair[0], new_pair[1]
            sign *= sgn
        else:
             # Should be covered above, but fallback for safety
             raise ValueError(f"Unexpected Pauli pair {pair} for CNOT")

    elif g_name == 'CZ':
        a, b = args
        pair = (p[a], p[b])
        
        # Table for CZ (Symmetric)
        # Mapping: X -> XZ, Y -> YZ (basically appends Z to the other qubit)
        cz_map = {
            ("I","I"): ("II", +1), ("I","X"): ("ZX", +1), ("I","Y"): ("ZY", +1), ("I","Z"): ("IZ", +1),
            ("X","I"): ("XZ", +1), ("X","X"): ("YY", +1), ("X","Y"): ("YX", -1), ("X","Z"): ("XI", +1),
            ("Y","I"): ("YZ", +1), ("Y","X"): ("XY", -1), ("Y","Y"): ("XX", +1), ("Y","Z"): ("YI", +1),
            ("Z","I"): ("ZI", +1), ("Z","X"): ("IX", +1), ("Z","Y"): ("IY", +1), ("Z","Z"): ("ZZ", +1),
        }

        if pair in cz_map:
            new_pair, sgn = cz_map[pair]
            p[a], p[b] = new_pair[0], new_pair[1]
            sign *= sgn
        else:
             raise ValueError(f"Unexpected Pauli pair {pair} for CZ")

    elif g_name == 'SWAP':
        a, b = args
        p[a], p[b] = p[b], p[a]

    else:
        raise ValueError(f"Unknown gate: {g_name}")

    return ''.join(p), sign

def diagonalize_and_map(paulis: List[str], debug_checks: bool = True) -> Dict:
    """
    Constructs a Clifford circuit that diagonalizes the Pauli strings.
    """
    m_paulis = len(paulis)
    
    # 1. Convert all Paulis to binary symplectic form
    Z_rows, X_rows = [], []
    for s in paulis:
        z, x = pauli_to_zx(s) 
        Z_rows.append(z)
        X_rows.append(x)
        
    Z_all = np.vstack(Z_rows).astype(np.uint8)
    X_all = np.vstack(X_rows).astype(np.uint8)

    # 2. Partition Qubits (Numba-accelerated)
    qwc_qubits, general_qubits = partition_qubits_by_qwc_numba(Z_all, X_all)
    
    # QWC subspace
    qwc_gates = []
    for q in qwc_qubits:
        active_op = 'I'
        col_z = Z_all[:, q]
        col_x = X_all[:, q]
        
        for i in range(m_paulis):
            z, x = col_z[i], col_x[i]
            if z == 1 and x == 0: active_op = 'Z'; break
            if z == 0 and x == 1: active_op = 'X'; break
            if z == 1 and x == 1: active_op = 'Y'; break
        
        if active_op == 'X':
            qwc_gates.append(('H', (q,)))
        elif active_op == 'Y':
            qwc_gates.append(('Sdg', (q,)))
            qwc_gates.append(('H', (q,)))

    # FC subspace
    general_gates = []
    if general_qubits:
        sub_paulis = []
        for i in range(m_paulis):
            s_sub = "".join(paulis[i][q] for q in general_qubits)
            sub_paulis.append(s_sub)
            
        sub_circ = diagonalizing_clifford(sub_paulis, debug_checks=debug_checks)
        sub_gates_raw = sub_circ["gates"]
        
        # Remap local sub-problem indices to global indices
        index_map = {local: global_q for local, global_q in enumerate(general_qubits)}
        for name, local_args in sub_gates_raw:
            global_args = tuple(index_map[q] for q in local_args)
            general_gates.append((name, global_args))

    # 3. Combine and Map
    final_gates = qwc_gates + general_gates
    
    mappings = []
    active_measured_qubits: Set[int] = set()

    for s in paulis:
        final_s, sign = conjugate_pauli_by_circuit(s, final_gates)
        
        z_qubits = [q for q, ch in enumerate(final_s) if ch == 'Z']
        active_measured_qubits.update(z_qubits)
        
        mappings.append({
            "pauli": s,
            "final_Z_string": final_s,
            "sign": int(sign),
            "z_qubits_to_parity": z_qubits,
        })

    return {
        "measured": sorted(list(active_measured_qubits)),
        "gates": final_gates, "mappings": mappings}

def diagonalizing_clifford(paulis: List[str], debug_checks: bool = True) -> Dict:
    n = len(paulis[0])
    Z_rows, X_rows = [], []
    for s in paulis:
        z, x = pauli_to_zx(s)
        Z_rows.append(z); X_rows.append(x)
    Z_all = np.vstack(Z_rows).astype(np.uint8)
    X_all = np.vstack(X_rows).astype(np.uint8)
    assert check_commuting(Z_all, X_all), "Inputs are not mutually commuting."

    Z, X, keep_idx = drop_dependent_generators(Z_all, X_all)
    m = Z.shape[0]
    gates: List[Gate] = []

    def col_rank(A): return rank_gf2(A.copy().T)

    # Make X full-rank via H
    while col_rank(X) < m:
        improved = False
        current_r = col_rank(X)
        for q in range(n):
            apply_H_tab(Z, X, q)
            rnew = col_rank(X)
            if rnew > current_r:
                gates.append(('H', (q,)))
                improved = True
                break
            else:
                apply_H_tab(Z, X, q)
        if not improved:
            raise RuntimeError("Could not make X full-rank; inputs may be dependent.")

    # Row-reduce X to give unique pivot columns
    used_cols = set()
    pivots: List[int] = [-1] * m
    for i in range(m):
        # Choose fresh column p with X[i,p]==1
        p = None
        for q in range(n):
            if q not in used_cols and X[i, q]:
                p = q; break
        if p is None:
            # Try swapping with later row that has a fresh column
            swapped = False
            for r in range(i+1, m):
                for q in range(n):
                    if q not in used_cols and X[r, q]:
                        Z[[i, r]] = Z[[r, i]]
                        X[[i, r]] = X[[r, i]]
                        p = q
                        swapped = True
                        break
                if swapped: break
            if not swapped:
                # Create a pivot at some unused column by single CNOT from t with X[i,t]==1
                for q in range(n):
                    if q not in used_cols:
                        p = q; break
                t_candidates = [t for t in range(n) if X[i, t] and t != p]
                if not X[i, p]:
                    if not t_candidates:
                        raise RuntimeError("Cannot create a pivot in an unused column.")
                    t = t_candidates[0]
                    apply_CNOT_tab(Z, X, t, p); gates.append(('CNOT', (t, p)))
        pivots[i] = p
        used_cols.add(p)
        # Clear other Xs in row i via CNOT p->t
        for t in [t for t in range(n) if t != p and X[i, t]]:
            apply_CNOT_tab(Z, X, p, t); gates.append(('CNOT', (p, t)))
        # Clear X in column p for other rows via row ops (no gates)
        for r in range(m):
            if r != i and X[r, p]:
                Z[r, :] ^= Z[i, :]
                X[r, :] ^= X[i, :]

    # Sanity check: with X identity on pivot columns, Z over pivots must be symmetric
    if debug_checks:
        for i in range(m):
            for j in range(i+1, m):
                pi, pj = pivots[i], pivots[j]
                assert Z[i, pj] == Z[j, pi], "Z off-diagonal over pivots is not symmetric after Step 2."

    # Clear Z-diagonal at (i, p_i) with S on p_i when needed
    for i, p in enumerate(pivots):
        if Z[i, p]:
            apply_S_tab(Z, X, p); gates.append(('S', (p,)))

    # Sanity check: S toggles only diagonals; symmetry should persist
    if debug_checks:
        for i in range(m):
            for j in range(i+1, m):
                pi, pj = pivots[i], pivots[j]
                assert Z[i, pj] == Z[j, pi], "Z off-diagonal over pivots lost symmetry after Step 3."

    # Clear all Zs on NON-pivot columns using CZ(p_i, t)
    pivset = set(pivots)
    nonpivots = [t for t in range(n) if t not in pivset]
    for i, p in enumerate(pivots):
        for t in nonpivots:
            if Z[i, t]:                      # only this row toggles because X[:,t]==0
                apply_CZ_tab(Z, X, p, t)
                gates.append(('CZ', (p, t)))

    # Sanity check: after 3b, each row's Z support is confined to pivot columns
    if debug_checks:
        for i, p in enumerate(pivots):
            for t in range(n):
                if t not in pivset:
                    assert Z[i, t] == 0, "Non-pivot Z not cleared before pivot-pivot CZ."

    # Clear off-diagonal Z (within pivot block) with CZ using a single-sided check
    for i in range(m):
        for j in range(i+1, m):
            pi, pj = pivots[i], pivots[j]
            if Z[i, pj]:  # single-sided (not OR)
                apply_CZ_tab(Z, X, pi, pj); gates.append(('CZ', (pi, pj)))

    # H on pivots to map X -> Z
    for p in pivots:
        apply_H_tab(Z, X, p); gates.append(('H', (p,)))

    # Sanity check: generators should be pure Z on their pivots
    if debug_checks:
        for i, p in enumerate(pivots):
            assert np.sum(X[i]) == 0, "Final X support not zero."
            assert np.sum(Z[i]) == 1 and Z[i, p] == 1, "Final Z not a single Z on pivot."

    final_gen = [zx_to_pauli(Z[i], X[i]) for i in range(m)]
    return {"kept_indices": keep_idx, "pivots": pivots, "gates": gates, "final_generators": final_gen}

def drop_dependent_generators(Z: np.ndarray, X: np.ndarray):
    M = np.hstack([Z, X]) % 2
    keep = []
    R = np.zeros_like(M)
    r = 0
    for i in range(len(M)):
        trial = R.copy()
        trial[r, :] = M[i, :]
        if rank_gf2(trial[:r+1, :]) == r+1:
            R[r, :] = M[i, :]
            keep.append(i)
            r += 1
    return Z[keep].copy(), X[keep].copy(), keep

def fully_commute(O, P):
    """
    Return True iff Pauli strings O and P commute (full commutativity).
    O, P can be iterables of ints (0,1,2,3 -> I,X,Y,Z) or chars ('I','X','Y','Z').
    """
    if len(O) != len(P):
        raise ValueError("Pauli strings must have the same length.")
    
    def is_I(a):
        return a == 0 or a == 'I'
    
    parity = 0  # counts mismatching non-identity positions mod 2
    for o, p in zip(O, P):
        if not is_I(o) and not is_I(p) and o != p:
            parity ^= 1  # flip parity
    return parity == 0

@numba.njit
def fully_commute_numba(O, P):
    """
    Numba-accelerated check: do Pauli strings O, P commute fully?
    O, P are 1D arrays of ints in {0=I,1=X,2=Y,3=Z}.
    """
    if O.shape[0] != P.shape[0]:
        raise ValueError("Pauli strings must have the same length.")
    parity = 0
    for i in range(O.shape[0]):
        o = O[i]
        p = P[i]
        if o != 0 and p != 0 and o != p:
            parity ^= 1
    return parity == 0

@numba.njit
def fully_commute_batched(A, B=None):
    """
    Batched full-commutativity check.
    
    Parameters
    ----------
    A : 2D array of shape (m, n)
        Pauli strings, with ints in {0=I,1=X,2=Y,3=Z}.
    B : 2D array of shape (k, n), optional
        Another set of Pauli strings. If None, computes the (m x m) 
        compatibility matrix for A against itself.
    
    Returns
    -------
    C : 2D boolean array
        If B is given: shape (m, k). Entry C[i,j] = True iff A[i] commutes with B[j].
        If B is None: shape (m, m). Entry C[i,j] = True iff A[i] commutes with A[j].
    """
    m, n = A.shape
    if B is None:
        C = np.empty((m, m), dtype=np.bool_)
        for i in range(m):
            C[i, i] = True  # always commute with itself
            for j in range(i+1, m):
                parity = 0
                for q in range(n):
                    o = A[i, q]
                    p = A[j, q]
                    if o != 0 and p != 0 and o != p:
                        parity ^= 1
                val = (parity == 0)
                C[i, j] = val
                C[j, i] = val
        return C
    else:
        k, n2 = B.shape
        if n != n2:
            raise ValueError("Pauli strings in A and B must have the same length.")
        C = np.empty((m, k), dtype=np.bool_)
        for i in range(m):
            for j in range(k):
                parity = 0
                for q in range(n):
                    o = A[i, q]
                    p = B[j, q]
                    if o != 0 and p != 0 and o != p:
                        parity ^= 1
                C[i, j] = (parity == 0)
        return C

@numba.njit(cache=True)
def in_span_batch_numba(packed_obs_u64, basis_rows_u64, pivot_bits_u8):
    """
    Test membership in span(basis_rows_u64) for each packed_obs_u64[i].

    packed_obs_u64: (M,) uint64
    basis_rows_u64: (r,) uint64, each has its pivot bit set
    pivot_bits_u8:  (r,) uint8, pivot bit indices (descending)

    Returns: (M,) bool
    """
    M = packed_obs_u64.size
    r = basis_rows_u64.size
    out = np.empty(M, dtype=np.bool_)

    for i in range(M):
        v = packed_obs_u64[i]
        for j in range(r):
            p = pivot_bits_u8[j]
            if (v >> p) & np.uint64(1):
                v ^= basis_rows_u64[j]
        out[i] = (v == 0)
    return out

def partition_qubits_by_qwc(paulis: List[str]) -> Tuple[List[int], List[int]]:
    """
    Partitions the total set of qubits into two groups: 
    1. QWC qubits (qubits where the Paulis commute qubit-wise).
    2. General qubits (the complement, where entanglement is required).
    
    Returns: (qwc_qubits, general_qubits)
    """
    n = len(paulis[0])
    qwc_qubits = []
    general_qubits = []
    
    for q in range(n):
        seen_ops = set()
        is_qwc = True
        
        for p_str in paulis:
            op = p_str[q]
            if op != 'I':
                seen_ops.add(op)
        
        if len(seen_ops) > 1:
            is_qwc = False
        
        if is_qwc:
            qwc_qubits.append(q)
        else:
            general_qubits.append(q)
            
    return qwc_qubits, general_qubits

@numba.njit
def partition_qubits_by_qwc_numba(Z: np.ndarray, X: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Numba-accelerated partitioning of qubits into QWC and General sets.
    Iterates through columns of the symplectic matrix to detect conflicts.
    
    Args:
        Z (np.ndarray): Binary Z matrix (shape M x N, dtype uint8).
        X (np.ndarray): Binary X matrix (shape M x N, dtype uint8).
        
    Returns:
        (qwc_qubits, general_qubits) as lists of qubit indices.
    """
    m, n = Z.shape
    qwc_list = []
    general_list = []
    
    for j in range(n):
        has_x = False
        has_z = False
        has_y = False
        
        is_qwc = True
        
        for i in range(m):
            z_val = Z[i, j]
            x_val = X[i, j]
            
            # Skip Identity (0,0)
            if z_val == 0 and x_val == 0:
                continue 
            
            if z_val == 1 and x_val == 0:
                has_z = True
            elif z_val == 0 and x_val == 1:
                has_x = True
            elif z_val == 1 and x_val == 1:
                has_y = True
                
            # Check for conflicts:
            # If we have seen more than one type of non-identity operator, it's not QWC.
            if (int(has_x) + int(has_z) + int(has_y)) > 1:
                is_qwc = False
                break
        
        if is_qwc:
            qwc_list.append(j)
        else:
            general_list.append(j)
            
    return qwc_list, general_list

def generate_qwc_circuit_partial(paulis: List[str], qwc_qubits: List[int]) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    Generates single-qubit gates for the specified QWC qubits.
    """
    gates = []
    
    for q in qwc_qubits:
        active_op = 'I'
        for p_str in paulis:
            if p_str[q] != 'I':
                active_op = p_str[q]
                break
        
        if active_op == 'X':
            gates.append(('H', (q,)))
        elif active_op == 'Y':
            gates.append(('Sdg', (q,))) 
            gates.append(('H', (q,)))
            
    return gates

def merge_diagonalization_results(diag_results: List[Dict]) -> Dict:
    """
    Merge multiple diagonalization_result dicts (typically one per kC block)
    into a single diagonalization_result representing the product of the circuits.

    Assumes blocks are disjoint so simple concatenation of gates is valid.
    """
    merged_gates = []
    merged_mappings = []
    merged_measured = set()

    for d in diag_results:
        merged_gates.extend(d.get("gates", []))
        merged_mappings.extend(d.get("mappings", []))
        for q in d.get("measured", []):
            merged_measured.add(int(q))

    return {
        "gates": merged_gates,
        "mappings": merged_mappings,
        "measured": sorted(merged_measured),
    }

def mismatched_qubits(O, P, return_commute=False):
    """
    Return indices where O and P have non-QWC support:
    both are non-identity and different (X/Y/Z vs X/Y/Z with O[i] != P[i]).

    For Pauli strings encoded as:
        0 -> I, 1 -> X, 2 -> Y, 3 -> Z,

    Parameters
    ----------
    O, P : 1D array-like of ints
        Arrays of length num_qubits, entries in {0,1,2,3}.
    return_commute : bool, optional (default: False)
        If False: return only the list of mismatch indices.
        If True : return (mismatch_indices, commute_bool).

    """
    mismatch = []
    parity = 0  # 0 = even #mismatches so far, 1 = odd

    for i, (o, p) in enumerate(zip(O, P)):
        if o != 0 and p != 0 and o != p:
            mismatch.append(i)
            parity ^= 1  # flip parity

    if return_commute:
        commute = (parity == 0)
        return mismatch, commute
    else:
        return mismatch

@numba.njit
def mismatched_qubits_numba(O, P):
    """
    Numba-accelerated version that returns both:
      - the indices of non-QWC mismatches,
      - a bool indicating whether O and P commute globally.

    Parameters
    ----------
    O, P : 1D np.ndarray[int64]
        Length num_qubits, entries in {0,1,2,3}.
    """
    n = O.shape[0]
    out = np.empty(n, dtype=np.int64)
    count = 0
    parity = 0  # 0 = even, 1 = odd

    for i in range(n):
        o = O[i]
        p = P[i]
        if o != 0 and p != 0 and o != p:
            out[count] = i
            count += 1
            parity ^= 1

    mismatch_indices = out[:count]
    commute = (parity == 0)
    return mismatch_indices, commute

@numba.njit
def mismatched_qubits_batched_numba(A, B=None):
    """
    Batched version of mismatched_qubits_numba.

    Parameters
    ----------
    A : 2D np.ndarray[int64], shape (m, n)
        Set of Pauli strings, with entries in {0=I,1=X,2=Y,3=Z}.
    B : 2D np.ndarray[int64], shape (k, n), optional
        Second set of Pauli strings. If None, we compare A against itself.

    Returns
    -------
    mismatch_indices : 3D np.ndarray[int64]
        If B is None:
            shape (m, m, n_qubits)
        else:
            shape (m, k, n_qubits)

        For each pair (i,j), the list of mismatched qubit indices is stored
        in mismatch_indices[i, j, :], padded with -1. The *valid* mismatch
        positions for (i,j) are:

            mismatch_indices[i, j, mismatch_indices[i, j, :] != -1]

        A "mismatch" means: at that qubit q,
            A[i, q] != 0, B[j, q] != 0, and A[i, q] != B[j, q].

    commute : 2D np.ndarray[bool_]
        If B is None:
            shape (m, m)
        else:
            shape (m, k)

        commute[i, j] is True iff A[i] and B[j] commute globally, i.e.,
        the number of mismatched qubits between them is even.
    """
    m, n = A.shape

    if B is None:
        # A vs A
        mismatch_indices = np.full((m, m, n), -1, dtype=np.int64)
        commute = np.empty((m, m), dtype=np.bool_)

        for i in range(m):
            # diagonal: identical strings -> no mismatches, always commute
            commute[i, i] = True
            # mismatch_indices[i, i, :] already all -1
            for j in range(i + 1, m):
                count = 0
                parity = 0  # parity of number of mismatches

                for q in range(n):
                    o = A[i, q]
                    p = A[j, q]
                    if o != 0 and p != 0 and o != p:
                        mismatch_indices[i, j, count] = q
                        mismatch_indices[j, i, count] = q
                        count += 1
                        parity ^= 1

                val = (parity == 0)
                commute[i, j] = val
                commute[j, i] = val

        return mismatch_indices, commute

    else:
        # A vs B
        k, n2 = B.shape
        if n != n2:
            raise ValueError("Pauli strings in A and B must have the same length.")

        mismatch_indices = np.full((m, k, n), -1, dtype=np.int64)
        commute = np.empty((m, k), dtype=np.bool_)

        for i in range(m):
            for j in range(k):
                count = 0
                parity = 0

                for q in range(n):
                    o = A[i, q]
                    p = B[j, q]
                    if o != 0 and p != 0 and o != p:
                        mismatch_indices[i, j, count] = q
                        count += 1
                        parity ^= 1

                commute[i, j] = (parity == 0)

        return mismatch_indices, commute
 
def optimize_clifford_decomposition(
    diagonalization_result: dict,
    n_qubits: int,
    qubit_connectivity: Optional[List[Tuple[int, int]]] = None,
    transpilation_trials: int = 40,
    method_connectivity: str = "greedy",
    swap_cost: int = 3,
    seed_base: int = 0,
    alltoall_synth_methods: Tuple[str, ...] = ("greedy", "AG"),
) -> Dict:
    """
    Optimizes the Clifford decomposition and updates the diagonalization result structure.

    Objective: minimize two-qubit depth in 'CNOT layers' where:
      - CX/CNOT/CZ cost 1
      - SWAP costs `swap_cost` (default 3)

    If qubit_connectivity is None (all-to-all):
      - try a small set of Clifford synthesis methods and pick the one with minimal CNOT-depth.

    If qubit_connectivity is provided:
      - route/optimize with Sabre over multiple trials and pick minimal CNOT-depth.
    """
    original_gates = diagonalization_result["gates"]
    qc_orig = gates_to_qiskit_circuit(n_qubits, original_gates)
    cliff_operator = Clifford(qc_orig)

    if qubit_connectivity is None:
        best_qc = None
        best_metric = float("inf")

        for synth_method in alltoall_synth_methods:
            qc_synth = synth_clifford_full(cliff_operator, method=synth_method)
            qc_clean = decompose_dense_clifford_gates(qc_synth)

            metric = cnot_depth_qiskit(qc_clean, swap_cost=swap_cost)
            if metric < best_metric:
                best_metric = metric
                best_qc = qc_clean

        if best_qc is None:
            raise RuntimeError("All-to-all synthesis failed to produce any circuit.")
        final_qc = best_qc

    else:
        if method_connectivity == "steiner":
            qc_structured = synth_clifford_full(cliff_operator, method="AG")
            qc_to_route = apply_Steiner_synthesis(qc_structured, qubit_connectivity)
        elif method_connectivity == "greedy":
            qc_to_route = synth_clifford_full(cliff_operator, method="greedy")
        else:
            raise ValueError(f"Unknown method_connectivity: {method_connectivity}")

        cmap = CouplingMap(qubit_connectivity)
        trivial_layout = list(range(n_qubits))

        best_qc = None
        best_metric = float("inf")

        rng = np.random.default_rng(seed_base)
        trials = int(transpilation_trials) if transpilation_trials is not None else 1
        trials = max(1, trials)

        for _ in range(trials):
            seed_t = int(rng.integers(0, 2**31 - 1))

            qc_candidate = transpile(
                qc_to_route,
                coupling_map=cmap,
                optimization_level=3,
                basis_gates=None,
                initial_layout=trivial_layout,
                layout_method="sabre",
                routing_method="sabre",
                seed_transpiler=seed_t,
            )

            metric = cnot_depth_qiskit(qc_candidate, swap_cost=swap_cost)
            if metric < best_metric:
                best_metric = metric
                best_qc = qc_candidate

        if best_qc is None:
            raise RuntimeError("Transpilation failed to produce any candidate circuit.")

        final_qc = decompose_dense_clifford_gates(best_qc)

    # Convert to your internal gate-list format
    new_gate_list = qiskit_to_gate_list(final_qc)

    # Recompute mappings under the optimized gate list
    original_mappings = diagonalization_result["mappings"]
    input_paulis = [m["pauli"] for m in original_mappings]

    new_mappings = []
    active_measured_qubits = set()

    for p_str in input_paulis:
        final_s, sign = conjugate_pauli_by_circuit(p_str, new_gate_list)
        z_qubits = [q for q, ch in enumerate(final_s) if ch == "Z"]
        active_measured_qubits.update(z_qubits)
        new_mappings.append({
            "pauli": p_str,
            "final_Z_string": final_s,
            "sign": int(sign),
            "z_qubits_to_parity": z_qubits,
        })

    return {
        "measured": sorted(active_measured_qubits),
        "gates": new_gate_list,
        "mappings": new_mappings,
    }

def optimize_clifford_decomposition_global_setting(
    diag_results: List[Dict],
    n_qubits: int,
    qubit_connectivity: Optional[List[Tuple[int, int]]] = None,
    transpilation_trials: int = 40,
    method_connectivity: str = "greedy",
    swap_cost: int = 3,
    seed_base: int = 0,
) -> Dict:
    """
    Given a list of per-block diagonalization_result dicts (all-to-all synthesis),
    concatenate them and run one global routing/optimization pass, selecting the
    best candidate by min CNOT-depth.
    """
    merged = merge_diagonalization_results(diag_results)
    return optimize_clifford_decomposition(
        diagonalization_result=merged,
        n_qubits=n_qubits,
        qubit_connectivity=qubit_connectivity,
        transpilation_trials=transpilation_trials,
        method_connectivity=method_connectivity,
        swap_cost=swap_cost,
        seed_base=seed_base,
    )
    
def pauli_to_zx(p: str):
    n = len(p)
    z = np.zeros(n, dtype=np.uint8)
    x = np.zeros(n, dtype=np.uint8)
    for j, ch in enumerate(p):
        if ch == 'I':
            pass
        elif ch == 'X':
            x[j] = 1
        elif ch == 'Z':
            z[j] = 1
        elif ch == 'Y':
            z[j] = 1; x[j] = 1
        else:
            raise ValueError(f"Invalid Pauli char {ch}")
    return z, x
    
def qibo_circuit_from_gate_list(n: int, gate_list: list, density_matrix=False):
    """
    Builds a Qibo circuit from the gate list.
    """
    c = models.Circuit(nqubits=n, density_matrix=density_matrix)
    
    for name, args in gate_list:
        if name == "H":
            (q,) = args; c.add(gates.H(q))
        elif name == "S":
            (q,) = args; c.add(gates.S(q))
        elif name == "Sdg":
            (q,) = args; c.add(gates.SDG(q)) 
        elif name == "X":
            (q,) = args; c.add(gates.X(q))
        elif name == "Y":
            (q,) = args; c.add(gates.Y(q))
        elif name == "Z":
            (q,) = args; c.add(gates.Z(q))
        elif name == "CNOT":
            a, b = args; c.add(gates.CNOT(a, b))
        elif name == "CZ":
            a, b = args; c.add(gates.CZ(a, b))
        elif name == "SWAP":
            a, b = args; c.add(gates.SWAP(a, b))
        elif name == "ID":
            (q,) = args; c.add(gates.I(q))
        else:
            raise ValueError(f"Unsupported gate: {name}")
            
    return c

def qiskit_to_gate_list(qc: QuantumCircuit) -> List[Tuple[str, Tuple[int, ...]]]:
    """
    Converts a Qiskit QuantumCircuit directly into the list-of-tuples format.
    Supports standard gates including Sdg, X, Y, Z, ID.
    """
    gate_list = []
    for instruction in qc.data:
        name = instruction.operation.name.lower()
        indices = tuple(qc.find_bit(q).index for q in instruction.qubits)
        
        if name == 'h': gate_list.append(('H', indices))
        elif name == 's': gate_list.append(('S', indices))
        elif name == 'sdg': gate_list.append(('Sdg', indices))
        elif name == 'x': gate_list.append(('X', indices))
        elif name == 'y': gate_list.append(('Y', indices))
        elif name == 'z': gate_list.append(('Z', indices))
        elif name == 'cx': gate_list.append(('CNOT', indices))
        elif name == 'cz': gate_list.append(('CZ', indices))
        elif name == 'swap': gate_list.append(('SWAP', indices))
        elif name == 'id': gate_list.append(('ID', indices))
        elif name == 'barrier': continue
        else:
            # Fallback for any other valid Clifford gate Qiskit might use
            # We treat them as unsupported if not in our explicit list
            raise ValueError(f"Unsupported gate in optimized circuit: {name}")
    return gate_list

def rank_gf2(M: np.ndarray) -> int:
    M = M.copy() % 2
    m, n = M.shape
    r = 0
    for c in range(n):
        pivot = None
        for i in range(r, m):
            if M[i, c]:
                pivot = i; break
        if pivot is None:
            continue
        if pivot != r:
            M[[r, pivot]] = M[[pivot, r]]
        for i in range(m):
            if i != r and M[i, c]:
                M[i, :] ^= M[r, :]
        r += 1
        if r == m: break
    return r

def zx_to_pauli(z: np.ndarray, x: np.ndarray) -> str:
    s = []
    for zj, xj in zip(z, x):
        zj = int(zj); xj = int(xj)
        if zj == 0 and xj == 0: s.append('I')
        elif zj == 0 and xj == 1: s.append('X')
        elif zj == 1 and xj == 0: s.append('Z')
        else: s.append('Y')
    return ''.join(s)