import os
import numpy as np

def process_hamiltonian_file(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    for i in range(0, len(lines), 2):
        try:
            pauli_term = lines[i].strip()
            weight = float(complex(lines[i + 1].strip()).real)

            # Replace the real part of the weight with a Gaussian-distributed value
            new_weight = np.random.normal(0, 1)

            # Append the Pauli term and new weight to the output
            output_lines.append(f"{pauli_term}\n")
            output_lines.append(f"({new_weight:.6f}+0j)\n")
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid pair at lines {i + 1} and {i + 2}: {e}")

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

input_file = "d:/priori/Hamiltonians_v2/LiH_sto3g_12qubits/jw.txt"
output_file = "d:/priori/Hamiltonians_v2/LiH_sto3g_12qubits/jw_gaussian.txt"
process_hamiltonian_file(input_file, output_file)
