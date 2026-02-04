import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_weight_distribution(hamiltonian_terms, weights):
    # Calculate Gaussian fit
    mean = np.mean(weights)
    std_dev = np.std(weights)
    x = np.linspace(min(weights), max(weights), 500)
    gaussian_curve = norm.pdf(x, mean, std_dev)

    plt.figure(figsize=(10, 6))

    # Plot the Gaussian bell curve
    plt.plot(x, gaussian_curve, label='Gaussian Bell Curve', color='red', linewidth=2)

    plt.title('Gaussian Bell Curve of Weights')
    plt.xlabel('Weight Value')
    plt.ylabel('Probability Density')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_hamiltonian_file(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    hamiltonian_terms = []  # Collect Hamiltonian terms here
    weights = []  # Collect weights here
    for i in range(0, len(lines), 2):
        try:
            pauli_term = lines[i].strip()
            weight = float(complex(lines[i + 1].strip()).real)

            # Replace the real part of the weight with a Gaussian-distributed value
            new_weight = np.random.normal(0, 1)
            hamiltonian_terms.append(pauli_term)  # Store the Hamiltonian term
            weights.append(new_weight)  # Store the new weight

            # Append the Pauli term and new weight to the output
            output_lines.append(f"{pauli_term}\n")
            output_lines.append(f"({new_weight:.6f}+0j)\n")
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid pair at lines {i + 1} and {i + 2}: {e}")

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    # Plot the distribution of weights
    plot_weight_distribution(hamiltonian_terms, weights)

input_file = "d:/priori/Hamiltonians_v2/LiH_sto3g_12qubits/jw.txt"
output_file = "d:/priori/Hamiltonians_v2/LiH_sto3g_12qubits/jw_gaussian.txt"
process_hamiltonian_file(input_file, output_file)
