import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def calculate_support_statistics_and_generate_uniform(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    support_sizes = []
    coefficients = []
    pauli_strings = []

    # Calculate support sizes and collect coefficients
    for i in range(0, len(lines), 2):
        try:
            pauli_term = lines[i].strip()
            weight = float(complex(lines[i + 1].strip()).real)

            # Calculate support size (number of non-identity terms in the Pauli string)
            support_size = sum(1 for char in pauli_term if char != 'I')
            support_sizes.append(support_size)
            coefficients.append(weight)
            pauli_strings.append(pauli_term)
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid pair at lines {i + 1} and {i + 2}: {e}")

    # Calculate average and standard deviation of support sizes
    #avg_support = np.mean(support_sizes)
    #std_support = np.std(support_sizes)

    avg_support = 5.8
    std_support = 1

    print(f"Average support size: {avg_support}, Standard deviation: {std_support}")

    # Generate random Pauli strings with uniformly-distributed support sizes
    output_lines = []
    for i in range(len(pauli_strings)):
        random_support_size = np.random.randint(max(1, int(avg_support - std_support)), int(avg_support + std_support) + 1)
        random_support_size = max(1, min(random_support_size, 14))  # Clamp between 1 and 14
        # Generate a random Pauli string with the given support size
        random_pauli = list('I' * len(pauli_strings[i]))
        non_identity_indices = np.random.choice(len(random_pauli), random_support_size, replace=False)
        for idx in non_identity_indices:
            random_pauli[idx] = np.random.choice(['X', 'Y', 'Z'])

        random_pauli_string = ''.join(random_pauli)
        output_lines.append(f"{random_pauli_string}\n")
        output_lines.append(f"({coefficients[i]:.6f}+0j)\n")

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    print(f"Random Pauli strings with uniformly-distributed support sizes written to {output_file}")


input_file = "d:/priori/Hamiltonians_v2/BeH2_sto3g_14qubits/jw.txt"
output_file = "d:/priori/Hamiltonians_v2/BeH2_sto3g_14qubits/jw_uniformrandompaulis_M5.8D1.0.txt"
calculate_support_statistics_and_generate_uniform(input_file, output_file)


def calculate_support_statistics_and_generate_random(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")

    with open(input_file, 'r') as f:
        lines = f.readlines()

    support_sizes = []
    coefficients = []
    pauli_strings = []

    # Calculate support sizes and collect coefficients
    for i in range(0, len(lines), 2):
        try:
            pauli_term = lines[i].strip()
            weight = float(complex(lines[i + 1].strip()).real)

            # Calculate support size (number of non-identity terms in the Pauli string)
            support_size = sum(1 for char in pauli_term if char != 'I')
            support_sizes.append(support_size)
            coefficients.append(weight)
            pauli_strings.append(pauli_term)
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid pair at lines {i + 1} and {i + 2}: {e}")

    # Calculate average and standard deviation of support sizes
    #avg_support = np.mean(support_sizes)
    #std_support = np.std(support_sizes)

    avg_support = 5.8
    std_support = 1

    print(f"Average support size: {avg_support}, Standard deviation: {std_support}")

    # Generate random Pauli strings with Gaussian-distributed support sizes
    output_lines = []
    for i in range(len(pauli_strings)):
        random_support_size = int(np.random.normal(avg_support, std_support))
        #random_support_size = max(1, random_support_size)  # Ensure at least one non-identity term
        random_support_size = max(1, min(random_support_size, 14))  # Clamp between 1 and 14

        # Generate a random Pauli string with the given support size
        random_pauli = list('I' * len(pauli_strings[i]))
        non_identity_indices = np.random.choice(len(random_pauli), random_support_size, replace=False)
        for idx in non_identity_indices:
            random_pauli[idx] = np.random.choice(['X', 'Y', 'Z'])

        random_pauli_string = ''.join(random_pauli)
        output_lines.append(f"{random_pauli_string}\n")
        output_lines.append(f"({coefficients[i]:.6f}+0j)\n")

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    print(f"Random Pauli strings with Gaussian-distributed support sizes written to {output_file}")


input_file = "d:/priori/Hamiltonians_v2/BeH2_sto3g_14qubits/jw.txt"
output_file = "d:/priori/Hamiltonians_v2/BeH2_sto3g_14qubits/jw_gaussianrandompaulis_M3.0D1.2.txt"
calculate_support_statistics_and_generate_random(input_file, output_file)

def plot_weight_distribution(hamiltonian_terms, weights , title="Gaussian Bell Curve of Weights"):
    # Calculate Gaussian fit
    mean = np.mean(weights)
    std_dev = np.std(weights)
    x = np.linspace(min(weights), max(weights), 500)
    gaussian_curve = norm.pdf(x, mean, std_dev)

    plt.figure(figsize=(10, 6))

    # Plot the Gaussian bell curve
    plt.plot(x, gaussian_curve, label='Gaussian Bell Curve', color='red', linewidth=2)

    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Probability Density')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()



print("Uniform distribution of weights")
def uniform_weights(input_file, output_file, a =0.5, b=1.5):
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

            # Replace the real part of the weight with a uniformly-distributed value
            new_weight = np.random.uniform(a, b)
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
    plt.figure(figsize=(10, 6))
    plt.plot(hamiltonian_terms, weights, 'o', label='Uniformly Distributed Weights', color='blue')
    plt.xlabel('Hamiltonian Terms')
    plt.ylabel('Weight Value')
    plt.title('Uniform Distribution of Weights')
    plt.show()

input_file = "d:/priori/Hamiltonians_v2/BeH2_sto3g_14qubits/jw.txt"
output_file = "d:/priori/Hamiltonians_v2/BeH2_sto3g_14qubits/jw_uniform0.5-1.5.txt"
uniform_weights(input_file, output_file, a=0.5, b=1.5)

print("Gaussian distribution of weights")
def gaussian_weights(input_file, output_file, center =1, width=0.5):
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
            new_weight = np.random.normal(center, width)
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
    plot_weight_distribution(hamiltonian_terms, weights, title="Gaussian Distribution of Weights")

input_file = "d:/priori/Hamiltonians_v2/BeH2_sto3g_14qubits/jw.txt"
output_file = "d:/priori/Hamiltonians_v2/BeH2_sto3g_14qubits/jw_gaussian_1-5.txt"
gaussian_weights(input_file, output_file, center=1, width=0.5)



