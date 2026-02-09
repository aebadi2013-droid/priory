from qiskit_nature.drivers.second_quantization import ElectronicStructureDriverType, ElectronicStructureMoleculeDriver
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper, ParityMapper #BravyiKitaevSuperFastMapper
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_nature.algorithms import GroundStateEigensolver
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliOp, SummedOp

import numpy as np, os
from scipy.sparse.linalg import eigsh
from shadowgrouping_v2.helper_functions import char_to_int

# list of available mappings
mappings = {"JW": JordanWignerMapper, "BK": BravyiKitaevMapper, "Parity": ParityMapper } # "BKSF": BravyiKitaevSuperFastMapper

def get_groundstate(molecule,
                    mapping=JordanWignerMapper,
                    basis="sto3g",
                    driver_type=ElectronicStructureDriverType.PYSCF,
                    driver_kwargs = None,
                    verbose = False
                   ):
    """
        Takes a molecule description as input and returns the energy and the coefficients in the comp. basis of the ground-state.
        Additionally, the state_dict is returned that labels the coefficients to the underlying basis state.
        Additional arguments are:
            basis (str):    Basis set for the chemistry calculations. See qiskit documentation for a list of available types.
            driver_type:    Choose a driver type from the list of supported drivers in the qiskit documentation
            mapping:        Fermion-to-qubit mapping. See qiskit documentation for the list of supported mappings
            driver_kwargs:  Optional arguments passed to the chosen driver of type Union[Dict[str, Any], NoneType]
            verbose (bool): Plot the details of the groundstate to console
    """
    # init driver
    driver = ElectronicStructureMoleculeDriver(molecule, basis=basis, driver_type=driver_type, driver_kwargs=driver_kwargs)
    es_problem = ElectronicStructureProblem(driver)
    qubit_converter = QubitConverter(mapping())
    
    numpy_solver = NumPyMinimumEigensolver()
    calculator = GroundStateEigensolver(qubit_converter, numpy_solver)
    result = calculator.solve(es_problem)    
    energy, state, state_dict = result.eigenenergies[0], result.eigenstates[0].primitive.data, result.eigenstates[0].primitive.to_dict()
    if verbose:
        for f in result.formatted():
            print(f)
    
    return energy, state, state_dict

class Hamiltonian():
    """ Helper class to turn a list of Pauli operators with accompanying weights into a (sparse) Hamiltonian and diagonalize it.
        Code copied and modified from https://github.com/charleshadfield/adaptiveshadows/blob/main/python/hamiltonian.py
    """

    def __init__(self, weights, observables):
        self.weights = weights
        self.observables = observables

    def SummedOp(self):
        paulis = []
        for P, coeff_P in zip(self.observables,self.weights):
            paulis.append(coeff_P * PauliOp(Pauli(P)))
        return SummedOp(paulis)
    
    def LDN_factor_calculator(self):
        attenuation = [1 - 4/3 * self.p_array[i] for i in range(len(self.observables[0]))]
        LDN_factors_array = []
        
        for i in range(len(self.observables)):
            factors = []
            for j in range(len(self.observables[0])):
                pauli_char = self.observables[i][j]
                factors.append(attenuation[j] if pauli_char != 'I' else 1)
            LDN_factor = np.prod(factors)
            LDN_factors_array.append(LDN_factor)
        
        self.LDN_factors = LDN_factors_array
        return
    
    def SummedOp_LDN(self):
        paulis = []
        for P, coeff_P, LDN_factor in zip(self.observables,self.weights, self.LDN_factors):
            paulis.append(LDN_factor * coeff_P * PauliOp(Pauli(P)))
        return SummedOp(paulis)

    def ground(self, sparse=False):
        if not sparse:
            mat = self.SummedOp().to_matrix()
            evalues, evectors = np.linalg.eigh(mat)
        else:
            mat = self.SummedOp().to_spmatrix()
            evalues, evectors = eigsh(mat, which='SA')
            # SA looks for algebraically small evalues
        index = np.argmin(evalues)
        ground_energy = evalues[index]
        ground_state = evectors[:,index]
        return ground_energy, ground_state

def load_pauli_list(folder_hamiltonian,molecule_name,basis_name,encoding,verbose=False,sparse=False,diagonalize=True):
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
    print(f"full path to floder",full_folder_path)
    file_name = None
    file_energy = None
    for file in available_files:
        if file.lower().startswith(encoding[:2].lower()) and "grouped" not in file:
            file_name = file
        elif file == "ExactEnergy.txt":
            file_energy = file

    assert file_name is not None, f"File not found for encoding '{encoding}'."
    assert file_energy is not None, "File not found for ground-state energy."
    
    if diagonalize:
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
        print("check for print")
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

def get_pauli_list(molecule,
                   mapping=JordanWignerMapper,
                   basis="sto3g",
                   driver_type=ElectronicStructureDriverType.PYSCF,
                   driver_kwargs = None,
                   verbose = False
                  ):
    """
        Takes a molecule description as input and returns the corresponding qubit Hamiltonian P.
        Here, P[:,1] is the list of coefficients for the Pauli operators in P[:,0], respectively.
        Additional arguments are:
            basis (str):    Basis set for the chemistry calculations. See qiskit documentation for a list of available types.
            driver_type:    Choose a driver type from the list of supported drivers in the qiskit documentation
            mapping:        Fermion-to-qubit mapping. See qiskit documentation for the list of supported mappings
            driver_kwargs:  Optional arguments passed to the chosen driver of type Union[Dict[str, Any], NoneType]
            verbose (bool): Plot the details of the 2nd quantisation and qubit conversion to console
    """
    # init driver
    driver = ElectronicStructureMoleculeDriver(molecule, basis=basis, driver_type=driver_type, driver_kwargs=driver_kwargs)
    es_problem = ElectronicStructureProblem(driver)
    # 2nd quantisation
    second_q_op = es_problem.second_q_ops()
    if verbose:
        print(second_q_op["ElectronicEnergy"])
        print()
    # fermion-to-qubit mapping
    qubit_converter = QubitConverter(mapping())
    qubit_op = qubit_converter.convert(second_q_op["ElectronicEnergy"])
    if verbose:
        print("Qubit Operator")
        print(qubit_op)
        print()
    # qiskit-to-numpy export
    P = []
    for q in qubit_op.to_pauli_op():
        P.append( [str(q.primitive),q.coeff] )
    return np.array(P,dtype=object)
