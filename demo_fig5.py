from shadowgrouping.measurement_schemes import N_delta, SettingSampler
from shadowgrouping.molecules import CHEMICAL_ACCURACY, available_molecules
from shadowgrouping.hamiltonian import mappings
import numpy as np
import matplotlib.pyplot as plt

folder_ressource_estims = "data/fig5/"
file_name_base = "molecule_{}_mapping_{}_basis_{}.txt"
OGM_folder = "../OverlappedGrouping/CutSet/"
name = "OGM_{}_{}{}.txt"
delta = 0.02

def get_Ham_details(fname):
    with open(fname,"r") as f:
        num_qubits = int(f.readline().strip().split("=")[-1])
        num_terms  = int(f.readline().strip().split("=")[-1])
        one_norm   = float(f.readline().strip().split("=")[-1])
    return num_qubits,num_terms,one_norm

# load |h| for each molecule and basis choice.
# This norm is independent of the fermion-to-qubit mapping, but the number of terms may vary. Hence, we minimize over this number.
# We also keep track of the largest qubit number for convenience
num_max = 0
norm_dict = {"sto3g":{},"6-31g":{}}
for basis in norm_dict.keys():
    for molecule_name in available_molecules:
        if molecule_name.find("_")!=-1:
            continue
        M_min = int(1e12)
        for map_name in mappings.keys():
            temp_file_name = folder_ressource_estims+file_name_base.format(molecule_name,map_name,basis)
            try:
                num_qubits, M, norm = get_Ham_details(temp_file_name)
            except Exception as e:
                print(e)
            else:
                M_min = min(M_min,M)
                num_max = max(num_max,num_qubits)
        norm_dict[basis][molecule_name] = (num_qubits, M_min, norm)
        
y_min, y_max = np.infty, 0
f = N_delta(delta)**2
for (basis,dict_mol),color in zip(norm_dict.items(),("r","b")):
    if basis=="sto3g":
        cliques = {}
        # get the number of overlapping cliques and minimize over mappings
        for molecule,(n,_,_) in dict_mol.items():
            num_cliques = np.infty
            for map_name in mappings.keys():
                method = SettingSampler(np.random.rand(10,n),np.random.rand(10),OGM_folder+name.format(molecule,n,map_name.lower()))
                num_cliques = min(len(method.p),num_cliques)
            cliques[molecule] = num_cliques
    bottom = False
    jitter_direction = -1
    for molecule,(n,nterms,norm) in dict_mol.items():
        norm*= norm*f/CHEMICAL_ACCURACY**2
        y_min, y_max = min(y_min,norm), max(y_max,norm*nterms)
        if molecule == "NH3":
            continue
        jitter = 0.25*jitter_direction if n==14 or n==26 else 0
        jitter_direction *= -1 if n==14 or n==26 else 1
        plt.errorbar(n+jitter,norm,yerr=np.array([[0],[nterms*norm]]),color=color)
        if basis=="sto3g":
            plt.plot(n+jitter,cliques[molecule]*norm,"g*")
        if bottom:
            plt.text(n+jitter-0.5,norm*0.5,molecule,fontsize=14)
        else:
            plt.text(n+jitter-0.5,norm*nterms*1.5,molecule,fontsize=14)
        bottom = not bottom
    # exception for NH3
    n, nterms, norm = dict_mol["NH3"]
    norm_rescaled = norm**2/CHEMICAL_ACCURACY**2*f
    plt.errorbar(n,norm_rescaled,yerr=np.array([[0],[nterms*norm_rescaled]]),label=basis,color=color)
    if basis=="sto3g":
        plt.plot(n,cliques["NH3"]*norm_rescaled,"g*")
    plt.text(n+0.15,norm_rescaled*10**(np.log10(nterms)/2),"NH3",fontsize=14)
plt.xlim(3,num_max*1.1)
plt.xlabel("Number of qubits",fontsize=18)
plt.xticks(fontsize=16)
plt.semilogy()
plt.ylim(0.8*y_min,1.5*y_max)
plt.ylabel("Required measurements",fontsize=18)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(loc="upper left",fontsize=16)
plt.savefig("generated_figures/fig5_demo.png",bbox_inches="tight")
