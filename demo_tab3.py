import numpy as np
from shadowgrouping.benchmark import load_dict
from shadowgrouping.molecules import available_molecules, available_molecules_latex, available_molecules_E_GS
from shadowgrouping.hamiltonian import mappings
from demo_tab1 import TexTable

folder = "data/tab1/"
folder_single_shot = "data/tab3/"

def read_energy_estimations(filedir,molecule_name,mapping_name,method_name,num_reps=None,use_one_norm=False):
    """ Reads in all files in >>filedir<< of the form {molecule_name}_molecule_{mapping_name}_{method_name}_energy_estimations.txt
        and returns the RMSE of the values in the file.
        If num_reps is None (default), only a single file is read-in. Else, a running index from range(num_reps) is appended to method_name.
        Defaults to the two-norm of the deviations. One-norm can be activated by setting use_one_norm to True.
    """
    if num_reps is None:
        filename = filedir + "{}_molecule_{}_{}_energy_estimations.txt".format(molecule_name,mapping_name,method_name)
        with open(filename,"r") as f:
            f.readline() # throw-away line
            E_GS = float(f.readline().strip().split()[-1])
        estimations = np.loadtxt(filename)
    else:
        assert isinstance(num_reps,int), "num_reps either has to be a non-negative integer or None-type."
        assert num_reps >=0, "num_reps either has to be a non-negative integer or None-type."
        filename = filedir + "{}_molecule_{}_{}_{}_energy_estimations.txt".format(molecule_name,mapping_name,method_name,"{}")
        with open(filename.format(0),"r") as f:
            f.readline() # throw-away line
            E_GS = float(f.readline().strip().split()[-1])
        estimations = np.loadtxt(filename.format(0))
        for i in range(1,num_reps):
            estimations = np.append(estimations,np.loadtxt(filename.format(i)))
            
    temp = np.abs(estimations-E_GS) if use_one_norm else (estimations-E_GS)**2
    RMSE = np.mean(temp) if use_one_norm else np.sqrt(np.mean(temp))
    RMSE_std = np.std(temp) if use_one_norm else np.sqrt(np.std(temp))
    return RMSE, RMSE_std/np.sqrt(len(temp)), E_GS


if __name__=="__main__":
    rmse_dict_methods = {}
    std_dict_methods  = {}
    for ind,molecule_name in enumerate(available_molecules):
        for map_name in mappings.keys():
            try:
                temp_dict = load_dict(folder + "{}_molecule_{}_empirical.txt".format(molecule_name,map_name))
                if ind==0:
                    method_names = list(temp_dict.keys())
                    #method_names = [method for method in method_names if method not in ["AEQuO-inadaptive","ShadowAdaptive"]]
            except Exception as e:
                print(e)
                continue
            else:
                for key in method_names:
                    if key != "RandomPaulis":
                        continue
                    vals = temp_dict.get(key,None)
                    if vals is None:
                        continue
                    rmse_dict_methods[(molecule_name,map_name,key)] = vals[0]
                    std_dict_methods[(molecule_name,map_name,key)]  = vals[1]

    # add estimate of the sinlge shot estimator to dictionaries
    for (molecule_name,map_name,_) in rmse_dict_methods.copy().keys():
        rmse, std, _ = read_energy_estimations(folder_single_shot,molecule_name,"L1_check",map_name)
        rmse_dict_methods[(molecule_name,map_name,"SingleShot")] = rmse
        std_dict_methods[(molecule_name,map_name,"SingleShot")] = std


    energies = {m: available_molecules_E_GS[i] for i,m in enumerate(available_molecules)}
    table = TexTable(rmse_dict_methods,available_molecules,list(mappings.keys()),["RandomPaulis","SingleShot"],energies,available_molecules_latex,std_dict_methods)

    for molecule_name in available_molecules:
        print(molecule_name)
        for map_name in mappings.keys():
            print(map_name)
            for label in table.methods:
                key = (molecule_name,map_name,label)
                rmse_data = rmse_dict_methods.get(key,None)
                if rmse_data is None:
                    continue
                print(label,np.round(1000*rmse_dict_methods[key],1),"+/-",np.round(1000*std_dict_methods[key],1))
