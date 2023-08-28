import numpy as np
from shadowgrouping.benchmark import load_dict
from shadowgrouping.molecules import available_molecules, available_molecules_latex, available_molecules_E_GS
from shadowgrouping.hamiltonian import mappings

folder = "data/tab1/"

class TexTable():
    """ Convenience class for bringing the single energy estimations from various sources into one digestiable LaTeX table.
        The defining features of the table are the molecules with their respective TeX-names and ground-state energy E_GS.
        For each of them, we further split into various provided mappings and measurement allocation methods as well.
        For each combination of (molecule,mapping,method) there has to be an entry in the dictionary RMSE_dict, otherwise it is omitted.
        If values for standard deviation are provided, also plots them to file.
    """
    def __init__(self,RMSE_dict,molecule_names,mapping_names,method_names,E_GS,molecule_texnames,std=None):
        self.dict = RMSE_dict
        self.molecules = molecule_names
        self.mappings  = mapping_names
        self.methods   = method_names
        self.energies  = E_GS
        self.molec_tex = molecule_texnames
        self.print_std = std is not None
        self.stds      = std if self.print_std else {}
        
    def get_rmse_rows(self):
        """ Run through all combinations of (molecule,mapping) and track the best performaning method for each. """
        best_method_idxs = []
        rows = []
        rows_std = []
        for molecule in self.molecules:
            rows_mol = []
            stds_mol = []
            for mapping in self.mappings:
                temp = []
                temp_stds = []
                for method in self.methods:
                    temp.append(self.dict.get((molecule,mapping,method),np.infty))
                    if temp[-1] < np.infty:
                        temp_stds.append(self.stds.get((molecule,mapping,method),-1))
                    else:
                        temp_stds.append(-1)
                rows_mol.append(np.array(temp))
                stds_mol.append(np.array(temp_stds))
                # get the index of the minimum value over the methods
                temp = np.argmin(rows_mol[-1])
                if not rows_mol[-1][temp] < np.infty:
                    temp = -1
                best_method_idxs.append(temp)
            rows.append(np.array(rows_mol))
            rows_std.append(np.array(stds_mol))
        return np.array(rows), np.array(rows_std), np.array(best_method_idxs)

    def plot_values_tex_table(self,outfile):
        rows, stds, idxs = self.get_rmse_rows()
        running_ind = 0
        # put data to file
        with open(outfile,"w") as f:
            # header lines
            f.write("\\begin{tabular}{c|c|"+"c"*len(self.methods)+"}\nMolecule & \multirow{2}{*}{Enc.}")
            for method in self.methods:
                f.write(" & " + method)
            f.write("\\\\ \n")
            # second header line
            f.write("($E_\\mathrm{GS}$ [mHa]) & " + "& "*len(self.methods))
            f.write("\\\\ \\hline\n")
            # molecule rows        
            for block, block_stds, molecule, molecule_tag in zip(rows, stds, self.molecules, self.molec_tex):
                E_GS = self.energies.get(molecule)
                if E_GS is None:
                    running_ind += len(self.mappings)
                    continue
                first_cols = ["\\multicolumn{1}{c|}{}"]*len(self.mappings)
                first_cols[0] = first_cols[0][:-1] + "\\multirow{"+ "{}".format(len(self.mappings)) +"}{*}{\\shortstack[c]{" + molecule_tag + "\\\\ {:.2f}".format(E_GS) + "$\\times 10^3$}}}"
                for row, row_std, start, enc in zip(block, block_stds, first_cols, self.mappings):
                    f.write(start) # first col
                    f.write(" & \\multicolumn{1}{c|}{" + enc + "}")
                    for j,(val,std) in enumerate(zip(row,row_std)):
                        std = 1000*std if std > 0 else None
                        rounder = RoundRelevant(1000*val,std)
                        if j==idxs[running_ind]:
                            # highlighted value indicating minimum
                            f.write(rounder.print_val_std(prepend=" & \\textbf{",append="}"))
                        else:
                            if np.isinf(val):
                                f.write(" & ")
                            else:
                                f.write(rounder.print_val_std(prepend=" & "))
                    running_ind += 1
                    # add \\ command for TeX
                    if enc == self.mappings[-1]:
                        f.write(" \\\\ \\hline\n")
                    else:
                        f.write(" \\\\\n")
            # footer lines
            f.write("\\end{tabular}\n")
        return

rmse_dict = {}
std_dict  = {}
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
                vals = temp_dict.get(key,None)
                if vals is None:
                    continue
                rmse_dict[(molecule_name,map_name,key)] = vals[0]
                std_dict[(molecule_name,map_name,key)]  = vals[1]


energies = {m: available_molecules_E_GS[i] for i,m in enumerate(available_molecules)}
table = TexTable(rmse_dict,available_molecules,list(mappings.keys()),method_names,energies,available_molecules_latex,std_dict)

for molecule_name in available_molecules:
    print(molecule_name)
    for map_name in mappings.keys():
        print(map_name)
        for label in table.methods:
            key = (molecule_name,map_name,label)
            rmse_data = rmse_dict.get(key,None)
            if rmse_data is None:
                continue
            print(label,np.round(1000*rmse_dict[key],1),"+/-",np.round(1000*std_dict[key],1))
