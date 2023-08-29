import numpy as np
import matplotlib.pyplot as plt
from shadowgrouping.measurement_schemes import N_delta
from shadowgrouping.benchmark import load_dict_provable
from shadowgrouping.hamiltonian import mappings
from shadowgrouping.molecules import available_molecules
from os import mkdir
from os.path import isdir
import argparse

######### Folder default for data storage. Can be overriden by optional argument to script #########
folder = "data/fig4/"
print_only = ["Shadow","Adaptive","Overlapped","Single"]
N_extended = [70,98]
delta = 0.02
####################################################################
########### style params etc #######################################
####################################################################
parser = argparse.ArgumentParser(description="Recreate the plots from a given data folder. Defaults to showing the data from the manuscript plots but can be altered to use custom data. To do so, use the -f <folder_location> option.")
parser.add_argument("-f","--folder", type=str,
                    help="Provide the folder where the data resides. Default: {}".format(folder),
                    default=folder
                   )

threshold = int(np.ceil(N_delta(delta)))
plt.style.use('tableau-colorblind10')
cycler = plt.rcParams["axes.prop_cycle"]
colorlist = list(cycler)
schemes_plotted = {"ShadowBernstein": (colorlist[0]["color"],"dotted"),
                   "ShadowBernstein-truncated": (colorlist[0]["color"],"solid"),
                   "ShadowDerandomization": (colorlist[4]["color"],"dashed"),
                   "AdaptivePaulis": (colorlist[1]["color"],"dashed"),
                   "OverlappedGrouping": (colorlist[2]["color"],"dashdot")
                  }
label_names = {"ShadowBernstein": "ShadowG.",
               "ShadowBernstein-truncated": "SG (trunc.)",
               "ShadowDerandomization": "SG (Derand)",
               "AdaptivePaulis": "Adaptive",
               "OverlappedGrouping": "OGM"
              }

def insert_value_to_ticks(arr,val,threshold=0.1):
    """ Insert a given value <val> into a sorted array <arr>.
        Must lie within the range of the array, however, the first and last elements of <arr> are cut-off afterwards.
        When inserted, array values that are within <threshold> x array-range are thrown away.
        Returns the new filtered array with the value inserted at the last index.
        Also returns the corresponding indices used for sorting the new array.
    """
    assert np.allclose(arr,np.sort(arr)), "Please provide a sorted array."
    assert arr[0] <= val and arr[-1] >= val, "Please provide a value within the range of arr."
    # check whether values are close in range
    R = arr[-1] - arr[0]
    arr = arr[1:-1]
    kept = np.bitwise_or(arr >= val+threshold*R, arr <= val-threshold*R)
    new_arr = np.append(arr[kept],val)
    inds = np.argsort(new_arr)
    return new_arr, inds
####################################################################

if __name__ == "__main__":
    args = parser.parse_args()
    folder = args.folder
    # create temporary folder for storing outputs
    if not isdir("generated_figures"):
        mkdir("generated_figures")
    for molecule_name in available_molecules:
        plt.figure(figsize=(14,4))
        maximum, minimum = -np.inf, np.inf
        for i,map_name in enumerate(mappings.keys()):
            # load data
            if i==0:
                ax = plt.subplot(131)
            else:
                ax_alt = plt.subplot(131+i,sharey=ax)
            try:
                Nsteps, epsilons = load_dict_provable(folder + "{}_molecule_{}_provable.txt".format(molecule_name,map_name))
                Nsteps = np.append(N_extended,Nsteps)
            except:
                pass
            else:
                hnorm = []
                for scheme in label_names.keys():
                    vals = epsilons.get(scheme)
                    if vals is None:
                        continue
                    vals = np.append(np.full(len(N_extended),vals[0]),vals)
                    color, style = schemes_plotted[scheme]
                    plt.semilogx(Nsteps,vals,color=color,linestyle=style,label=scheme,linewidth=2.5)
                    maximum, minimum = max(maximum,vals[0]), min(minimum,vals[-1])
            # plot data
            plt.xlim(Nsteps[0],Nsteps[-1])
            plt.ylim(0.95*minimum,1.05*maximum)
            plt.text(0.75*Nsteps[-1],0.95*maximum,chr(ord("a")+i)+")",fontsize="xx-large")
            plt.xticks([100,500,1000],[100,500,1000],fontsize="x-large")
            plt.fill_between([Nsteps[0],threshold],np.full(2,0),np.full(2,1.05*maximum),alpha=0.1,color="gray")
            if i==0:
                ticksrange, _ = plt.yticks()
                if molecule_name == "H2":
                    ticksrange = np.round(ticksrange,1)
                else:
                    ticksrange = ticksrange.astype(int)
                ticksrange_new, inds = insert_value_to_ticks(ticksrange,maximum,threshold=0.1)
                locs = ticksrange_new[inds]
                labels = ticksrange_new[:-1]
                if molecule_name != "H2":
                    labels = labels.astype(int)
                labels = list(labels) + ["$|\mathbf{h}|_{\ell_1}$"]
                labels = np.array(labels,dtype=str)[inds]
                plt.ylabel("$\epsilon_{\mathrm{stat}} + \epsilon_{\mathrm{sys}}$ [Ha]",fontsize="x-large")
                plt.yticks(ticksrange_new[inds], labels,fontsize="x-large")
            else:
                plt.tick_params('y', labelleft=False, length=0)
            if i==1:
                plt.xlabel("Number of allocated measurements $N_\mathrm{tot}$",fontsize="x-large")
            elif i==2:
                plt.legend(bbox_to_anchor=(1,0.5), loc="center left",fontsize="large")
            plt.grid()
        plt.subplots_adjust(wspace=0.05)
        plt.savefig("generated_figures/fig4_demo_{}.png".format(molecule_name),bbox_inches="tight")
        
