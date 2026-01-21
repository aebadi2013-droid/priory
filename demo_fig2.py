import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from shadowgrouping.measurement_schemes import N_delta
from os import mkdir
from os.path import isdir

#######################################################
##### params #####
#######################################################
delta = np.logspace(-10,0,1000)
delta = delta[delta<0.5]
numerator = N_delta(delta)
# shadow hi
# test branch
n_min, n_max = min(numerator), max(numerator)
#######################################################
# color palette chosen from
# https://colorpalettes.net/color-palette-4361/
#######################################################
color_line   = "#EF7663"
color_filled = "#E7E1C4"
color_dotted = "#BCC8B4"
#######################################################

if __name__ == "__main__":
    # create temporary folder for storing outputs
    if not isdir("generated_figures"):
        mkdir("generated_figures")
    plt.figure(figsize=(6,3),dpi=400)
    plt.loglog(delta,numerator,color=color_line)
    #plt.text(6e-4,2e3,"$N_\delta$",color="blue",fontsize="x-large")
    plt.fill_between(delta,0.9*n_min,numerator,color=color_filled,alpha=0.4)
    plt.text(1e-8,100,r'$\epsilon_\mathrm{sys} \leq \epsilon_\mathrm{stat}$',fontsize=22)
    for x in np.arange(1,7):
        sigma = 1-erf(x/np.sqrt(2))
        plt.vlines(sigma,0.9*n_min,1e4,colors=color_dotted,linestyles="dotted",alpha=0.8)
        plt.text(0.1*sigma,1.05*n_min,r'${}\sigma$'.format(x),color=color_dotted,fontsize=16)
    plt.xlim(left=min(delta),right=0.5)
    plt.ylim(0.9*min(numerator),1e3)
    plt.xlabel('inconfidence level $\delta$',fontsize="x-large")
    plt.ylabel('collected samples $N_i$',fontsize="x-large")
    plt.xticks([1e-10,1e-8,1e-6,1e-4,1e-2],["$10^{"+str(i)+"}$" for i in np.arange(-10,-1,2)],fontsize="x-large")
    plt.yticks([1e2,1e3],["$10^{"+str(i)+"}$" for i in np.arange(2,4)],fontsize="x-large")
    #plt.minorticks_off()
    plt.savefig("generated_figures/fig2_demo.png",bbox_inches="tight")
