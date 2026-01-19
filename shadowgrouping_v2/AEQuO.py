import numpy as np
import itertools
from shadowgrouping_v2.hamiltonian import int_to_char
from shadowgrouping_v2.measurement_schemes import Shadow_Grouping

# Source code has been sourced from https://github.com/AndrewJena/VQE_measurement_optimization
# The following class has been modified (and a few hidden member functions added by promoting helper functions) to be compatible with the rest of this API
# Compatability with the original source code has been verified
# No helper function below has been modified

class AEQuO(Shadow_Grouping):
    """ Greedy bucket-filling algorithm, explained in https://arxiv.org/abs/2110.15339. Adapted from the corresponding repo.
        Algorithm parameters:
        adaptiveness_L (int, >=0): degree of adaptiveness to measurement outcomes. A value of 0 means no adaptivity, larger values increase it.
        interval_isometry_l (float, >=0): A value of 0 signifies equidistant updates, larger values skew this towards more updates early on.
        budget (int, >=0): Predefined measurement budget. A value of zero means non-adaptive allocation and overwrites the other parameters.
    """
    
    def __init__(self,observables,weights,offset,adaptiveness_L=0,interval_skewness_l=0,budget=0):
        super().__init__(observables,weights,0.1,None) # Third input is epsilon (which is not relevant) and fourth is weight function (also not relevant)
        # init formatting for AEQuO
        self.offset = offset
        self.__observables_to_AEQuO_list()  # Just an helper function that turns the ShadowGrouping format to the one digestible by AEQuO. 
        assert isinstance(adaptiveness_L,int), "adaptiveness_L-value has to be integer."
        assert adaptiveness_L >= 0, "Adaptiveness_L-value has to non-negative, but was {}.".format(adaptiveness_L)
        assert interval_skewness_l >= 0, "Interval_skewness_l-value has to non-negative, but was {}.".format(interval_skewness_l)
        assert isinstance(budget,int), "budget-value has to be integer."
        assert budget >= 0, "budget-value has to be non-negative, but was {}.".format(budget)
        self.shots = budget # New variable exclusive of AEQuO class
        self.L = adaptiveness_L + 1 if self.shots > 0 else 1 # New variable exclusive of AEQuO class
        self.l = interval_skewness_l # New variable exclusive of AEQuO class
        self.update_steps = Ll_updates(self.L,self.l,self.shots) # New variable exclusive of AEQuO class
        self.is_adaptive = self.L > 1
        self.is_sampling = self.is_adaptive
        
        # self.outcome_dict stores counts of bit pairs corresponding to outcomes of pairs of simultaneously measured observables
        # convention: AEQuO takes the identity as well, so we have to count it in self.num_obs
        self.outcome_dict = np.array([[{(1,1):0,(1,-1):0,(-1,1):0,(-1,-1):0} for a1 in range(self.num_obs+1)] for a0 in range(self.num_obs+1)])
        # partition the commutativity graph using largest-degree first algorithm
        self.cliques = LDF(qubitwise_commutation_graph(self.paulis))
        # initialize variance estimates - these are stored as self.V internally
        self.sampled_cliques = []
        self.sampled_cliques_since_update = []
        self.outcomes_since_update = []
        self.update_variance_estimate()
        # get two independent iterators over these cliques
        cliques1,cliques2 = itertools.tee(self.cliques,2)
        # speed-up tricks for non-overlapping cliques or non-adaptive settings
        # Second option (i.e., 'else') is standard for AEQuO: we allow for overlapping cliques and adaptivity
        if not self.update_steps&set(range(1,self.shots)) and not any(set(aa1)&set(aa2) for aa1,aa2 in itertools.product(cliques1,cliques2) if not aa1==aa2):
            # self.setting_function is the generator of a new clique (i.e., measurement setting)
            self.setting_function = self.non_overlapping_bayes_min_var
            # helper arrays
            self.clique_counts = [0]*len(self.cliques)
            self.clique_stds = [np.sqrt(self.V[clique][:,clique].sum()) for clique in self.cliques]
        else:
            # self.setting_function is the generator of a new clique (i.e., measurement setting)
            self.setting_function = self.overlapping_bayes_min_var
        self.S = np.zeros((self.num_obs+1,self.num_obs+1),dtype=int) # N_obs x N_obs
        #self.Ones = [np.ones((i,i),dtype=int) for i in range(self.num_obs+1+1)] # increasing arrays of size i x i for i = 1, ... , N_obs
        self.index_set = set(range(self.num_obs+1))
        self.update_steps = np.sort(list(self.update_steps))[1:]  # exclude the "0"-entry
        
        return
        
    def reset(self):
        super().reset()
        self.outcome_dict = np.array([[{(1,1):0,(1,-1):0,(-1,1):0,(-1,-1):0} for a1 in range(self.num_obs+1)] for a0 in range(self.num_obs+1)])
        self.S = np.zeros((self.num_obs+1,self.num_obs+1),dtype=int) # N_obs x N_obs
        self.index_set = set(range(self.num_obs+1))
        self.sampled_cliques = []
        self.sampled_cliques_since_update = []
        self.outcomes_since_update = []
        self.update_variance_estimate()
        if self.setting_function == self.non_overlapping_bayes_min_var:
            self.clique_counts = [0]*len(self.cliques)
            self.clique_stds = [np.sqrt(self.V[clique][:,clique].sum()) for clique in self.cliques]
        return
    
    def find_setting(self):
        if len(self.sampled_cliques) == self.shots:
            print("Warning! Starting to inquire more samples than predefined measurement budget of {} for this class.".format(self.shots))
            print("Further measurement settings can be accessed, however, no update step for the variance estimates is performed.")
        if len(self.sampled_cliques) in self.update_steps and len(self.sampled_cliques) > 0:
            # update variance estimate accordingly
            self.update_variance_estimate()
            self.sampled_cliques_since_update = []
            self.outcomes_since_update = []
        
        clique = self.setting_function() # just sampling one clique from the fixed self.cliques
        # transform into Pauli string for compatibility with parent class
        setting, clique = self._clique_to_Pauli_observable(clique)
        # update class counters
        self.N_hits[clique] += 1
        return setting
        
    def overlapping_bayes_min_var(self):
        # The standard version of the setting_function, i.e., the function that samples a new setting (a.k.a. clique) from a fixed set
        S1 = self.S + 1 # Adding one sample to every single pair of observables
        s = 1/(self.S.diagonal()|(self.S.diagonal()==0))
        s1 = 1/S1.diagonal()
        factor = self.num_obs+1-np.count_nonzero(self.S.diagonal())
        S1[range(self.num_obs+1),range(self.num_obs+1)] = [a if a != 1 else -factor for a in S1.diagonal()]
        V1 = self.V*(self.S*s*s[:,None] - S1*s1*s1[:,None]) # Variances
        V2 = 2*self.V*(self.S*s*s[:,None] - self.S*s*s1[:,None]) # Co-variances
        cliques1 = iter(self.cliques) # this is an iterator of self.cliques
        # The next line is where AEQuO "prioritizes cliques that are statistically likely to have bigger contributions to the error" (p. 5 of paper)
        clique = sorted(max(cliques1,key=lambda xx : V1[xx][:,xx].sum()+V2[xx][:,list(self.index_set.difference(xx))].sum()))
        self.sampled_cliques.append(clique)
        self.sampled_cliques_since_update.append(clique)
        self.S[np.ix_(clique,clique)] += 1
        
        return clique
    
    def non_overlapping_bayes_min_var(self):
        # The alternative version of the setting_function, i.e., the function that samples a new setting (a.k.a. clique) from a fixed set
        """ If the partition has no overlapping sets, we can speed up the allocation of measurements. """
        # Since there are no overlaps, there are no correlations, so we do not have to worry about covariances.
        # Just pick clique with largest total standard deviation
        # Again, this is where AEQuO "prioritizes cliques that are statistically likely to have bigger contributions to the error" (p. 5 of paper)
        max_index = np.argmax(self.clique_stds)
        clique = self.cliques[max_index]
        self.clique_counts[max_index] += 1
        self.clique_stds[max_index] *= ((self.clique_counts[max_index]-1) or not (self.clique_counts[max_index]-1))/(self.clique_counts[max_index]+1)
        self.S[np.ix_(clique,clique)] += 1#self.Ones[len(clique)]
        self.sampled_cliques.append(clique)
        self.sampled_cliques_since_update.append(clique)
        
        return clique
    
    def update_variance_estimate(self,update_V=True):
        """ Updates variance graph calculated with Bayesian estimates. """
        if len(self.sampled_cliques_since_update) != len(self.outcomes_since_update):
            print("Warning at step {}!".format(len(self.sampled_cliques)))
            print("Not every allocated clique (there are {} allocations since last update and {} outcomes right now) received an outcome.".format(len(self.sampled_cliques_since_update),len(self.outcomes_since_update)))
            print("Skipping the update.")
        else:
            # preprocessing: updated array of measurement outcome counts
            for clique,outcome in zip(self.sampled_cliques_since_update,self.outcomes_since_update):
                for (a0,c0),(a1,c1) in itertools.product(zip(clique,outcome),repeat=2):
                    self.outcome_dict[a0,a1][(c0,c1)] += 1
            # update variance graph
            if update_V:
                self.V = bayes_variance_graph(self.outcome_dict,self.coeffs).adj
        return
    
    def receive_outcome(self,outcome):
        """ Convenience function to obtain the outcome of the previously chosen clique. """
        if len(self.sampled_cliques_since_update) == len(self.outcomes_since_update):
            print("Warning at step {}! Trying to feed an outcome for which no clique has been allocated yet.".format(len(self.sampled_cliques)))
            print("Given outcome has not been incorporated into scheme.")
        else:
            outcome = self._process_outcome(self.sampled_cliques_since_update[len(self.outcomes_since_update)],outcome)
            self.outcomes_since_update.append(outcome)
        return
        
    def __observables_to_AEQuO_list(self):
        """ Helper function that turns the format to the one digestible by AEQuO. """
        pauli_strings = ["I"*self.obs.shape[1]]
        for obs in self.obs:
            string = ""
            for o in obs:
                string += int_to_char[o]
            pauli_strings.append(string)
        self.paulis = string_to_pauli(pauli_strings)
        self.coeffs = np.hstack(([self.offset],self.w))
        return

    def _clique_to_Pauli_observable(self,clique):
        """ Helper function that returns the sampled clique to a Pauli string (since qubit-wise commutativity is assumed).
            Performs a check whether this string actually commutes with all observables within the sampled clique.
            Returns a valid measurement setting as required for the parent class and the altered clique for further internal usage.
        """
        # the commutativity graph includes the identity term - we can simply drop it
        clique = np.array(clique[1:]) - 1 if clique[0] == 0 else np.array(clique) - 1
        clique_members = self.obs[clique]
        setting = np.max(clique_members, axis=0)
        filtered = setting != 0
        clique_members[clique_members==0] = 4 # throw away identities
        # Now, np.min(clique_members,axis=0) has to match up with its np.max(...) except where setting == 0
        double_check = np.min(clique_members, axis=0)
        assert np.allclose(setting[filtered],double_check[filtered]), "The clique {} does not allow for a qubit-wise commutativity-compatible measurement setting.".format(clique)
        return setting, clique
    
    def _process_outcome(self,clique,outcome):
        """ Helper function that calculates the outcome of each clique member within clique directly.
        """
        includes_identity = clique[0] == 0
        clique = np.array(clique[1:]) - 1 if includes_identity else np.array(clique) - 1
        clique_members = self.obs[clique]
        outcome = np.repeat(outcome.reshape((1,-1)),len(clique_members),axis=0)
        outcome[clique_members==0] = 1 # set to 1 if outside the support the respective to mask it out, hit observables
        data = [1] if includes_identity else []
        data = data + list(np.prod(outcome,axis=1).astype(int))
        return data
    
    def get_energy(self):
        estim_mean = 0.0
        for i in range(self.paulis.paulis()):
            estim_mean += self.coeffs[i]*naive_Mean(self.outcome_dict[i,i])
        return estim_mean

############################################################
############################################################
############ HELPER FUNCTIONS ##############################
############################################################
############################################################

# PAULIS

# a class for storing sets of Pauli operators as pairs of symplectic matrices
class pauli:
    def __init__(self,X,Z):
        # Inputs:
        #     X - (numpy.array) - X-part of Pauli in symplectic form with shape (p,q)
        #     Z - (numpy.array) - Z-part of Pauli in symplectic form with shape (p,q)
        if X.shape != Z.shape:
            raise Exception("X- and Z-parts must have same shape")
        self.X = X
        self.Z = Z

    # check whether self has only X component
    def is_IX(self):
        # Outputs:
        #     (bool) - True if self has only X componenet, False otherwise
        return not np.any(self.Z)

    # check whether self has only Z component 
    def is_IZ(self):
        # Outputs:
        #     (bool) - True if self has only Z componenet, False otherwise
        return not np.any(self.X)

    # check whether the set of Paulis are pairwise commuting on every qubit
    def is_qubitwise_commuting(self):
        # Outputs:
        #     (bool) - True if self is pairwise qubitwise commuting set of Paulis
        p = self.paulis()
        PP = [self.a_pauli(i) for i in range(p)]
        return not any(any((PP[i0].X[0,i2]&PP[i1].Z[0,i2])^(PP[i0].Z[0,i2]&PP[i1].X[0,i2]) for i2 in range(self.qubits())) for i0,i1 in itertools.combinations(range(p),2))

    # pull out the ath Pauli from self
    def a_pauli(self,a):
        # Inputs: 
        #     a - (int) - index of Pauli to be returned
        # Outputs:
        #     (pauli) - the ath Pauli in self
        return pauli(np.array([self.X[a,:]]),np.array([self.Z[a,:]]))

    # count the number of Paulis in self
    def paulis(self):
        # Output: (int)
        return self.X.shape[0]

    # count the number of qubits in self
    def qubits(self):
        # Outputs: (int)
        return self.X.shape[1]

    # delete Paulis indexed by aa
    def delete_paulis_(self,aa):
        # Inputs: 
        #     aa - (list of int)
        if type(aa) is int:
            self.X = np.delete(self.X,aa,axis=0)
            self.Z = np.delete(self.Z,aa,axis=0)
        else:
            for a in sorted(aa,reverse=True):
                self.X = np.delete(self.X,a,axis=0)
                self.Z = np.delete(self.Z,a,axis=0)
        return self

    # return self after deletion of qubits indexed by aa
    def delete_qubits_(self,aa):
        # Inputs: 
        #     aa - (list of int)
        if type(aa) is int:
            self.X = np.delete(self.X,aa,axis=1)
            self.Z = np.delete(self.Z,aa,axis=1)
        else:
            for a in sorted(aa,reverse=True):
                self.X = np.delete(self.X,a,axis=1)
                self.Z = np.delete(self.Z,a,axis=1)

    # return deep copy of self
    def copy(self):
        # Outputs: (pauli)
        X = np.array([[self.X[i0,i1] for i1 in range(self.qubits())] for i0 in range(self.paulis())],dtype=bool)
        Z = np.array([[self.Z[i0,i1] for i1 in range(self.qubits())] for i0 in range(self.paulis())],dtype=bool)
        return pauli(X,Z)

    # print string representation of self
    def print(self):
        sss = pauli_to_string(self)
        if type(sss) is str:
            print(sss)
        else:
            for ss in sss:
                print(ss)

    # print symplectic representation of self
    def print_symplectic(self):
        for i in range(self.paulis()):
            print(''.join(str(int(i1)) for i1 in self.X[i,:]),''.join(str(int(i1)) for i1 in self.Z[i,:]))



# convert a collection of strings (or single string) to a pauli object
def string_to_pauli(sss):
    # Inputs:
    #     sss - (list{str}) or (str) - string representation of Pauli
    # Outputs:
    #     (pauli) - Pauli corresponding to input string(s)
    XDict = {"I":0,"X":1,"Y":1,"Z":0}
    ZDict = {"I":0,"X":0,"Y":1,"Z":1}
    if type(sss) is str:
        X = np.array([[XDict[s] for s in sss]],dtype=bool)
        Z = np.array([[ZDict[s] for s in sss]],dtype=bool)
        return pauli(X,Z)
    else:
        X = np.array([[XDict[s] for s in ss] for ss in sss],dtype=bool)
        Z = np.array([[ZDict[s] for s in ss] for ss in sss],dtype=bool)
        return pauli(X,Z)

# convert a pauli object to a collection of strings (or single string)
def pauli_to_string(P):
    # Inputs:
    #     P - (pauli) - Pauli to be stringified
    # Outputs:
    #     (list{str}) - string representation of Pauli
    X,Z = P.X,P.Z
    ssDict = {(0,0):"I",(0,1):"Z",(1,0):"X",(1,1):"Y"}
    if P.paulis() == 0:
        return ''
    elif P.paulis() == 1:
        return ''.join(ssDict[(X[0,i],Z[0,i])] for i in range(P.qubits()))
    else:
        return [''.join(ssDict[(X[i0,i1],Z[i0,i1])] for i1 in range(P.qubits())) for i0 in range(P.paulis())]

# the symplectic inner product of two pauli objects (each with a single Pauli)
def qubitwise_inner_product(P0,P1):
    # Inputs:
    #     P0 - (pauli) - must have shape (1,q)
    #     P1 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (int) - qubitwise inner product of Paulis modulo 2
    if (P0.paulis() != 1) or (P1.paulis() != 1):
        raise Exception("Qubitwise inner product only works with pair of single Paulis")
    if P0.qubits() != P1.qubits():
        raise Exception("Qubitwise inner product only works if Paulis have same number of qubits")
    return any((P0.X[0,i]&P1.Z[0,i])^(P0.Z[0,i]&P1.X[0,i]) for i in range(P0.qubits()))

# the product of two pauli objects
def pauli_product(P0,P1):
    # Inputs:
    #     P0 - (pauli) - must have shape (1,q)
    #     P1 - (pauli) - must have shape (1,q)
    # Outputs:
    #     (pauli) - product of Paulis
    if P0.paulis() != 1 or P1.paulis() != 1:
        raise Exception("Product can only be calculated for single Paulis")
    return pauli(np.logical_xor(P0.X,P1.X),np.logical_xor(P0.Z,P1.Z))


# GRAPHS

# a class for storing graphs as adjacency matrices
#     since we are dealing with covariance matrices with both vertex and edge weights,
#     this is a suitable format to capture that complexity
class graph:
    # Inputs:
    #     adj_mat - (numpy.array) - (weighted) adjacency matrix of graph
    #     dtype   - (numpy.dtype) - data type of graph weights
    def __init__(self,adj_mat=np.array([]),dtype=float):
        self.adj = adj_mat.astype(dtype)

    # adds a vertex to self
    def add_vertex_(self,c=1):
        # Inputs:
        #     c - (float) - vertex weight
        if len(self.adj) == 0:
            self.adj = np.array([c])
        else:
            m0 = np.zeros((len(self.adj),1))
            m1 = np.zeros((1,len(self.adj)))
            m2 = np.array([[c]])
            self.adj = np.block([[self.adj,m0],[m1,m2]])

    # weight a vertex
    def lade_vertex_(self,a,c):
        # Inputs:
        #     a - (int)   - vertex to be weighted
        #     c - (float) - vertex weight
        self.adj[a,a] = c

    # weight an edge
    def lade_edge_(self,a0,a1,c):
        # Inputs:
        #     a0 - (int)   - first vertex
        #     a1 - (int)   - second vertex
        #     c  - (float) - vertex weight
        self.adj[a0,a1] = c
        self.adj[a1,a0] = c

    # returns a set of the neighbors of a given vertex
    def neighbors(self,a):
        # Inputs:
        #     a - (int) - vertex for which neighbors should be returned
        # Outputs:
        #     (list{int}) - set of neighbors of vertex a
        aa1 = set([])
        for i in range(self.ord()):
            if (a != i) and (self.adj[a,i] != 0):
                aa1.add(i)
        return aa1

    # returns list of all edges in self
    def edges(self):
        # Outputs:
        #     (list{list{int}}) - list of edges in self
        aaa = []
        for i0,i1 in itertools.combinations(range(self.ord()),2):
            if i1 in self.neighbors(i0):
                aaa.append([i0,i1])
        return aaa

    # check whether a collection of vertices is a clique in self
    def clique(self,aa):
        # Inputs:
        #     aa - (list{int}) - list of vertices to be checked for clique
        # Outputs:
        #     (bool) - True if aa is a clique in self; False otherwise
        for i0,i1 in itertools.combinations(aa,2):
            if self.adj[i0,i1] == 0:
                return False
        return True

    # returns the degree of a given vertex
    def degree(self,a):
        # Inputs:
        #     a - (int) - vertex for which degree should be returned
        # Outputs:
        #     (int) - degree of vertex a
        return np.count_nonzero(self.adj[a,:])

    # returns the number of vertices in self
    def ord(self):
        # Outputs:
        #     (int) - number of vertices in self
        return self.adj.shape[0]

    # print adjacency matrix representation of self
    def print(self):
        for i0 in range(self.ord()):
            print('[',end=' ')
            for i1 in range(self.ord()):
                s = self.adj[i0,i1]
                if str(s)[0] == '-':
                    print(f'{self.adj[i0,i1]:.2f}',end=" ")
                else:
                    print(' '+f'{self.adj[i0,i1]:.2f}',end=" ")
            print(']')

    # print self as a list of vertices together with their neighbors
    def print_neighbors(self):
        for i0 in range(self.ord()):
            print(i0,end=": ")
            for i1 in self.neighbors(i0):
                print(i1,end=" ")
            print()

    # return a deep copy of self
    def copy(self):
        # Outputs:
        #     (graph) - deep copy of self
        return graph(np.array([[self.adj[i0,i1] for i1 in range(self.ord())] for i0 in range(self.ord)]))

# returns all non-empty cliques in a graph
def nonempty_cliques(A):
    # Inputs:
    #     A - (graph) - graph for which all cliques should be found
    # Outputs:
    #     (list{list{int}}) - a list containing all non-empty cliques in A
    p = A.ord()
    aaa = set([frozenset([])])
    for i in range(p):
        iset = set([i])
        inter = A.neighbors(i)
        aaa |= set([frozenset(iset|(inter&aa)) for aa in aaa])
    aaa.remove(frozenset([]))
    return list([list(aa) for aa in aaa])

# reduces a clique covering of a graph by removing cliques with lowest weight
def post_process_cliques(A,aaa,k=1):
    # Inputs:
    #     A   - (graph)           - varaince graph from which weights of cliques can be obtained
    #     aaa - (list{list{int}}) - a clique covering of the Hamiltonian
    #     k   - (int)             - number of times each vertex must be covered
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which cover A
    p = A.ord()
    V = A.adj
    s = np.array([sum([i in aa for aa in aaa]) for i in range(p)])
    D = {}
    for aa in aaa:
        D[str(aa)] = V[aa][:,aa].sum()
    aaa1 = aaa.copy()
    aaa1 = list(filter(lambda x : all(a>=(k+1) for a in s[aa]),aaa1))
    while aaa1:
        aa = min(aaa1,key=lambda x : D[str(x)])
        aaa.remove(aa)
        aaa1.remove(aa)
        s -= np.array([int(i in aa) for i in range(p)])
        aaa1 = list(filter(lambda x : all(a>=(k+1) for a in s[aa]),aaa1))
    return aaa

# returns a largest-degree-first clique partition of a graph
def LDF(A):
    # Inputs:
    #     A - (graph) - graph for which partition should be found
    # Outputs:
    #     (list{list{int}}) - a list containing cliques which partition A
    p = A.ord()
    remaining = set(range(p))
    N = {}
    for i in range(p):
        N[i] = A.neighbors(i)
    aaa = []
    while remaining:
        a = max(remaining,key=lambda x : len(N[x]&remaining))
        aa0 = set([a])
        aa1 = N[a]&remaining
        while aa1:
            a2 = max(aa1,key=lambda x : len(N[x]&aa1))
            aa0.add(a2)
            aa1 &= N[a2]
        aaa.append(aa0)
        remaining -= aa0
    return [sorted(list(aa)) for aa in aaa]

# returns the qubitwise commutation graph of a given Pauli
def qubitwise_commutation_graph(P):
    # Inputs:
    #     P - (pauli) - Pauli to check for qubitwise commutation relations
    # Outputs:
    #     (graph) - an edge is weighted 1 if the pair of Paulis qubitwise commute
    p = P.paulis()
    return graph(np.array([[1-qubitwise_inner_product(P.a_pauli(i0),P.a_pauli(i1)) for i1 in range(p)] for i0 in range(p)]))

# ESTIMATED PHYSICS FUNCTIONS

def naive_Mean(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    if (x0+x1) == 0:
        return 0
    return (x0-x1)/(x0+x1)

# Bayesian estimation of mean from samples
def bayes_Mean(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    return (x0-x1)/(x0+x1+2)

# Bayesian estimation of variance from samples
def bayes_Var(xDict):
    # Inputs:
    #     xDict - (Dict) - number of ++/+-/-+/-- outcomes for single Pauli
    # Outputs:
    #     (float) - Bayesian variance of mean
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    return 4*((x0+1)*(x1+1))/((x0+x1+2)*(x0+x1+3))

# Bayesian estimation of covariance from samples
def bayes_Cov(xyDict,xDict,yDict):
    # Inputs:
    #     xyDict - (Dict) - number of ++/+-/-+/-- outcomes for pair of Paulis
    #     xDict  - (Dict) - number of ++/+-/-+/-- outcomes for first Pauli
    #     xDict  - (Dict) - number of ++/+-/-+/-- outcomes for second Pauli
    # Outputs:
    #     (float) - Bayesian estimate of mean
    xy00,xy01,xy10,xy11 = xyDict[(1,1)],xyDict[(1,-1)],xyDict[(-1,1)],xyDict[(-1,-1)]
    x0,x1 = xDict[(1,1)],xDict[(-1,-1)]
    y0,y1 = yDict[(1,1)],yDict[(-1,-1)]
    p00 = 4*((x0+1)*(y0+1))/((x0+x1+2)*(y0+y1+2))
    p01 = 4*((x0+1)*(y1+1))/((x0+x1+2)*(y0+y1+2))
    p10 = 4*((x1+1)*(y0+1))/((x0+x1+2)*(y0+y1+2))
    p11 = 4*((x1+1)*(y1+1))/((x0+x1+2)*(y0+y1+2))
    return 4*((xy00+p00)*(xy11+p11) - (xy01+p01)*(xy10+p10))/((xy00+xy01+xy10+xy11+4)*(xy00+xy01+xy10+xy11+5))

# approximates the variance graph using Bayesian estimates
def bayes_variance_graph(X,cc):
    # Inputs:
    #     X  - (numpy.array{dict}) - array for tracking measurement outcomes
    #     cc - (list{float})       - coefficients of Hamiltonian
    # Outputs:
    #     (numpy.array{float}) - variance graph calculated with Bayesian estimates
    p = len(cc)
    return graph(np.array([[(cc[i0]**2)*bayes_Var(X[i0,i0]) if i0==i1 else cc[i0]*cc[i1]*bayes_Cov(X[i0,i1],X[i0,i0],X[i1,i1]) for i1 in range(p)] for i0 in range(p)]))

# SIMULATION ALGORITHMS

# convert from L,l notation to set of update steps
def Ll_updates(L,l,shots):
    # Inputs:
    #     L     - (int) - number of sections into which shots should be split
    #     l     - (int) - exponential scaling factor for size of sections
    #     shots - (int) - total number of shots required
    # Outputs:
    #     (set{int}) - set containing steps at which algorithm should update
    r0_shots = shots/sum([(1+l)**i for i in range(L)])
    shot_nums = [round(r0_shots*(1+l)**i) for i in range(L-1)]
    shot_nums.append(shots-sum(shot_nums))
    return set([0]+list(itertools.accumulate(shot_nums))[:-1])



