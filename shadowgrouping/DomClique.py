import numpy as np
import networkx as nx
from shadowgrouping.hamiltonian import int_to_char
from shadowgrouping.measurement_schemes import Shadow_Grouping


##########################################################################################
### Helper functions #####################################################################
##########################################################################################
def hit_by(O, P):
    """Returns whether O is hit by P"""
    for o, p in zip(O, P):
        if not (o == 0 or p == 0 or o == p):
            return False
    return True

def setting_to_str(arr):
    out = ""
    for a in np.array(arr).flatten():
        out += str(a)
    return out

# equation 6 from manuscript
N_delta = lambda delta: 4*(2*np.sqrt(-np.log(delta)) + 1)**2

##########################################################################################
### Measurement schemes used for benchmark ###############################################
##########################################################################################


class DomClique(Shadow_Grouping):
    def __init__(self, observables, weights):
        """
        Initialize the QubitGraphAnalyzer with observables and weights.
        
        Args:
            observables (list): A list of observables.
            weights (list): A list of weights corresponding to the observables.
        """
        if len(observables) != len(weights):
            raise ValueError("The length of 'observables' and 'weights' must be the same.")
        
        self.observables = observables
        self.weights = weights
        self.G = nx.Graph()
        self.twe, self.tcwe = 0, 0  # Edge weight calculations
        self.nwe, self.tnwe = 0, 0  # Node weight calculations
        self.lwe, self.tlwe = 0, 0  # Local weight calculations
        self.wavg = np.zeros(len(observables))
        self.neighbournum = np.zeros(len(observables))
        self.nodeweight = np.zeros(len(observables))
        self._build_graph()
        self.is_sampling = False

    def reset(self):
        """
        Reset all attributes to their initial state, clearing the graph and any computed properties.
        """
        # Clear the graph
        self.G.clear()

        # Reset graph-related weights and totals
        self.twe, self.tcwe = 0, 0  # Edge weight calculations
        self.nwe, self.tnwe = 0, 0  # Node weight calculations
        self.lwe, self.tlwe = 0, 0  # Local weight calculations

        # Reset node attributes
        self.wavg = np.zeros(len(self.observables))
        self.neighbournum = np.zeros(len(self.observables))
        self.nodeweight = np.zeros(len(self.observables))

    def find_setting(self):
        graph = self._build_graph()  # Fixed: don't pass `self` here
        sortedindices = self.sort_nodes(graph)
        dominatingset = self.greedy_ndominating_set()
        clique = self.maximal_cliques()
        # transform into Pauli string for compatibility with parent class
        setting, clique = self._clique_to_Pauli_observable(clique)
        return setting,{}

    def _build_graph(self):
        """Build the graph by adding edges based on commutativity."""
        for i in range(len(self.observables)):
            tavg, conn = 0, 0
            for j in range(i + 1, len(self.observables)):
                if hit_by(self.observables[i], self.observables[j]):
                    we = round(np.abs(self.weights[i]) * np.abs(self.weights[j]), 5)
                    self.G.add_edge(i, j, weight=we)
                    self.twe += we
                    self.nwe += np.abs(self.weights[i]) + np.abs(self.weights[j])
                    self.lwe += np.abs(self.weights[i]) * np.abs(self.weights[j])
                    tavg += np.abs(self.weights[i]) * np.abs(self.weights[j])
                    conn += 1

            self.wavg[i] = 0 if conn == 0 else tavg / conn
            self.neighbournum[i] = conn
            self.nodeweight[i] = tavg

        # Calculate theoretical total edge and node weights
        for i in range(len(self.observables)):
            for j in range(i + 1, len(self.observables)):
                self.tcwe += round(np.abs(self.weights[i]) * np.abs(self.weights[j]), 5)
                self.tnwe += np.abs(self.weights[i]) + np.abs(self.weights[j])
                self.tlwe += np.abs(self.weights[i]) * np.abs(self.weights[j])

        # Return the built graph
        return self.G

    def sort_nodes(self, neighbournum, nodeweight, wavg):
        """
        Sort nodes based on number of neighbours, total weight, and average weight, 
        and print the sorted results.
        """
        # Sort nodes based on number of neighbours
        nsorted_indices = sorted(range(len(neighbournum)), key=lambda x: neighbournum[x], reverse=True)
        print("Nodes sorted by number of neighbours:")
        for index in nsorted_indices:
            print(f"Node {index}: number of neighbours = {neighbournum[index]}")

        # Sort nodes based on node total weight
        wsorted_indices = sorted(range(len(nodeweight)), key=lambda x: nodeweight[x], reverse=True)
        print("Nodes sorted by node total weights:")
        for index in wsorted_indices:
            print(f"Node {index}: total weight = {nodeweight[index]}")

        # Sort nodes based on their average weight
        asorted_indices = sorted(range(len(wavg)), key=lambda x: wavg[x], reverse=True)
        print("Nodes sorted by average weight:")
        for index in asorted_indices:
            print(f"Node {index}: average weight = {wavg[index]}")

        return nsorted_indices, wsorted_indices, asorted_indices

    def greedy_ndominating_set(self):
        """
        Find a dominating set using a greedy algorithm based on node degrees.
        
        Returns:
            set: A dominating set of nodes determined by node degrees.
        """
        node_degrees = dict(self.G.degree())  # Calculate node degrees
        nsorted_indices = sorted(node_degrees, key=node_degrees.get, reverse=True)  # Sort nodes by degree

        ndominating_set = set()
        ncovered_nodes = set()

        for node in nsorted_indices:
            if len(ncovered_nodes) == len(self.G.nodes):
                break
            if node not in ncovered_nodes:
                ndominating_set.add(node)
                ncovered_nodes.add(node)
                ncovered_nodes.update(self.G.neighbors(node))

        return ndominating_set

    def greedy_wdominating_set(self):
        """
        Find a dominating set using a greedy algorithm based on node weights.
        
        Returns:
            set: A dominating set of nodes determined by node weights.
        """
        wsorted_indices = sorted(range(len(self.nodeweight)), key=lambda x: self.nodeweight[x], reverse=True)

        wdominating_set = set()
        wcovered_nodes = set()

        for node in wsorted_indices:
            if len(wcovered_nodes) == len(self.G.nodes):
                break
            if node not in wcovered_nodes:
                wdominating_set.add(node)
                wcovered_nodes.add(node)
                wcovered_nodes.update(self.G.neighbors(node))

        return wdominating_set

    def greedy_adominating_set(self):
        """
        Find a dominating set using a greedy algorithm based on average node weights.
        
        Returns:
            set: A dominating set of nodes determined by average node weights.
        """
        asorted_indices = sorted(range(len(self.wavg)), key=lambda x: self.wavg[x], reverse=True)

        adominating_set = set()
        acovered_nodes = set()

        for node in asorted_indices:
            if len(acovered_nodes) == len(self.G.nodes):
                break
            if node not in acovered_nodes:
                adominating_set.add(node)
                acovered_nodes.add(node)
                acovered_nodes.update(self.G.neighbors(node))

        return adominating_set

    def maximal_cliques_around_node(self, G, node):
        neighbors = list(G.neighbors(node))
        subgraph_nodes = neighbors + [node]
        subgraph = G.subgraph(subgraph_nodes)
        cliques = list(nx.find_cliques(subgraph))  # Ensure correct function is used
        return cliques

    def maximal_cliques(self):
        MaxCliques = []  # Initialize the list of maximal cliques
        ndominating_set = self.greedy_ndominating_set()

        for v in ndominating_set:
            cliques = self.maximal_cliques_around_node(self.G, v)

            cliques_sorted = sorted(cliques, key=lambda clique: len(clique), reverse=True)
            uncovered_nodes = set(self.G.neighbors(v))

            while uncovered_nodes:
                for clique in cliques_sorted:
                    if uncovered_nodes & set(clique):
                        MaxCliques.append(clique)
                        uncovered_nodes.difference_update(clique)
                        break

        return MaxCliques






  # Define _clique_to_Pauli_observable within this class
    def _clique_to_Pauli_observable(self, clique):
        """ Helper function to return the sampled clique to a Pauli string."""
        clique = np.array(clique[1:]) - 1 if clique[0] == 0 else np.array(clique) - 1
        clique_members = self.obs[clique]
        setting = np.max(clique_members, axis=0)
        filtered = setting != 0
        clique_members[clique_members == 0] = 4  # Ignore identities
        double_check = np.min(clique_members, axis=0)
        assert np.allclose(setting[filtered], double_check[filtered]), (
            f"The clique {clique} does not allow for a qubit-wise commutativity-compatible "
            "measurement setting."
        )
        return setting, clique



