import numpy as np
from qibo import models, gates
from shadowgrouping.measurement_schemes import hit_by
from shadowgrouping.hamiltonian import int_to_char


class StateSampler():
    """ Convenience class that holds a fixed state of length 2**num_qubits. The latter number is inferred automatically.
        Provides a sampling method that obtains samples from the state in the chosen basis.
        
        Input:
        - state, numpy array of size 2**N with N = num_qubits. Coefficients are to be specified in the computational basis.
    """
    def __init__(self,state):
        self.state = np.array(state)
        self.num_qubits = int(np.round(np.log2(len(state)),0))
        assert len(self.state) == 2**self.num_qubits, "State size has to be of size 2**N for some integer N."
        self.circuit = models.Circuit(self.num_qubits)
        self.X = [gates.H,gates.I] # use Hadamard gate to switch to +/- basis
        S_dagger = lambda i: gates.U1(i,-np.pi/2)
        self.Y = [S_dagger,gates.H] # use Hadamard + phase gate to switch to +/-i basis
        self.Z = [gates.I,gates.I] # no need to change if already in comp. basis or not measuring at all
        self.I = self.Z
        return
        
    def sample(self,meas_basis=None,nshots=1):
        """ Draws <nshots> samples from the state.
            If no measurement basis is defined, samples are drawn in the computational basis.
            
            Inputs:
            - meas_basis as a Pauli string, i.e. a str of length num_qubits containing only X,Y,Z,I.
            - nshots (int) specifying how many samples to draw.
            
            Returns:
            - samples, numpy array of shape (nshots x  num_qubits).
        """
        c = self.circuit.copy(deep=True)
        if meas_basis is not None:
            assert len(meas_basis)==self.num_qubits, "Measurement basis has to be specified for each qubit."
            c.add([getattr(self,s)[0](i) for i,s in enumerate(meas_basis)])
            c.add([getattr(self,s)[1](i) for i,s in enumerate(meas_basis)])
        c.add(gates.M(*range(self.num_qubits)))
        out = c(initial_state=self.state.copy(),nshots=nshots)
        return -2*out.samples() + 1 # map {0,1} to {1,-1} outcomes
    
    def index_to_string(self,index_list):
        """ Helper function that maps a list of Pauli indices to a Pauli string, i.e.
            0 -> I, 1 -> X, 2 -> Y, 3 -> Z
            Returns the Pauli string.
        """
        pauli_string = ""
        for ind in np.array(index_list,dtype=int):
            assert ind in range(4), "Elements of index_list have to be in {0,1,2,3}."
            pauli_string += int_to_char[ind]
        return pauli_string

class Energy_estimator():
    """ Convenience class that holds both a measurement scheme and a StateSampler instance.
        The main workflow consists of proposing the next (few) measurement settings and measuring them given the state in StateSampler.
        Furthermore, it tracks all measurement settings and their respective outcomes (of value +/-1 per qubit).
        Based on these values, the current energy estimate can be calculated.
        
        Inputs:
        - measurement_scheme, see class Measurement_Scheme and subclasses for information.
        - state, see class StateSampler.
        - Energy offset (defaults to 0) for the energy estimation.
          This typically consists of the identity term in the corresponding Hamiltonian decomposition.
        - repeats (int), counting how often the same measure()-step has to be repeated. Can be used to gather statistics for the same measurement proposal
    """
    def __init__(self,measurement_scheme,state,offset=0,repeats=1):
        # TODO: shape checking for measurements in measurement_scheme and num_qubits in state
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state        = state
        self.settings     = []
        assert isinstance(repeats,int) and repeats >= 1, "Repeats should be an integer >= 1, but was {}.".format(repeats)
        self.repeats      = repeats
        self.outcomes     = {i: [] for i in range(self.repeats)}
        self.offset       = offset
        # convenience counters to keep track of measurements settings and respective outcomes
        self.num_settings = 0
        self.num_outcomes = 0
        
    def reset(self):
        self.settings = []
        self.outcomes = {i: [] for i in range(self.repeats)}
        # reset counters and measurement_scheme
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()
        
    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        settings = []
        for i in range(num_steps):
            p, _ = self.measurement_scheme.find_setting()
            settings.append(p)
        settings = np.array(settings)
        self.settings = np.vstack((self.settings,settings)) if len(self.settings)>0 else settings
        self.num_settings += num_steps
        return
    
    def measure(self):
        """ If there are proposed settings in self.settings that have not been measured, do so.
            The internal state of the StateSampler does not alter by doing so.
        """
        num_meas = self.num_settings - self.num_outcomes
        if num_meas > 0:
            # run all the last prepared measurement settings
            # from the settings list, fetch the unique settings and their respective counts
            unique_settings, counts = np.unique(self.settings[-num_meas:],axis=0,return_counts=True)
            
            outcomes = {i: [] for i in range(self.repeats)}
            ordered_settings = []
            for unique,nshots in zip(unique_settings,counts):
                samples = self.state.sample(meas_basis=self.state.index_to_string(unique),nshots=self.repeats*nshots)
                for i in range(self.repeats):
                    outcomes[i].append(samples[i*nshots:(i+1)*nshots])
                ordered_settings.append(np.repeat(unique[np.newaxis,:],nshots,axis=0))
            for i in range(self.repeats):
                outcomes[i] = np.vstack(outcomes[i])
            if len(self.outcomes[0]) == 0:
                self.outcomes = outcomes
            else:
                for i,outcome in outcomes.items():
                    self.outcomes[i] = np.vstack((self.outcomes[i],outcome))
            self.settings[-num_meas:] = np.vstack(ordered_settings)
            self.num_outcomes += num_meas
        else:
            print("No more measurements required at the moment. Please propose new setting(s) first.")
        return
    
    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energy. """
        if np.allclose(self.measurement_scheme.N_hits,0):
            # if no obsevables are hit yet, just return the offset value
            return self.offset
        w = self.measurement_scheme.w
        obs = self.measurement_scheme.obs
        energy = np.zeros((self.repeats,len(w)))
        for i,outcomes in self.outcomes.items():
            for p,outcome in zip(self.settings,outcomes):
                is_hit = [hit_by(o,p) for o in obs]
                N_hit  = np.sum(is_hit)
                outcome = np.repeat(outcome.reshape((1,-1)),N_hit,axis=0)
                outcome[obs[is_hit]==0] = 1 # set to 1 if outside the support the respective to mask it out, hit observables
                energy[i,:][is_hit] += w[is_hit] * np.prod(outcome,axis=1)
            # exclude those cases where N_hits == 0 to avoid dividing by 0. Due to the first if-clause, there is no empty sum here
        measured_at_least_once = self.measurement_scheme.N_hits != 0
        energy = np.sum( (energy.T)[measured_at_least_once] / self.measurement_scheme.N_hits[measured_at_least_once][:,np.newaxis], axis=0 )
        return energy + self.offset
    