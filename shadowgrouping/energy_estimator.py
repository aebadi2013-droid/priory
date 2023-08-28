import numpy as np
from qibo import models, gates
from shadowgrouping.measurement_schemes import hit_by
from shadowgrouping.hamiltonian import int_to_char

##################### Helper functions ########################################################
def save_energy_estimations(estimations,groundstate_energy,savefile=""):
    savename = savefile + "_energy_estimations.txt"
    np.savetxt(savename,estimations,header="Averaged over {} runs\nE_GS: {}".format(len(estimations),groundstate_energy))
    return

def load_energy_estimations(savefile=""):
    estimations = np.loadtxt(savefile + "_energy_estimations.txt",skiprows=2)
    with open(savefile + "_energy_estimations.txt","r") as f:
        print(f.readline())
        s = f.readline()
    energy = float(s.strip().split()[-1])
    return estimations, energy
###############################################################################################


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
        out = out.samples()
        # mask-out non-measured qubits in <meas-basis>
        out = out * np.array([s!="I" for s in meas_basis],dtype=int)[np.newaxis,:]
        return -2*out + 1 # map {0,1} to {1,-1} outcomes
    
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
    
class Sign_estimator():
    
    def __init__(self,measurement_scheme,state,offset):
        assert measurement_scheme.num_qubits == state.num_qubits, "Measurement and state scheme do not match in terms of qubit number."
        self.measurement_scheme = measurement_scheme
        self.state        = state
        self.offset       = offset
        self.setting_inds = []
        self.outcomes     = []
        self.num_settings = 0
        self.num_outcomes = 0
    
    def reset(self):
        self.setting_inds = []
        self.outcomes     = []
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()
        
    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        inds = self.measurement_scheme.find_setting(num_steps)
        self.setting_inds = np.append((self.setting_inds,inds)) if len(self.setting_inds)>0 else inds
        self.num_settings += num_steps
        return
    
    def measure(self):
        """ If there are proposed settings in self.settings that have not been measured, do so.
            The internal state of the VQE does not alter by doing so.
        """
        num_meas = self.num_settings - self.num_outcomes
        if num_meas > 0:
            # run all the last prepared measurement settings
            # from the settings list, fetch the unique settings and their respective counts
            recent_settings = self.setting_inds[-num_meas:]            
            outcomes = np.zeros(num_meas,dtype=int)
            for unique,nshots in zip(*np.unique(recent_settings,return_counts=True)):
                setting = self.measurement_scheme.obs[unique]
                samples = self.state.sample(meas_basis=self.state.index_to_string(setting),nshots=nshots)
                outcomes[recent_settings==unique] = np.prod(samples,axis=-1)
            self.outcomes = np.append(self.outcomes,outcomes)
            self.num_outcomes += num_meas
        else:
            print("No more measurements required at the moment. Please propose new setting(s) first.")
        return
    
    def get_energy(self):
        """ Takes the current outcomes and estimates the corresponding energy. """
        if self.num_outcomes == 0:
            # if no measurements have been done yet, just return the offset value
            return self.offset
        w = self.measurement_scheme.w
        sgn = np.sign(w)
        norm = np.sum(np.abs(w))
        energy = np.mean(self.outcomes*sgn[self.setting_inds])*norm
        return energy + self.offset

class Energy_estimator():
    """ Convenience class that holds both a measurement scheme and a VQA instance.
        The main workflow consists of proposing the next (few) measurement settings and measuring them in the VQA.
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
        self.is_adaptive  = measurement_scheme.is_adaptive
        if self.is_adaptive:
            self.update_steps = np.array(measurement_scheme.update_steps)
            assert hasattr(measurement_scheme,"receive_outcome"), "The given method does not have the function receive_outcome()."
        
    def reset(self):
        self.settings = []
        self.outcomes = {i: [] for i in range(self.repeats)}
        # reset counters and measurement_scheme
        self.num_settings, self.num_outcomes = 0, 0
        self.measurement_scheme.reset()
        
    def propose_next_settings(self,num_steps=1):
        """ Find the <num_steps> next setting(s) via the provided measurement scheme. """
        if self.is_adaptive:
            # check that num_steps does not exceed the next threshold for updating the measurement_scheme internally
            # if so, limit num_steps accordingly
            thresholds = self.update_steps - self.num_settings
            thresholds = thresholds[thresholds>0] # throw away all reached thresholds but the last one (threshold is ordered)
            if len(thresholds)>0:
                max_steps_allowed = thresholds[0]
                if num_steps > max_steps_allowed:
                    print("Warning! Trying to allocate more settings than allowed before updating the measurement scheme with outcomes.")
                    print("Num_steps = {0} reduced to {1}. Allocating num_steps={1} instead.".format(num_steps,max_steps_allowed))
                    num_steps = max_steps_allowed
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
            The internal state of the VQE does not alter by doing so.
        """
        num_meas = self.num_settings - self.num_outcomes
        if num_meas > 0:
            # run all the last prepared measurement settings
            # from the settings list, fetch the unique settings and their respective counts
            # the other arrays are only needed in case of self.is_adaptive
            unique_settings, inverse, counts = np.unique(self.settings[-num_meas:],axis=0,return_inverse=True,return_counts=True)
            
            outcomes = {i: [] for i in range(self.repeats)}
            ordered_settings = []
            
            for unique,nshots in zip(unique_settings,counts):                
                samples = self.state.sample(meas_basis=self.state.index_to_string(unique),nshots=self.repeats*nshots)
                for i in range(self.repeats):
                    temp = samples[i*nshots:(i+1)*nshots]
                    outcomes[i] = temp if len(outcomes[i])==0 else np.vstack((outcomes[i],temp))
                temp = np.repeat(unique[np.newaxis,:],nshots,axis=0)
                ordered_settings = temp if len(ordered_settings)==0 else np.vstack((ordered_settings,temp))
            if self.is_adaptive:
                # reorder the outcomes to the initially allocated settings
                ordered_inds = np.repeat(range(len(counts)),repeats=counts) # current order of outcomes[i] as sampled
                for i in range(self.repeats):
                    reordered_outcomes = np.full(outcomes[i].shape,2,dtype=int)
                    for ind in range(len(counts)):
                        assert np.min(reordered_outcomes[inverse==ind]) == 2, "Error in reshuffling outcomes: trying to overwrite filled rows"
                        reordered_outcomes[inverse==ind] = outcomes[i][ordered_inds==ind]
                    assert np.max(reordered_outcomes) != 2, "Error in reshuffling outcomes: not all rows have been filled."
                    outcomes[i] = reordered_outcomes
            if self.is_adaptive or hasattr(self.measurement_scheme,"receive_outcome"):
                for outcome in outcomes[0]:
                    # breaking ties here as the method has to build upon a fixed instantiation of outcomes
                    self.measurement_scheme.receive_outcome(outcome)
                # no reordering for self.settings is needed as nothing has been overwritten
            else:
                # overwrite the reshuffled settings
                self.settings[-num_meas:] = ordered_settings
                
            if len(self.outcomes[0]) == 0:
                self.outcomes = outcomes
            else:
                for i,outcome in outcomes.items():
                    self.outcomes[i] = np.vstack((self.outcomes[i],outcome))
                
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
        N_hits = np.zeros(len(w),dtype=int) # rebuild this for consistency reasons
        for i,outcomes in self.outcomes.items():
            for p,outcome in zip(self.settings,outcomes):
                is_hit = [hit_by(o,p) for o in obs]
                N_hits += is_hit
                N_hit  = np.sum(is_hit)
                outcome = np.repeat(outcome.reshape((1,-1)),N_hit,axis=0)
                outcome[obs[is_hit]==0] = 1 # set to 1 if outside the support the respective to mask it out, hit observables
                energy[i,:][is_hit] += w[is_hit] * np.prod(outcome,axis=1)
        if not self.is_adaptive:
            N_hits = self.measurement_scheme.N_hits
        # exclude those cases where N_hits == 0 to avoid dividing by 0. Due to the first if-clause, there is no empty sum here
        measured_at_least_once = N_hits != 0
        energy = np.sum( (energy.T)[measured_at_least_once] / N_hits[measured_at_least_once][:,np.newaxis], axis=0 )
        return energy + self.offset
    
    
