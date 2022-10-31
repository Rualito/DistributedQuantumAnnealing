# Version: 0.3 - updated for MPC

import itertools
# %matplotlib inline
import matplotlib.pyplot as plt

from qutip import *
import numpy as np


# precomputing some operators
si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

def spin_tensors(N):
    # these are the $\sigma_d^i$, Pauli spin operators which only act on spin i
    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si) # making an operator list with the identity up to N spins

        op_list[n] = sx # at the position n replace si by sx
        sx_list.append(tensor(op_list))
        
        op_list[n] = sy # at the position n replace sx by sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz # at the position n replace sy by sz
        sz_list.append(tensor(op_list))
    return sx_list, sy_list, sz_list

ket_plus = 1/np.sqrt(2)* (basis(2,0)+basis(2,1))
ket_minus = 1/np.sqrt(2)* (basis(2,0)-basis(2,1))

def schedule_linear(t, args):
    return t/args['t max']

def schedule_pause_quench(t, args):
    start_time = args['start time']
    duration = args['duration']
    at_pause = args['schedule at pause']
    t_max = args['t max']
    if start_time+duration > t_max:
        raise ValueError("start_time too late / duration too long: start_time+duration > t_max")

    return schedule_piecewise_linear(t, [[0,0], [start_time,at_pause], [start_time+duration, at_pause], [t_max,1.0]])

def schedule_piecewise_linear(t, schedule_points):

    # schedule_points: 2 * n array - [ [2], [2], [2], ... n]
    # each pair in the array has the format [time, schedule]
    #  
    for i in range(len(schedule_points)-1):
        point_now = schedule_points[i]
        point_next = schedule_points[i+1]
        if t >= point_now[0] and t < point_next[0]:
            # y = (y1-y0)/(x1-x0) * (x-x0) + y0 - linear interpolation
            result = (point_next[1]-point_now[1])/(point_next[0]-point_now[0]) * (t-point_now[0]) + point_now[1]

            if result > 1 or result < 0:
                raise ValueError(f"Schedule out of bounds: {result}")

            return result
    # print(f"Warning: outside schedule_points range (time={t}) - assuming constant")
    return (schedule_points[0][0] if t < schedule_points[0][1] else schedule_points[-1][1])


# Commute A and B
def cm(A, B):
    return A*B - B*A

def hnorm(norm, H):
    if norm=='spectral':
        return np.max(np.abs(H.eigenstates()[0]))
    
    return H.norm(norm)

def overlap_probability_degen_gs(state, psi_gs_arr):
    temp = 0
    for psi_gs in psi_gs_arr:
        temp += np.abs(psi_gs.overlap(state))**2

    return temp


default_sch_args = {'t max': 100}

# schedule_params = {'start time': t_max*0.8, 
#                    'duration': t_max*0.1, 
#                    'schedule at pause':0.4, 
#                    't max':t_max}

class SimpleAdiabatic:
    # initializes the Hamiltonian, displays some useful graphs and statistics

    def __init__(self, local_fields:dict, couplings:dict, schedule_func=schedule_linear, schedule_args=default_sch_args, time_resolution=200, total_eigenv=None, as_qubo=False, ignore_qubits=[], ignore_weight=0, non_local={}):
        # local_fields format: {'a': h_a, 'b':h_b, ...} - { [name] : [local field strength]}
        # couplings format {('a','b'):J_ab, ...}
        # tuple ('a', 'b') is immutable (for it to be hashable for the dict)
        # ignore_qubits - don't apply sigma^x on some qubits: ['a', 'b', ...]
        # non_local: example: {('qb1', 'qb2'):{'operation':sigz(s), 'freq':10}, ('c', 'd):...}
        # the operation is a function that returns the 4x4 matrix U for each schedule s=t/t_f

        self.sx_i, self.sy_i, self.sz_i = spin_tensors(len(local_fields))
        self.couplings = couplings
        self.local_fields = local_fields
        self.sch_args = schedule_args
        self.schedule_func = schedule_func
        self.non_local = non_local
        self.time_list = np.linspace(0, schedule_args['t max'], time_resolution)

        self.total_eigen = total_eigenv if total_eigenv else 2**len(self.local_fields)
        self.H_0 = 0 
        for i,qb in enumerate(local_fields.keys()):
            if qb not in ignore_qubits:
                self.H_0 += self.sx_i[i]
            elif ignore_weight:
                self.H_0 += ignore_weight*self.sx_i[i]

        # psi_0 is in the ground state of H_0
        self.psi_0 = self.H_0.eigenstates()[1][0]

        self.H_1 = 0

        self.name_index = {}
        # if as_qubo:
        #     print(f" This is represented as a qubo problem ")
        for index, name in enumerate(local_fields):
            h_i = local_fields[name] 
            self.name_index[name] = index # associate qubit names to an index
            # print(f"Adding local field to {index}: {h_i}")
            #    h_i * \sigma_z^i
            self.H_1 += h_i*((1-self.sz_i[index])/2 if as_qubo else self.sz_i[index])
        
        for (qub1, qub2), strength in couplings.items():
            index_1 = self.name_index[qub1] # index associated with first qubit
            index_2 = self.name_index[qub2] # index associated with second qubit
            # index_1 may be = index_2 - allows for a more generic way to define the problem
            # print(f"Adding coupling {(index_1, index_2)} of {strength}")
            self.H_1 += strength * ((1-self.sz_i[index_1])/2 if as_qubo else self.sz_i[index_1]) * ((1-self.sz_i[index_2])/2 if as_qubo else self.sz_i[index_2])

        # describing the total hamiltonian as a list of time functions
        self.H_T = [ [self.H_0, lambda t, arg: 1-schedule_func(t, schedule_args)], 
                    [self.H_1, lambda t, arg: schedule_func(t, schedule_args)]]
    
    def reload_problem(self):
        # in case some variable has been changed this needs to update
        # this will break if number of qubits changes
        self.H_T = [ [self.H_0, lambda t, arg: 1-self.schedule_func(t, self.sch_args)], 
            [self.H_1, lambda t, arg: self.schedule_func(t, self.sch_args)]]

        self.psi_0 = self.H_0.eigenstates()[1][0]

    def __process_rho__(self, t, psi):
        # input: 
        #   t - current time in the integration
        #   psi - current solution to the Hamiltonian (Lindbladian in general)
        
        # this function saves the results at each specified time in taulist
        # global idx, min_gap, final_psi
        # evaluate the Hamiltonian with gradually switched on interaction 
        # essentially calculating the hamiltonian at time t: H(t) = f1(t) H_1 + f2(t) H_2
        H = qobj_list_evaluate(self.H_T, t, {}) 
        self.final_psi = psi
        # find the M=solver_args['eigenvals']
        #  lowest eigenvalues of the system
        evals, ekets = H.eigenstates(eigvals=self.total_eigen)
        
        self.evals_mat[self.__idx,:] = np.real(evals)
        
        # find the overlap between the eigenstates and psi 
        for n, eket in enumerate(ekets):
            self.P_mat[self.__idx,n] = abs((eket.dag().data * psi.data)[0,0])**2    
        
        if self.__idx == 0:
            for i in range(self.total_eigen):
                for j in range(i+1, self.total_eigen):
                    # evals[j] is always >= evals[i] 
                    self.min_gap[i,j] = evals[j] - evals[i]
            # self.min_gap = evals[1]-evals[0]
        else:
            for i in range(self.total_eigen):
                for j in range(i+1, self.total_eigen):
                    self.min_gap[i, j] = evals[j] - evals[i] if evals[j] - evals[i] < self.min_gap[i, j] else self.min_gap[i, j]

        # current 'process id' - time index in taulist
        self.__idx += 1

    def solve_main_eq(self):
        self.__idx = 0
        self.min_gap = np.zeros( (self.total_eigen,)*2)-1

        self.evals_mat = np.zeros( (len(self.time_list), self.total_eigen) )
        self.P_mat = np.zeros( (len(self.time_list), self.total_eigen) )

        mesolve(self.H_T, self.psi_0, self.time_list, [], self.__process_rho__)

        # self.evals_mat[-1,:]
    
    def print_combinatorial_H1(self):
        len_qbits = len(self.local_fields)
        print( ("{} "*len_qbits + " |\tE\tP ").format(*self.local_fields.keys()))
        # evals, _ = H_1.eigenstates()
        # iterate over all combinations
        for i, ev_c in enumerate(itertools.product([0,1], repeat=len_qbits)):
            psi = tensor([basis(2,i) for i in ev_c])
            
            print( ("{} "*len_qbits).format(*ev_c) + f" |\t{(psi.dag()*self.H_1*psi).tr()+0.0:.3} \t{(psi.dag()*self.final_psi).norm()**2+0.0:.3}")

    def print_combinatorial_H1_highest_P(self, num=4, latex=False):
        len_qbits = len(self.local_fields)
        print( ("{} "*len_qbits + " |\tE\tP ").format(*self.local_fields.keys()))
        # evals, _ = H_1.eigenstates()
        # iterate over all combinations
        # data = {}
        comb_list = []
        data_arr = np.zeros((2**len_qbits, 3))
        for i, ev_c in enumerate(itertools.product([0,1], repeat=len_qbits)):
            psi = tensor([basis(2,i) for i in ev_c])
            if latex:
                state_comb = ("{}"*len_qbits).format(*ev_c)
            else: 
                state_comb = ("{} "*len_qbits).format(*ev_c)
            energy = (psi.dag()*self.H_1*psi).tr()+0.0
            prob = (psi.dag()*self.final_psi).norm()**2+0.0
            data_arr[i, 0] = i
            data_arr[i, 1] = energy
            data_arr[i, 2] = prob
            comb_list.append(state_comb)

            # data[state_comb] = {'E':energy, 'P':prob}
            # print( state_comb + f" |\t{energy:.3} \t{prob:.3}")
        data_sorted = data_arr[ data_arr[:, 2].argsort()[::-1]]
        
        for i in range(min([num,2**len_qbits])):
            index = int(data_sorted[i,0])
            energy = data_sorted[i,1]
            prob = data_sorted[i,2]

            state_comb = comb_list[index]
            if latex:
                print( f"\\ket{{{state_comb}}}" + f"&{energy:.3} & {prob:.3} \\\\ ")
            else: 
                print( state_comb + f" |\t{energy:.3} \t{prob:.3}")

    def get_final_state_energy(self):
        return (self.final_psi.dag() * self.H_1 * self.final_psi).tr()+0.0  
    
    def show_eigenvalue_plot(self, ax):

        # first draw thin lines outlining the energy spectrum
        for n in range(len(self.evals_mat[0,:])):
            ls,lw = ('b',1) if n == 0 else ('k', 0.25)
            ax.plot(self.time_list/max(self.time_list), self.evals_mat[:,n], ls, lw=lw)

        # second, draw line that encode the occupation probability of each state in 
        # its linewidth. thicker line => high occupation probability.
        for idx in range(len(self.time_list)-1):
            for n in range(len(self.P_mat[0,:])):
                lw = 0.5 + 4*self.P_mat[idx,n]    
                if lw > 0.55:
                    ax.plot(np.array([self.time_list[idx], self.time_list[idx+1]])/self.sch_args['t max'], 
                                np.array([self.evals_mat[idx,n], self.evals_mat[idx+1,n]]), 
                                'r', linewidth=lw)    
                
        ax.set_xlabel(r'$t$', fontsize=25)
        ax.set_ylabel(r'$\varepsilon_i$', fontsize=25)
        ax.set_title("Energyspectrum of a chain of %d spins.\n " % (len(self.local_fields))
                        + "The occupation probabilities are encoded in the red line widths.")
        
        # plt.show()

    def show_occupation_plot(self):
        #
        # plot the occupation probabilities for the few lowest eigenstates
        #

        for n in range(len(self.P_mat[0,:])):
            if n == 0:
                plt.plot(self.time_list/max(self.time_list), 0 + self.P_mat[:,n], 'r', linewidth=2)
            else:
                plt.plot(self.time_list/max(self.time_list), 0 + self.P_mat[:,n])

        plt.xlabel(r'$s(t)$')
        plt.ylabel('Occupation probability')
        plt.title("Occupation probability of the %d lowest " % self.total_eigen +
                        "eigenstates for a chain of %d spins" % len(self.local_fields))
        plt.legend(("Ground state",))

        plt.show()
    
    def show_EvsP(self):
        e_list = self.evals_mat[-1]
        p_list = self.P_mat[-1]
        logp_list = np.log(p_list)

        plt.scatter(e_list, logp_list, color='red')
        plt.xlabel("Energy of the state")
        plt.ylabel("Log Probability of final state")
        # plt.yscale('log')

        linear_model = np.polyfit(e_list, logp_list, 1)
        linear_model_fn = np.poly1d(linear_model)
        e_range = np.linspace(min(e_list), max(e_list), 2)
        plt.plot(e_range, linear_model_fn(e_range), color='blue')
        print(f"log[P](E) = {linear_model[0]:.4} + {linear_model[1]:.4}*E")
        plt.show()
    
    def get_log_relation_EvP(self):
        # returns the linear fit of the log[P] vs E graph
        e_list = self.evals_mat[-1]
        p_list = self.P_mat[-1]
        logp_list = np.log(p_list)

        return np.polyfit(e_list, logp_list, 1)
    
    def get_min_gap(self, e_tol=1.0e-4):
        # e_tol: minimum energy to consider above ground state
        for i in range(1,self.total_eigen):
            if self.min_gap[0,i] > e_tol:
                return self.min_gap[0,i]
        
    def get_state_error(self):
        H_F = self.H_T[1][0] 
        psi_gs = H_F.eigenstates()[1][0]
        psi_temp = psi_gs - self.final_psi
        return psi_temp.norm()

    def get_energy_error(self):
        H_F = self.H_T[1][0] 
        egs = H_F.eigenstates()
        # psi_gs = egs[1][0]
        # psi_temp = psi_gs - self.final_psi
        ea = (self.final_psi.dag() * H_F * self.final_psi).tr()+0.0 - egs[0][0] # adiabatic energy error
        return ea

    def calc_metric(self, metric,hamiltonian_norm='fro' ):
        if metric == 'E0':
            return self.H_1.eigenstates()[0][0]
        if metric == 'En':# energy of final state
            return np.real((self.final_psi.dag() * self.H_1 * self.final_psi).tr())
        if metric == 'OE':# Overlap error: 1 - < | >**2 = 1-fidelity 
            egs = self.HF.eigenstates()[0][0]
            psi_gs_arr = [self.H_1.eigenstates()[1][i] for i, en in enumerate(self.H_1.eigenstates()[0]) if en == egs]
            return 1 - overlap_probability_degen_gs(self.final_psi, psi_gs_arr)
        if metric == 'Emed':
            return np.real((self.psi_0.dag() * self.H_1 * self.psi_0).tr())
        if metric == '||HF||':
            return hnorm(hamiltonian_norm,self.H_1)
        if metric == '||H0||':
            return hnorm(hamiltonian_norm,self.H_0)
        # if metric == '||[HL, HN]||':
        #     return hnorm(hamiltonian_norm, cm(self.HL, self.HN))
        # if metric == '||HL||':
        #     return hnorm(hamiltonian_norm,self.HL)
        # if metric == '||HN||':
        #     return hnorm(hamiltonian_norm,self.HN)
        # if metric == '||[HL, [HL, HN]]||':
        #     return hnorm(hamiltonian_norm, cm(self.HL, cm(self.HL, self.HN)))
        # if metric == '||[HN, [HL, HN]]||':
        #     return hnorm(hamiltonian_norm, cm(self.HN, cm(self.HL, self.HN)))
        
        # if metric == :
        #     return 
        return 'NA'
    
    def calc_errors(self, metrics, hamiltonian_norm='fro'):
        # def calc_method_errors(st_psif, HF, psi_gs_arr):
        
        metrics_result = {}

        # if ('SE' in metrics) or ('AEn' in metrics): # if pure annealing simulation is required
        #     locals = {}
        #     for qb in self.qubits:
        #         locals[qb] = 0
        #     sa = SimpleAdiabatic(locals, {}, time_resolution=50, schedule_args={'t max':self.simulation_params['tf']})
        #     sa.H_0 = self.H0
        #     sa.H_1 = self.HF
        #     sa.reload_problem()
        #     sa.solve_main_eq()
        
        # # metrics that require SA
        # if 'SE' in metrics: # state error: || |SA> - |UT> ||
        #     sa_psif = sa.final_psi
        #     metrics_result['SE'] = (self.psif - sa_psif).norm()
        # if 'AEn' in metrics: # simple adiabatic (pure annealing) energy
        #     metrics_result['AEn'] = sa.get_final_state_energy()

        # Other metrics
        for mtcs in metrics:
            metrics_result[mtcs] = self.calc_metric(mtcs, hamiltonian_norm=hamiltonian_norm)
        
        return metrics_result



