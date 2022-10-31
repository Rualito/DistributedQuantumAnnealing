# Version 0.3.1 MPT
# Added metrics: ||H||, ||[HL, HN]||, ||[HL, [HL, HN]]||,||[HN, [HL, HN]]||
# 

from SimpleAdiabatic import *
# from AdiabaticTrotter import AdiabaticTrotter

from qutip import tensor
qutip.settings.num_cpus = 4

from qutip.qip.operations.gates import swap

# qutip.settings.num_cpus = 1


import numpy as np

def time_list_evaluate(time_func_list, time, args):
    sum_ = 0
    for Ht in time_func_list:
        sum_ += Ht[0] * Ht[1](time, args)
    return sum_ 

def overlap_probability_degen_gs(state, psi_gs_arr):
    temp = 0
    for psi_gs in psi_gs_arr:
        temp += np.abs(psi_gs.overlap(state))**2

    return temp

# Commute A and B
def cm(A, B):
    return A*B - B*A

def hnorm(norm, H):
    if norm=='spectral':
        return np.max(np.abs(H.eigenstates()[0]))
    
    return H.norm(norm)


class UltraTrotter: # evolve with Unitaries, UT
    # HL_spec: dict of local hamiltonians, to apply in separate
    #  
    #      format:
    #       {'HL1': ( ['a', 'b', ...names], HL, time_format), 'HL2':... } 
    #       qubit names are defined here 
    #       time_format can be 'linear', 'antilinear' or 'complex'
    #  
    #   linear: t/t_F
    #   antilinear: 1-t/t_F
    #   complex: HL already in time list - [[H1, lambda f1(t)],  [ H2, lambda f2(t)], ...] 
    #  
    # HN_spec: dict of non-local hamiltonians, applied separately (assumed commutation)
    #       same format, qubit names same as above
    # evolution time defined in time_format 
    #  
    def __init__(self, HL_spec, HN_spec, AT_args):
        self.qubits = [] # total qubits in system

        self.HL_spec = HL_spec
        self.HN_spec = HN_spec

        self.H0 = AT_args[0] + AT_args[2] 
        self.HF = AT_args[1] + AT_args[3] 
        self.HL = AT_args[0] + AT_args[1] 
        self.HN = AT_args[2] + AT_args[3]
    
        self.qubit_index = {}

        # indicates the ids of all the hamiltonians, which operate in separate later
        self.hamilt_ids = {'local': [], 'non-local':[]}

        mx_idx = 0 # matrix index; for fast matrix exponentiation
        self.hm = {} # storage of all the matrices, and their properties, for later access
        
        for hl_id in HL_spec:
            hamilt_qubits = HL_spec[hl_id][0]

            for i,qb_name in enumerate(hamilt_qubits):
                    self.qubit_index[qb_name] = i
                    self.qubits.append(qb_name)

            hamilt_matrix = HL_spec[hl_id][1]
            hamilt_timef = HL_spec[hl_id][2] # time format
            
            
            self.hamilt_ids['local'].append(mx_idx)
            exp_precalc = 1
            self.hm[mx_idx] = {'matrix':hamilt_matrix, 'qubits':hamilt_qubits, 'exp_precalc':exp_precalc, 'timef':hamilt_timef}
            mx_idx+=1
            
        for hn_id in HN_spec:
            hamilt_qubits = HN_spec[hn_id][0]
            hamilt_matrix = HN_spec[hn_id][1] 
            hamilt_timef = HN_spec[hn_id][2] # time format

            self.hamilt_ids['non-local'].append(mx_idx)
            exp_precalc = 1
            
            self.hm[mx_idx] = {'matrix':hamilt_matrix, 'qubits':hamilt_qubits, 'exp_precalc':exp_precalc, 'timef':hamilt_timef}
            mx_idx+=1
        self.total_mx = mx_idx

        # Default initial state
        self.psi_0 = self.H0.eigenstates()[1][0]
        self.qubits_SWAP = {}
        
        self.max_norm = 0

        for idx in self.hm:
            curr_qb = tuple(self.hm[idx]['qubits'])
            swaps = self.make_swap(curr_qb)
            self.qubits_SWAP[curr_qb] = [1, 1]
            for i, sw in enumerate(swaps):
                self.qubits_SWAP[curr_qb][i] = sw
        
    def make_swap(self, qubits):
        SWAP_list = []  
        
        system_identity = tensor([si]*len(self.qubits))
        
        qubits_temp = list(self.qubits)
        # qubit index list
        # to keep track of swaps
        qb_list = {qb:i for i,qb in enumerate(self.qubits)}

        for i, qb in enumerate(qubits): # qubits: qubit labels that interact via Um
            tg1 = qb_list[qb] # from wherever he is (target 1)
            tg2 = i # to wherever he needs to go (target 2)
            
            tg1_lbl = qb # label of target 1
            tg2_lbl = qubits_temp[i] # label of target 2
            
            # swapping labels
            temp = qb_list[tg1_lbl]
            qb_list[tg1_lbl] = qb_list[tg2_lbl]
            qb_list[tg2_lbl] = temp

            # swapping positions
            temp = qubits_temp[tg1]
            qubits_temp[tg1] = qubits_temp[tg2]
            qubits_temp[tg2] = temp

            if tg1 == tg2:
                SWAP_list.append(system_identity)
            else:
                SWAP_list.append(swap(N=len(self.qubits), 
                targets=[tg1, tg2]))
        
        rt_swap = 1
        for sw in SWAP_list:
            rt_swap = sw*rt_swap
        
        # returns the swap operation
        return rt_swap, rt_swap.dag()
        
    def build_system_unitary(self, Um, expand, swap1, swap2):
        # Um - unitary matrix to build into system 
        # qubits - to which qubits to apply the unitary 
        # First applies tensor product to build Um_12 in the larger qubit space
        # Then applies swaps to select the right qubits for interaction 
        # If U is applied in qubits i,j then Um=Um_12
        # U = SW_1i SW_2j Um_12 SW_j2 SW_i1 
        # if matrix sequence is True, then its returned as list of the matrices to multiply
        # (In case the matrix product should be made in another processor) 
        
        # indentity = si
        Um_temp = [self.si]*expand
        Um_temp[0] = Um

        # U12 x I^n
        Um_extended = tensor(Um_temp)
        
        # Using the SWAP matrices calculated at __init__
        unit = (swap2 * Um_extended) * swap1
        return unit
        
    def calc_matrix_iteration(self, Hm_id, stepk, m_trotter, evol_time):

        # mx_dim = len(self.hm[Hm_id]['qubits']) # dimension of operator
         # identity operator in this qubit space
        timef = self.hm[Hm_id]['timef']

        if timef ==  'linear' or timef == 'antilinear':
            # if stepk=0 -> exp{0} = 1
            # if stepk == 0:
            #     self.mx_hist[Hm_id] = identity * self.mx_hist[Hm_id] # initialization 
            if stepk>0: 
                # new iteration is (past iteration) * (matrix exp)
                self.mx_hist[Hm_id] = self.hm[Hm_id]['exp_precalc'] * self.mx_hist[Hm_id] 
            
        elif timef == 'complex':
            timek = stepk * evol_time/m_trotter

            self.mx_hist[Hm_id] = (-1.0j * time_list_evaluate(self.hm[Hm_id]['matrix'], timek/evol_time, {}) * evol_time/m_trotter).expm()
        else:
            raise ValueError(f"Time format must be 'linear', 'antilinear' or 'complex', got '{timef}'")
        return self.mx_hist[Hm_id]

    def evolve_discrete_unitaries(self, m_trotter, evol_time, stepf=-1, get_max_norm=False, hamiltonian_norm='fro'):
        # self.hamiltonians = 
        #           {'local': 
        #               {'linear':[
        #                   {'matrix':hamilt_matrix, 
        #                   'qubits':hamilt_qubits, 'idx':index}], 
        #               'antilinear':[], 
        #               'complex':[]}, 
        #           'non-local': {'linear':[], 'antilinear':[], 'complex':[]}}
        
        self.simulation_params = {'M':m_trotter, 'tf':evol_time}

        # matrix history, keeping track of the past iteration
        # keeps track of current (expm)**k
        self.mx_hist = [1]*self.total_mx 
        
        self.si = qutip.qeye(2)


        self.H_T = [ [self.H0, lambda t, arg: 1-t/evol_time], [self.HF, lambda t, arg: t/evol_time] ] 

        # the exponential precalc is done once, for each matrix
        for hm_id in self.hm:
            # exp_precalc = 1
            timef = self.hm[hm_id]['timef']
            hm = self.hm[hm_id]['matrix'] # matrix in which the hamiltonian is represented
            mx_dim = len(self.hm[hm_id]['qubits']) # dimension of operator
            identity = tensor([self.si]*mx_dim)

            if timef == 'linear':
                self.mx_hist[hm_id] = identity
                self.hm[hm_id]['exp_precalc'] = (-1.0j * hm * evol_time / (m_trotter)**2 ).expm()
            
            elif timef == 'antilinear':
                self.mx_hist[hm_id] = (-1.0j * hm * evol_time/m_trotter).expm() # matrix at k=0
                self.hm[hm_id]['exp_precalc'] = (1.0j*hm*evol_time/(m_trotter)**2 ).expm()
            
            elif timef == 'complex':
                # the exponential precalc is for 'complex' timef is done on the fly
                # so this quantity doesn't make much sense
                self.hm[hm_id]['exp_precalc'] = 1 
                self.mx_hist[hm_id] = 1
        
        # save the hamiltonian norms for possible use later
        self.norm_history = []
        unitary = 1
        step = 0
        ids = [*self.hamilt_ids['local'], *self.hamilt_ids['non-local']]
        for k in range(m_trotter):

            if get_max_norm:
                current_norm = time_list_evaluate(self.H_T, time=k * evol_time/m_trotter, args={}).norm(norm=hamiltonian_norm)
                self.norm_history.append(current_norm)
            
            for h_id in self.hamilt_ids['local']:
                uk = self.calc_matrix_iteration(h_id, k, m_trotter, evol_time)
                qubits = self.hm[h_id]['qubits']
                Uk = self.build_system_unitary(uk, len(self.qubits)-len(qubits)+1, self.qubits_SWAP[tuple(qubits)][0], self.qubits_SWAP[tuple(qubits)][1])

                if k == 0 and h_id == 0:
                    unitary = Uk
                else:
                    # t0 = time.process_time_ns()
                    unitary = Uk * unitary
                    # total_matmul+=time.process_time_ns()-t0
                step+=1 
                if stepf > 0 and step >= stepf:
                    print("Exiting, outputing unitary %d, qubits "%(h_id), self.hm[h_id]['qubits'])
                    return unitary, Uk, uk
            # unitary = 1
            for h_id in self.hamilt_ids['non-local']:
                uk = self.calc_matrix_iteration(h_id, k, m_trotter, evol_time)
                qubits = self.hm[h_id]['qubits']
                Uk = self.build_system_unitary(uk, len(self.qubits)-len(qubits)+1, self.qubits_SWAP[tuple(qubits)][0], self.qubits_SWAP[tuple(qubits)][1])

                # if k == 0 and h_id == 0:
                #     unitary = Uk
                # else:
                    # t0 = time.process_time_ns()
                unitary = Uk * unitary
                    # total_matmul+=time.process_time_ns()-t0
                step+=1 
                if stepf > 0 and step >= stepf:
                    print("Exiting, outputing unitary %d, qubits "%(h_id), self.hm[h_id]['qubits'])
                    return unitary, Uk, uk

        self.unitary_result = unitary
        return unitary

    def get_psif(self, psi0=None):
        if self.psi_0 is None and psi0 is None:
            raise ValueError("psi0 not defined")
        # setting psi0
        self.psi_0 = psi0 if not (psi0 is None) else self.psi_0
        psif = self.unitary_result * psi0
        self.psif = psif
        return psif
    
    def calc_metric(self, metric,hamiltonian_norm='fro' ):
        if metric == 'E0':
            return self.HF.eigenstates()[0][0]
        if metric == 'En':# energy of final state
            return np.real((self.psif.dag() * self.HF * self.psif).tr())
        if metric == 'OE':# Overlap error: 1 - < | >**2 = 1-fidelity 
            egs = self.HF.eigenstates()[0][0]
            psi_gs_arr = [self.HF.eigenstates()[1][i] for i, en in enumerate(self.HF.eigenstates()[0]) if en == egs]
            return 1 - overlap_probability_degen_gs(self.psif, psi_gs_arr)
        if metric == 'Emed':
            return np.real((self.psi_0.dag() * self.HF * self.psi_0).tr())
        if metric == '||HF||':
            return hnorm(hamiltonian_norm,self.HF)
        if metric == '||H0||':
            return hnorm(hamiltonian_norm,self.H0)
        if metric == '||[HL, HN]||':
            return hnorm(hamiltonian_norm, cm(self.HL, self.HN))
        if metric == '||HL||':
            return hnorm(hamiltonian_norm,self.HL)
        if metric == '||HN||':
            return hnorm(hamiltonian_norm,self.HN)
        if metric == '||[HL, [HL, HN]]||':
            return hnorm(hamiltonian_norm, cm(self.HL, cm(self.HL, self.HN)))
        if metric == '||[HN, [HL, HN]]||':
            return hnorm(hamiltonian_norm, cm(self.HN, cm(self.HL, self.HN)))
        
        # if metric == :
        #     return 
        return 'NA'
    
    def calc_errors(self, metrics, hamiltonian_norm='fro'):
        # def calc_method_errors(st_psif, HF, psi_gs_arr):
        
        metrics_result = {}

        if ('SE' in metrics) or ('AEn' in metrics): # if pure annealing simulation is required
            locals = {}
            for qb in self.qubits:
                locals[qb] = 0
            sa = SimpleAdiabatic(locals, {}, time_resolution=50, schedule_args={'t max':self.simulation_params['tf']})
            sa.H_0 = self.H0
            sa.H_1 = self.HF
            sa.reload_problem()
            sa.solve_main_eq()
        
        # metrics that require SA
        if 'SE' in metrics: # state error: || |SA> - |UT> ||
            sa_psif = sa.final_psi
            metrics_result['SE'] = (self.psif - sa_psif).norm()
        if 'AEn' in metrics: # simple adiabatic (pure annealing) energy
            metrics_result['AEn'] = sa.get_final_state_energy()

        # Other metrics
        for mtcs in metrics:
            if not (mtcs in {'SE', 'AEn'}): 
                metrics_result[mtcs] = self.calc_metric(mtcs, hamiltonian_norm=hamiltonian_norm)
        
        return metrics_result



