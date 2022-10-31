# MT Version 0.3.4.2
# With improved MBL metrics
# Now with proper H_reduced 
# Includes XX and YY couplings; fixed DT bug
# Trying to fix _reduced
from SimpleAdiabatic import *
from DistributedTrotter import DistributedTrotter
# from AdiabaticTrotter import AdiabaticTrotter

from qutip import tensor
from qutip.operators import qeye
from qutip.states import ket, ket2dm, bell_state
from qutip.qip.operations import controlled_gate, cnot, rz, snot, swap
from qutip.qobj import Qobj

qutip.settings.num_cpus = 4


from qiskit.quantum_info.synthesis.two_qubit_decompose \
    import two_qubit_cnot_decompose

# qutip.settings.num_cpus = 1

import numpy as np
import functools

from UltraTrotter import UltraTrotter
from DistributedTrotter import DistributedTrotter, generate_swaps
from utils import make_hamiltonian_from_spec,overlap_probability_degen_gs_rho

def time_list_evaluate(time_func_list, time, args):
    sum_ = 0
    for Ht in time_func_list:
        sum_ += Ht[0] * Ht[1](time, args)
    return sum_ 

def hnorm(norm, H):
    if norm=='spectral':
        return np.max(np.abs(H.eigenstates()[0]))
    
    return H.norm(norm)


class MultiPartyTrotter:
    # Requirements:
    #   Define shared variables
    #        Compute joint ground state appropriately
    #   Simulate with UT or DT depending on fidelity
    #   Return errors based on requested metrics 
    # Assume MPC from 'y' couplings
    #  
    def __init__(self, Hspec, method='UT'):

        self.sublocals = Hspec[0]
        self.system_spec = Hspec[1]
        Hams = make_hamiltonian_from_spec(self.sublocals, self.system_spec)
        self.Hams = Hams
        self.H0 = Hams['AT'][0] + Hams['AT'][2] 
        self.HF = Hams['AT'][1] + Hams['AT'][3] 
        self.HL = Hams['AT'][0] + Hams['AT'][1] 
        self.HN = Hams['AT'][2] + Hams['AT'][3]
        self.HL_spec = Hams['UT'][0]
        self.HN_spec = Hams['UT'][1]

        self.sa = SimpleAdiabatic({'':0}, {})
        self.sa.H_0 = self.H0
        self.sa.H_1 = self.HF
        self.sa.reload_problem()

        self.psif = None
        self.rhof = None

        self.method = method
        self.ut = UltraTrotter(self.HL_spec, self.HN_spec, Hams['AT'] )
        self.qubits = self.ut.qubits
        # self.qubits_index
        self.dt = DistributedTrotter(self.HL_spec, self.HN_spec, Hams['AT'])
            # self.qubits = self.dt.qubits
        if self.method == 'UT' :
            self.reduced_energies, self.reduced_eigenstates =\
                MultiPartyTrotter.get_joint_eigenstates(self.sublocals, self.system_spec)
        elif self.method == 'DT':
            self.reduced_energies, self.reduced_eigenstates =\
                MultiPartyTrotter.get_joint_eigenstates(self.sublocals, self.system_spec, state_type='rho')
            
    def get_joint_eigenstates(sublocals, system_spec, state_type='psi'):
        
        qubits_list = []
        qubits_index = {}
        idx = 0 
        
        for sub in sublocals:
            for qb in sub:
                qubits_list.append(qb)
                qubits_index[qb] = idx
                idx += 1
        
        Hams = make_hamiltonian_from_spec(sublocals, system_spec)
        HF = Hams['AT'][1] + Hams['AT'][3] 

        # get YY paired qubits
        paired_qubits = []
        for qbs, cpl, tm in system_spec['nonlocal']:
            # print(qbs, cpl, tm)
            if (cpl[0] in {'x','y'}) and (len(qbs)==2) and (tm==0) and qubits_index[qbs[0]]!= qubits_index[qbs[1]]:
                # if qubits coupled by yy are not on the same annealer
                paired_qubits.append(qbs)
        if len(paired_qubits) == 0:
            # print("Nothing to do, returning...")
            return HF.eigenstates()

        # HF_reduced = HF
        qubits_list_new = qubits_list.copy()
        qubit_list_history = []
        # pairing_operations = {}
        for q1, q2 in paired_qubits:
            # HF_reduced = MultiPartyTrotter.join_qubits(qubits_list_new, 
            #                 host_qb=q1, target_qb=q2, H=HF_reduced)
            # print(f"{(q1, q2)=}, {qubits_list_new}")
            qubit_list_history.append(qubits_list_new.copy())
            qubits_list_new.remove(q2)
        
        qubit_list_history.append(qubits_list_new.copy())
        # print(qubit_list_history)

        HF_reduced = MultiPartyTrotter.reduce_hamiltonian(sublocals, system_spec, HF)
        
        reduced_eigenvals, reduced_eigenstates = HF_reduced.eigenstates()
        # print(paired_qubits)
        new_reduced_eigenstates = []
        
        for k, state in enumerate(reduced_eigenstates):
            temp_state = state
            for i, (q1, q2) in list(enumerate(paired_qubits))[::-1]:
                # need to go through list in reverse
                
                qubit_list_init = qubit_list_history[i+1]
                qubit_list_target = qubit_list_history[i]
                # apply qubit duplication recursively to reduced state
                # try:
                temp_state = MultiPartyTrotter.duplicate_qubits_on_state(temp_state, qubit_list_init, qubit_list_target, q1, [q1, q2])
                    # print(f"pair {(q1,q2)} done")
                # except Exception as e:
                #     print(f"qubit_list_init={qubit_list_init}")
                #     print(f"qubit_list_target={qubit_list_target}")
                #     print(f"{i},{(q1,q2)}")
                #     raise e
            # print(f"State {k} is done")
            if state_type == 'rho':
                temp_state = ket2dm(temp_state)
            new_reduced_eigenstates.append(temp_state)
    
        # expand eigenstates back to original size

        return reduced_eigenvals, new_reduced_eigenstates

    def get_mpc_psi0(sublocals, system_spec):
        qubits_list = []
        qubits_index = {}
        idx = 0 
        
        for sub in sublocals:
            for qb in sub:
                qubits_list.append(qb)
                qubits_index[qb] = idx
                idx += 1
        paired_qubits = []
        paired_sign = []
        for qbs, cpl, tm in system_spec['nonlocal']:
            # print(qbs, cpl, tm)
            if (cpl[0] in {'x', 'y'}) and (len(qbs)==2) \
                and (tm==0) and qubits_index[qbs[0]]!= qubits_index[qbs[1]]:
                # if qubits coupled by yy are not on the same annealer
                paired_qubits.append(qbs)
                paired_sign.append({'x':0, 'y':1}[cpl[0]])
        
        ket_minus = 1/np.sqrt(2) * (ket('0') - ket('1'))
        PHI_plus = bell_state('00')
        PHI_minus = bell_state('01')

        # if YY coupling >0 -> ground state is PHI+
        # if YY coupling <0 -> ground state is PHI-
        yy_state = (PHI_minus, PHI_plus)

        qubit_ordering = []
        state_list = []
        for (q1, q2), s in zip(paired_qubits, paired_sign):
            qubit_ordering.append(q1)
            qubit_ordering.append(q2)
            # associating bell state to qubit pairs
            state_list.append(yy_state[s])
        
        for qb in qubits_list:
            if qb not in qubit_ordering:
                qubit_ordering.append(qb)
                state_list.append(ket_minus)
        # print(qubit_ordering, qubits_list)
        swap_list = generate_swaps(starting_order=qubit_ordering, target_order=qubits_list)
        
        # state is disordered, need to apply swaps to get right ordering
        state_return = tensor(state_list)
        for sws in swap_list:
            state_return = swap(N=len(qubits_list), targets=sws) *state_return 
        
        return state_return

    def join_qubits(qb_list:list, host_qb, target_qb, H):
        # From a given Hamiltonian, places all fields
        #  and couplings applied on
        # target_qb to apply in host_qb
        # and traces out target_qb 

        # Getting the indexes of host and target qubits 
        target_id = -1
        host_id = -1
        for i, qb in enumerate(qb_list):
            if qb == host_qb:
                host_id = i
            if qb == target_qb:
                target_id = i
        assert target_id>-1 and host_id>-1

        # Reorder operation on qubits  
        temp_list = qb_list.copy()
        temp_list.remove(target_qb)
        ptraced_list = [target_qb, *temp_list] 

        SWP_0_tg_list = generate_swaps(starting_order=ptraced_list.copy(), target_order=qb_list.copy())    

        # print("temp list: ", temp_list)
        # print("origin list: ", ptraced_list)
        # print("target list: ", qb_list)
        # print("operations: ", SWP_0_tg_list)

        SWP_0_tg = tensor([si]*len(qb_list)) # identity
        for sws in SWP_0_tg_list:
            SWP_0_tg = swap(N=len(qb_list), targets=sws) *SWP_0_tg 


        SWP_h_tg = swap(N=len(qb_list), targets=[host_id, target_id])

        # tracing out target qubit 
        ptrace_keep = list(range(len(qb_list)))
        ptrace_keep.remove(target_id)

        Q_target = H - 0.5*SWP_0_tg*(tensor(si, H.ptrace(ptrace_keep)))*SWP_0_tg.dag()
        # print("Q_target: ", Q_target.diag())
        # add couplings on target to host -> that's the reason for the swap 
        H_new = H + SWP_h_tg * Q_target * SWP_h_tg.dag()

        return H_new.ptrace(ptrace_keep)/2

    def duplicate_qubits_on_state(state, original_qubit_list:list, target_qubit_list:list, dupe_from, dupe_to:list, state_type='psi'):
        # ex: 
        # original_qubit_list = [a, b, c, d, e]
        # target_qubit_list = [a, b, c1, d, c2, e]
        # dupe_from = c 
        # dupe_to = [c1, c2] 

        # get indexes of origin duplication qubits
        dupe_from_idx = 0
        for i, qb in enumerate(original_qubit_list):
            if qb == dupe_from:
                dupe_from_idx = i
        
        swap_from = [*dupe_to, *[qb for qb in target_qubit_list if not (qb in dupe_to) ]]
        swap_to = target_qubit_list

        # swap to reorder qubits
        swaps_list = generate_swaps(starting_order=swap_from, 
        target_order=swap_to)

        SWP_q1q2 = tensor([si]*len(target_qubit_list))
        for sws in swaps_list:
            SWP_q1q2 = swap(N=len(target_qubit_list), targets=sws) * SWP_q1q2

        state_end = 0
        for j in ['0', '1']:
            ketq_list = [si]*(len(original_qubit_list))
            ketq_list[dupe_from_idx] = ket(j)
            ketq = tensor(ketq_list) # |j>_q

            ketq1q2 = ket(j*len(dupe_to)) # |jj>_q1q2

            if state_type == 'psi':
                s1 = tensor([ketq1q2, 
                        ketq.dag()*state])
                state_end += SWP_q1q2* s1
            
            elif state_type == 'rho':
                state_end += SWP_q1q2*tensor([ket2dm(ketq1q2), ketq.dag()*state*ketq])*SWP_q1q2.dag()
        return state_end

    def reduce_hamiltonian(sublocals, system_spec, H):
        qubits_list = []
        qubits_index = {}
        idx = 0 
        
        for sub in sublocals:
            for qb in sub:
                qubits_list.append(qb)
                qubits_index[qb] = idx
                idx += 1

        paired_qubits = []
        for qbs, cpl, tm in system_spec['nonlocal']:
            # print(qbs, cpl, tm)
            if (cpl[0] in {'x','y'}) and (len(qbs)==2) and (tm==0) and qubits_index[qbs[0]]!= qubits_index[qbs[1]]:
                # if qubits coupled by yy are not on the same annealer
                paired_qubits.append(qbs)
        if len(paired_qubits) == 0:
            # print("Nothing to do, returning...")
            return H
        
        H_reduced = H
        qubits_list_new = qubits_list.copy()
        qubit_list_history = []
        # pairing_operations = {}
        for q1, q2 in paired_qubits:
            H_reduced = MultiPartyTrotter.join_qubits(qubits_list_new, 
                            host_qb=q1, target_qb=q2, H=H_reduced)
            # print(f"{(q1, q2)=}, {qubits_list_new}")
            qubit_list_history.append(qubits_list_new.copy())
            qubits_list_new.remove(q2)
        
        qubit_list_history.append(qubits_list_new.copy())
        # print(qubit_list_history)
        
        return H_reduced        

    def propagate(self,tf, M, fidelity=1.0, psi0=None):
        self.psi_0 = self.H0.eigenstates()[1][0] if psi0 is None else psi0
        if fidelity<1.0:
            self.method = 'DT'
        if self.method == 'UT':
            self.ut.evolve_discrete_unitaries(M, tf)
            self.ut.psi_0 = self.psi_0
            self.ut.get_psif(self.psi_0)
        elif self.method=='DT':
            self.dt.psi_0 = self.psi_0
            self.dt.evolve_discrete_unitaries(M, tf, ket2dm(self.psi_0), 
                    fidelity=fidelity)
        elif self.method =='SA':
            self.sa.reload_problem()
            self.sa.psi_0 = self.psi_0
            self.sa.solve_main_eq()

    def calc_errors(self, metrics, hamiltonian_norm='fro'):
        metrics_result = {}
        if self.method == 'UT':
            metrics_result = self.ut.calc_errors(metrics,
                         hamiltonian_norm=hamiltonian_norm)
        elif self.method=='DT':
            metrics_result = self.dt.calc_errors(metrics,
                         hamiltonian_norm=hamiltonian_norm)
        
        if 'E0_reduced' in metrics:
            metrics_result['E0_reduced'] = self.reduced_energies[0]
            
            # raise ValueError(f"Not implemented: {metrics}")
        if 'En_reduced' in metrics:
            if self.method == 'UT':
                en_sum = 0
                psif = self.ut.psif
                for ei, si in zip(self.reduced_energies, self.reduced_eigenstates):
                    en_sum += ei * (si.dag()*psif).norm()**2
                metrics_result['En_reduced'] = en_sum
            elif self.method == 'DT':
                rhof = self.dt.get_rhof()
                en_sum = 0
                for ei, si in zip(self.reduced_energies, self.reduced_eigenstates):
                    en_sum += ei * (si.dag() * rhof * si).norm()
                metrics_result['En_reduced'] = en_sum
        if 'Emed_reduced' in metrics:
            emed_sum = 0
            for ei, si in zip(self.reduced_energies, self.reduced_eigenstates):
                emed_sum += ei * (si.dag()*self.psi_0).norm()**2
            metrics_result['Emed_reduced'] = emed_sum
        if 'OE_reduced' in metrics:
            psi_gs_arr = []
            egs = np.min(self.reduced_energies)
            for es, en in zip(self.reduced_eigenstates, self.reduced_energies):
                if en == egs:
                    psi_gs_arr.append(es)
            metrics_result['OE_reduced'] = 1 - overlap_probability_degen_gs_rho(self.dt.get_rhof(), psi_gs_arr)
        if '||HF_reduced||' in metrics:
            metrics_result['||HF_reduced||'] = hnorm(hamiltonian_norm, MultiPartyTrotter.reduce_hamiltonian(self.sublocals, self.system_spec, self.HF))
        return metrics_result