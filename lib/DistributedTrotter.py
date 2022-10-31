# Version 0.3.1 MPC
# Added metrics: ||H||, ||[HL, HN]||, ||[HL, [HL, HN]]||,||[HN, [HL, HN]]||
# Made by Raúl Santos

from SimpleAdiabatic import *
# from AdiabaticTrotter import AdiabaticTrotter

from qutip import tensor
from qutip.operators import qeye
from qutip.states import ket, ket2dm, bell_state
from qutip.qip.operations import controlled_gate, cnot, rz, snot, swap
from qutip.qobj import Qobj

qutip.settings.num_cpus = 4


from qiskit.quantum_info.synthesis.two_qubit_decompose \
    import two_qubit_cnot_decompose


from utils import overlap_probability_degen_gs_rho
# qutip.settings.num_cpus = 1

import numpy as np
import functools

# Commute A and B
def cm(A, B):
    return A*B - B*A

def hnorm(norm, H):
    if norm=='spectral':
        return np.max(np.abs(H.eigenstates()[0]))
    
    return H.norm(norm)



def time_list_evaluate(time_func_list, time, args):
    sum_ = 0
    for Ht in time_func_list:
        sum_ += Ht[0] * Ht[1](time, args)
    return sum_ 


def generate_swaps(starting_order, target_order):
    w_order = starting_order
    st_lbl = {lb:i for i, lb in enumerate(starting_order)}
    w_lbl = st_lbl
    tg_lbl = {lb:i for i, lb in enumerate(target_order)}

    swaps = []
    for i, lb in enumerate(target_order):
        # go through all positions
        # at position i, switch to where the target label is
        if i == w_lbl[lb]: # if no swap needed, skip
            continue
        
        # registering the swap
        # putting the target label into target position
        swaps.append( (i, w_lbl[lb]) )
        
        l0 = w_order[i] # starting label at position i 
        p0 = i # position i

        l1 = lb # target label
        p1 = w_lbl[lb] # starting position of target label

        w_lbl[l0] = p1 # starting label goes to position of target label
        w_lbl[l1] = p0 # target label goes to position i (of the starting label)
        w_order[p0] = l1 # label at starting position is target label
        w_order [p1] = l0 # label where target label was is now starting label
        
    return swaps


@functools.lru_cache(maxsize=None)
def  generate_swap_ops_ordered(target:tuple, Nqb):
    # swap qubits 0, 1 to target[0], target[1]
    # if (tuple(target), Nqb) in generate_swap_ops.history:
    #     return generate_swap_ops.history[(tuple(target), Nqb)]
    
    target_order = []
    idx = 0
    for i in range(Nqb):
        if i == target[0]:
            target_order.append('A')
        elif i == target[1]:
            target_order.append('B')
        else:
            target_order.append(idx)
            idx+=1
    swap_list = generate_swaps(['A', 'B', *range(Nqb-2)], target_order)

    Iden_Nqb = tensor([si]*Nqb)
    sw = Iden_Nqb
    for sws in swap_list:
        sw = swap(N=Nqb, targets=sws) * sw
    
    print("Gen swaps: target: ", target)
    print("Target order: ", target_order)
    print("Swap list: ", swap_list)
    raise ValueError("Test")
    # generate_swap_ops.history[(tuple(target), Nqb)] = sw
    return sw


def u3_matrix(angles):
    th, phi, lm = angles
    m00 = np.cos(th/2)
    m01 = -np.exp(1.0j * lm) * np.sin(th/2)
    m10 = np.exp(1.0j * phi) * np.sin(th/2)
    m11 = np.exp(1.0j * (phi + lm)) * np.cos(th/2)

    return np.array([[m00, m01], [m10, m11]])

@functools.lru_cache(maxsize=None)
def build_DCnot(fidelity):

    x = 4.0/3 * (fidelity - 1.0/4) 
    rho_dist2 = x * ket2dm(bell_state('00')) + (1-x)/4 * tensor([qeye(2)]*2)
    
    dcnot_mx = np.zeros((4,4, 4,4), dtype=np.complex128)
    for i, stA in enumerate(itertools.product('01', repeat=2)):
        ketA = ket(stA)
        for j,stB in enumerate(itertools.product('01', repeat=2)):
            ketB = ket(stB)

            dcnot_mx[i][j] = (build_DCnot.Ucnot4 * \
                                tensor([ketA*ketB.dag(), rho_dist2]) *\
                                build_DCnot.Ucnot4.dag()).ptrace([0,1])
    # build_DCnot.history[fidelity] = dcnot_mx
    return dcnot_mx

# A, B, 1, 2
build_DCnot.Ucnot4 = controlled_gate(1.0j*rz(np.pi), 4,control=3, target=0) \
    * snot(4, 3) * cnot(4, 3,1)* cnot(4, 2,3) * cnot(4,0,2)

@functools.lru_cache(maxsize=None)
def cache_make_swap(qubit_tuple:tuple, target_qubits:tuple, debug=False):
    # put labels 0,1 onto positions target_qubits[0,1]
    
    SWAP_list = []  
    
    system_identity = tensor([si]*len(qubit_tuple))
    
    qubits_temp = list(qubit_tuple)
    # qubit index list
    # to keep track of swaps
    qb_lbl = {qb:i for i,qb in enumerate(qubit_tuple)}

    qb_list = qb_lbl

    swap_ids = []

    for i, qb in enumerate(target_qubits): # target_qubits: qubit labels that interact via Um
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
            SWAP_list.append(swap(N=len(qubit_tuple), 
            targets=[tg1, tg2]))
            swap_ids.append((tg1, tg2))
    
    rt_swap = 1
    for sw in SWAP_list:
        rt_swap = rt_swap*sw
    if debug:
        return rt_swap, rt_swap.dag(), swap_ids

    # returns the swap operation
    return rt_swap, rt_swap.dag()


def apply_DCnot(rho, system_qubits:tuple, target:tuple, fidelity):
    # target: apply distributed cnot on which qubits -> (control, target)
     
    assert(target[0]!=target[1])

    # density matrix for each state of the computational basis
    dcnot_mx = build_DCnot(fidelity)
    
    Nqb = len(rho.dims[0])
    # Iden_Nqb = tensor([si]*Nqb)
    
    rho_new = 0
    ketA = [0]*4
    ketB = [0]*4
    
    sw, sw_d = cache_make_swap(system_qubits, target)

    # ketA, ketB : generate complete computational basis
    for i, stA in enumerate(itertools.product('01', repeat=2)):
        if ketA[i]==0:
            ketA[i] = sw*tensor([ket(stA), *([si]*(Nqb-2))])
        for j,stB in enumerate(itertools.product('01', repeat=2)):
            if ketB[j]==0:
                ketB[j] = sw * tensor([ket(stB), *([si]*(Nqb-2))])
            
            # rho contracted with the computational basis state <ketA| rho |ketB>
            rho_sub = ketA[i].dag() * rho * ketB[j]

            # use precomputed density matrices 
            rho_new +=  tensor( Qobj(dcnot_mx[i][j], dims=[[2, 2], [2, 2]]), rho_sub) 
    
    rho_new = (sw * rho_new * sw_d).unit()
    return rho_new


def apply_DCnot_preswap_rho(rho, fidelity):
    # Assumes the control and target qubit already at positions (0,1), respectively
    # More efficient due to less swaps

    # density matrix for each state of the computational basis
    dcnot_mx = build_DCnot(fidelity)
    
    Nqb = len(rho.dims[0])
    # Iden_Nqb = tensor([si]*Nqb)
    
    rho_new = 0
    ketA = [0]*4
    ketB = [0]*4
    
    # sw, sw_d = cache_make_swap(system_qubits, target)

    # ketA, ketB : generate complete computational basis
    for i, stA in enumerate(itertools.product('01', repeat=2)):
        if ketA[i]==0:
            ketA[i] = tensor([ket(stA), *([si]*(Nqb-2))])
        for j,stB in enumerate(itertools.product('01', repeat=2)):
            if ketB[j]==0:
                ketB[j] = tensor([ket(stB), *([si]*(Nqb-2))])
            
            # rho contracted with the computational basis state <ketA| rho |ketB>
            rho_sub = ketA[i].dag() * rho * ketB[j]

            # use precomputed density matrices 
            rho_new +=  tensor( Qobj(dcnot_mx[i][j], dims=[[2, 2], [2, 2]]), rho_sub) 

    return rho_new.unit()


def apply_DUnitary(u2, rho, system_qubits:tuple, target:tuple, fidelity, exactCNOT=False, exactU=False, getUnitary=False, cnotUses=2):
    Nqb = len(rho.dims[0])
    
    sw, sw_d = cache_make_swap(tuple(system_qubits), target)
    
    if exactU:
        Ubig = sw * tensor([u2, *([si]*(Nqb-2)) ]) * sw_d
        if not getUnitary:
            return  Ubig * rho * Ubig.dag()
        else: 
            return  Ubig * rho * Ubig.dag(), Ubig, u2

    qc = two_qubit_cnot_decompose(u2.full(), _num_basis_uses=cnotUses)


    # operators before the CNOT gate
    U00 = Qobj(u3_matrix(qc.data[0][0].params))
    U01 = Qobj(u3_matrix(qc.data[1][0].params))
    U0 = tensor(U00, U01)
    # print(U0)

    # operators after the CNOT gate
    U10 = Qobj(u3_matrix(qc.data[3][0].params))
    U11 = Qobj(u3_matrix(qc.data[4][0].params))
    U1 = tensor(U10, U11)
    # print(U1)

    if cnotUses >= 2:
        U20 = Qobj(u3_matrix(qc.data[6][0].params))
        U21 = Qobj(u3_matrix(qc.data[7][0].params))
        U2 = tensor(U20, U21)

    
    Ubig0 = sw * tensor([U0, *([si]*(Nqb-2)) ]) * sw_d
    Ubig1 = sw * tensor([U1,*([si]*(Nqb-2))]) * sw_d
    
    if cnotUses == 2:
        Ubig2 = sw * tensor([U2,*([si]*(Nqb-2))]) * sw_d
    
    qb_lbl = {qb:i for i,qb in enumerate(system_qubits)}
    
    rho_new = Ubig0 * rho * Ubig0.dag()
    if exactCNOT:
        rho_new = cnot(N=Nqb, control=qb_lbl[target[0]], target=qb_lbl[target[1]])\
             * rho_new \
             * cnot(N=Nqb, control=qb_lbl[target[0]], target=qb_lbl[target[1]]).dag()
    else:
        rho_new = apply_DCnot(rho_new, system_qubits, target, fidelity)
    rho_new = Ubig1 * rho_new * Ubig1.dag()

    if cnotUses == 2:
        if exactCNOT:
            rho_new = cnot(N=Nqb, control=qb_lbl[target[0]], target=qb_lbl[target[1]])\
                * rho_new \
                * cnot(N=Nqb, control=qb_lbl[target[0]], target=qb_lbl[target[1]]).dag()
        else:
            rho_new = apply_DCnot(rho_new, system_qubits, target, fidelity)
        rho_new = Ubig2 * rho_new * Ubig2.dag()

    if not getUnitary:
        return rho_new
    else: 
        # gets exact unitary
        Ubig = sw.dag() * tensor([u2, *([si]*(Nqb-2)) ]) * sw
        return rho_new, Ubig


def apply_DUnitary_preswap(u2, rho, system_qubits:tuple, target:tuple, fidelity, exactCNOT=False, exactU=False, getUnitary=False, cnotUses=2):
    Nqb = len(rho.dims[0])
    
    sw, sw_d = cache_make_swap(tuple(system_qubits), target)
    
    # Swap target qubits to positions 0,1
    rho_work = sw_d * rho * sw 

    if exactU:
        Ubig = tensor([u2, *([si]*(Nqb-2)) ])
        rho_work = Ubig * rho_work * Ubig.dag()
        rho_new = sw * rho_work * sw_d
        if not getUnitary:
            return rho_new
        else: 
            return  rho_new, Ubig, u2

    qc = two_qubit_cnot_decompose(u2.full(), _num_basis_uses=cnotUses)


    # operators before the CNOT gate
    U00 = Qobj(u3_matrix(qc.data[0][0].params))
    U01 = Qobj(u3_matrix(qc.data[1][0].params))
    U0 = tensor(U00, U01)
    # print(U0)

    # operators after the CNOT gate
    U10 = Qobj(u3_matrix(qc.data[3][0].params))
    U11 = Qobj(u3_matrix(qc.data[4][0].params))
    U1 = tensor(U10, U11)
    # print(U1)

    if cnotUses >= 2:
        U20 = Qobj(u3_matrix(qc.data[6][0].params))
        U21 = Qobj(u3_matrix(qc.data[7][0].params))
        U2 = tensor(U20, U21)

    
    Ubig0 = tensor([U0, *([si]*(Nqb-2)) ])
    Ubig1 = tensor([U1,*([si]*(Nqb-2))])
    
    if cnotUses >= 2:
        Ubig2 = tensor([U2,*([si]*(Nqb-2))])
        
    rho_work = Ubig0 * rho_work * Ubig0.dag()

    # Apply mid (or final) CNOT
    if exactCNOT: 
        rho_work = cnot(N=Nqb, control=0, target=1)\
             * rho_work \
             * cnot(N=Nqb, control=0, target=1)
    else: # if distributed
        rho_work = apply_DCnot_preswap_rho(rho_work, fidelity)
    # Apply mid (or final) U
    rho_work = Ubig1 * rho_work * Ubig1.dag()

    if cnotUses >= 2:
        # Apply CNOT
        if exactCNOT: # if exact
            rho_work = cnot(N=Nqb, control=0, target=1)\
                * rho_work \
                * cnot(N=Nqb, control=0, target=1)
        else: # if distributed
            rho_work = apply_DCnot_preswap_rho(rho_work,  fidelity)
        
        # Apply final U 
        rho_work = Ubig2 * rho_work * Ubig2.dag()
    

    # Swap target qubits back to original positions
    rho_new = sw * rho_work * sw_d

    if not getUnitary:
        return rho_new
    else: 
        # gets exact unitary
        Ubig = sw_d * tensor([u2, *([si]*(Nqb-2)) ]) * sw
        return rho_new, Ubig



class DistributedTrotter: # evolve trotterization with distributed Unitaries, DT
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

        self.qubits_SWAP = {}

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
        # Um_temp = [Um.to(selfUm.to(self.dtype).dtype)] # moving things into the gpu
        # [Um_temp.append(si.to(self.dtype)) for _ in range(expand)]

        # U12 x I^n
        Um_extended = tensor(Um_temp)
        
        # Using the SWAP matrices calculated at __init__
        unit = 1
        # if self.gpu:
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
 
    def evolve_discrete_unitaries(self, m_trotter, evol_time, rho0, fidelity=1, exactCNOT=False, exactU=False, getUnitary=False, stepf=-1, usePreSwap=True, get_max_norm=False, hamiltonian_norm='fro'):
        # self.hamiltonians = 
        #           {'local': 
        #               {'linear':[
        #                   {'matrix':hamilt_matrix, 
        #                   'qubits':hamilt_qubits, 'idx':index}], 
        #               'antilinear':[], 
        #               'complex':[]}, 
        #           'non-local': {'linear':[], 'antilinear':[], 'complex':[]}}
        
        
        self.mx_hist = [1]*self.total_mx # matrix history, keeping track of the past iteration
        # keeps track of current (expm)**k
        self.si = qutip.qeye(2)

        self.H_T = [ [self.H0, lambda t, arg: 1-t/evol_time], [self.HF, lambda t, arg: t/evol_time] ] 

        # save the hamiltonian norms for possible use later
        self.norm_history = []
        
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
        
        self.rho0 = rho0
        rho_now = rho0

        unitary = 1
        unitary_out = 1
        step = 0
        # ids = [*, *self.hamilt_ids['non-local']]
        for k in range(m_trotter):

            if get_max_norm: # calculate hamiltonian norm
                current_norm = time_list_evaluate(self.H_T, time=k * evol_time/m_trotter, args={}).norm(norm=hamiltonian_norm)
                self.norm_history.append(current_norm)
            
            # calculate local propagation first
            unitary = 1
            for h_id in self.hamilt_ids['local']:
                # t0 = time.process_time_ns()
                
                # 2 qubit unitary
                uk = self.calc_matrix_iteration(h_id, k, m_trotter, evol_time)
                qubits = self.hm[h_id]['qubits']

                # full system unitary
                Uk = self.build_system_unitary(uk, len(self.qubits)-len(qubits)+1, self.qubits_SWAP[tuple(qubits)][0], self.qubits_SWAP[tuple(qubits)][1])

                # if h_id == 0:
                #     unitary = Uk
                # else:
                    # t0 = time.process_time_ns()
                unitary = Uk * unitary 
                if getUnitary or stepf>0:
                    unitary_out = Uk * unitary_out
                step+=1 
                if stepf > 0 and step >= stepf:
                    print("Exiting, outputing unitary %d, qubits "%(h_id), self.hm[h_id]['qubits'])
                    # print("Going out 1")
                    return unitary_out, Uk, uk
                    # total_matmul+=time.process_time_ns()-t0
            
            # Evolve local hamiltonians in a 'single' step
            rho_now = unitary * rho_now * unitary.dag()
            

            # Evolve distributed unitaries
            for h_id in self.hamilt_ids['non-local']:
                # t0 = time.process_time_ns()
                
                # 2 qubit unitary
                uk = self.calc_matrix_iteration(h_id, k, m_trotter, evol_time)

                # Distributed unitary applied on 2 qubits
                qubits = self.hm[h_id]['qubits']
                
                if usePreSwap:
                    rho_now = apply_DUnitary_preswap(uk, rho_now, tuple(self.qubits), (qubits[0],qubits[1]), fidelity, exactCNOT=exactCNOT, exactU=exactU, getUnitary=getUnitary)
                else: 
                    rho_now = apply_DUnitary(uk, rho_now, tuple(self.qubits), (qubits[0],qubits[1]), fidelity, exactCNOT=exactCNOT, exactU=exactU, getUnitary=getUnitary)

                if getUnitary or stepf>0:
                    unitary_out = Uk * unitary_out
                
                step+=1 
                if stepf > 0 and step >= stepf:
                    print("Exiting, outputing unitary %d, qubits "%(h_id), self.hm[h_id]['qubits'])
                    # print("Going out 2")
                    return unitary_out, Uk


        self.rhof = rho_now
        if getUnitary:
            # print("Going out 3")
            return rho_now, unitary_out
        else:
            return rho_now
    
    def get_rhof(self):
        return self.rhof

    def calc_metric(self, metric, hamiltonian_norm='fro' ):
        if metric == 'E0':
            return self.HF.eigenstates()[0][0]
        if metric == 'En':# energy of final state
            HF_sqrt = self.HF.sqrtm()
            return  np.real((HF_sqrt * self.rhof * HF_sqrt).tr())
        if metric == 'OE':# Overlap error: 1 - < | >**2 = 1-fidelity 
            egs = self.HF.eigenstates()[0][0]
            psi_gs_arr = [self.HF.eigenstates()[1][i] for i, en in enumerate(self.HF.eigenstates()[0]) if en == egs]
            return 1 - overlap_probability_degen_gs_rho(self.rhof, psi_gs_arr)
        if metric == 'Emed':
            return np.real((self.psi_0.dag() * self.HF * self.psi_0).tr())
        if metric == '||HF||':
            return hnorm(hamiltonian_norm,self.HF)
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
        if 'SE' in metrics: # state error: || |SA> - |UT> ||
            sa_psif = sa.final_psi
            metrics_result['SE'] = (self.rhof - ket2dm(sa_psif)).norm()
        if 'AEn' in metrics: # simple adiabatic (pure annealing) energy
            metrics_result['AEn'] = sa.get_final_state_energy()
        
        for mtcs in metrics:
            if not (mtcs in {'SE', 'AEn'}): 
                metrics_result[mtcs] = self.calc_metric(mtcs, hamiltonian_norm=hamiltonian_norm)
        # print(metrics_result)
        return metrics_result

