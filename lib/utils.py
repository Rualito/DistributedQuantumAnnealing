# Version: MPC .3.2
# now with print_state


from qutip import *
qutip.settings.num_cpus = 4

import numpy as np


__all__ = [
 'build_spin_chain',
 'overlap_probability_degen_gs',
 'calc_errors',
 'JID_to_params',
 'calc_method_errors',
 'make_hamiltonian_from_spec',
 'make_spec_dense',
 'JID_to_params_dense', 
 'calc_method_errors_rho', 
 'calc_method_errors_rho',
 'make_spec_cut',
 'print_state'
 ]

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

def print_state(state):
    Nqb = len(state.dims[0]) 
    st_list = []
    for st in itertools.product('01', repeat=Nqb):
        ketc = ket(''.join(st))
        ci = (ketc.dag() * state).norm()
        if ci!=0:
            st_list.append(f'{ci}|{"".join(st)}>')
    return ' + '.join(st_list)

def build_hamiltonian(qbits, linear_args={}, quadratic_args={}):
    sx_i, sy_i, sz_i = spin_tensors(len(qbits))

    pauli = {'I':tensor([si]*len(qbits)),
    'x': sx_i, 'y':sy_i, 'z':sz_i}

    qb_idx = {}
    for i,qb in enumerate(qbits):
        qb_idx[qb] = i

    H = 0*sx_i[0]
    
    # inserting linear terms
    for qb, (pauli_i,weight) in linear_args.items():
        # qubit index = qb_idx[qb]
        operator = pauli[pauli_i][qb_idx[qb]]
        H += weight * operator

    # inserting quadratic terms
    for (qb0, qb1), (pauli_i,weight) in quadratic_args.items():
        # qubit index = qb_idx[qb]
        operator0 = pauli[pauli_i][qb_idx[qb0]]
        operator1 = pauli[pauli_i][qb_idx[qb1]]
        
        H += weight * operator0*operator1
    
    return H
    

# Jgenf is a coupling factor generating function
# inputs the coupling index, outputs a coupling strength
# N qubits, l  local number (how many local subsystems)
def build_spin_chain(N:int, l:int, Jgenf):
    
    # Generate qubit names
    qubits = []
    for i in range(N):
        qubits.append(str(i))
    
    if ( int(N/l) != N/l):
        raise ValueError(f"Split: l={l} not compatible with number of qubits: N={N}")

    local_size = int(N/l)
    non_local_links = l-1

    local_qlist = [] 
    nonlocal_qlist = []

    for i in range(l):
        local_qlist.append([])
        for j in range(i*local_size, (i+1)*local_size):
            local_qlist[-1].append(qubits[j])
        final_q = (i+1)*local_size-1
        if i+1<l:   
            nonlocal_qlist.append( [qubits[final_q], qubits[final_q+1]] )
    
    H_null = build_hamiltonian(qubits, {}, {})

    # only sigmax 
    HL0 = build_hamiltonian(qubits, {qubits[i]:('x', 1) for i in range(N)}, {})
    HN0 = H_null
    idx = 0
    HL1 = H_null

    HL_spec = {}

    for k, qb_l in enumerate(local_qlist):
        # final, local hamiltonian
        # ignore local fields
        H_L_x = build_hamiltonian(qb_l, { q:('x', 1) for q in qb_l}, {})
        H_L_z = build_hamiltonian(qb_l, {}, {(qb, qb_l[i+1]):('z', Jgenf(idx)) for i, qb in enumerate(qb_l[:-1])})
        
        HL_spec[f'HL{k}'] = ( qb_l, [[H_L_x, lambda s, arg: 1-s], [H_L_z, lambda s, arg: s]], 'complex')

        HL1 += build_hamiltonian(qubits, {}, {(qb, qb_l[i+1]):('z', Jgenf(idx)) for i, qb in enumerate(qb_l[:-1])})
        idx += 1

    HN_spec = {}

    HN1 = H_null
    for k, qb_l in enumerate(nonlocal_qlist):
        # final, local hamiltonian
        # ignore local fields
        H_N_z = build_hamiltonian(qb_l, {}, {(qb_l[0], qb_l[1]):('z', Jgenf(idx))})
        
        HN_spec[f'HN{k}'] = (qb_l, H_N_z, 'linear')

        HN1 += build_hamiltonian(qubits, {}, {(qb_l[0], qb_l[1]):('z', Jgenf(idx))})
        idx += 1
    


    return {'AT':(HL0, HL1, HN0, HN1), 'UT':(HL_spec, HN_spec)}

def overlap_probability_degen_gs(state, psi_gs_arr):
    temp = 0
    for psi_gs in psi_gs_arr:
        temp += np.abs(psi_gs.overlap(state))**2

    return temp

def overlap_probability_degen_gs_rho(rho, psi_gs_arr):
    temp = 0
    for psi_gs in psi_gs_arr:
        temp += psi_gs.dag() * rho * psi_gs
        # np.abs(psi_gs.overlap(state))**2

    return np.real(temp.tr())

def calc_errors(sa_psif, st_psif, HF, egs, psi_gs_arr):
    ST_OvErr = 1 - overlap_probability_degen_gs(st_psif, psi_gs_arr)
    ST_EnErr = np.real((st_psif.dag() * HF * st_psif)[0][0][0] - egs)/np.abs(egs)
    # UT_StErr = 0


    SA_OvErr = 1 - overlap_probability_degen_gs(sa_psif, psi_gs_arr)
    SA_EnErr = np.real((sa_psif.dag() * HF * sa_psif)[0][0][0] - egs)/np.abs(egs)
    # SA_StErr = 0

    SA_ST_OvErr = 1-np.abs(st_psif.overlap(sa_psif))**2
    SA_ST_StErr = (st_psif - sa_psif).norm()

    return ST_OvErr, ST_EnErr, SA_OvErr, SA_EnErr, SA_ST_OvErr, SA_ST_StErr

# returns overlap error and energy of the state
def calc_method_errors(st_psif, HF, psi_gs_arr):
    ST_OvErr = 1 - overlap_probability_degen_gs(st_psif, psi_gs_arr)
    ST_En = np.real((st_psif.dag() * HF * st_psif)[0][0][0])

    return ST_OvErr, ST_En

def calc_method_errors_rho(rho_f, HF, psi_gs_arr):
    ST_OvErr = 1 - overlap_probability_degen_gs_rho(rho_f, psi_gs_arr)
    
    HF_sqrt = HF.sqrtm()
    ST_En = np.real((HF_sqrt * rho_f * HF_sqrt).tr())

    return ST_OvErr, ST_En

def JID_to_params(JID:int, NQubits:int,  Jabs=1):
    # By default |J_i| = 1

    J_arr = []
    for i in range(NQubits-1):
        J_arr.append( Jabs*(2*((JID & 1<<i) >> i)-1)  )

    return J_arr

# Specific to dense qubit networks
def JID_to_params_dense(JID:int, NQubits:int,  Jabs=1):
    # By default |J_i| = 1

    J_arr = []
    for i in range(int(NQubits*(NQubits-1)/2)):
        J_arr.append( Jabs*(2*((JID & 1<<i) >> i)-1)  )

    return J_arr

def make_hamiltonian_from_spec(sublocals, system_spec):
    # NOTE: Currently processing sigy separately from sigz
    # they don't commute, and so it is expected that they don't act on the same qubit non-locally
    # If it is required for some graph splitting, then it is assumed
    # that minor-embedding was used to surpass this limitation
    
    # Make ordered qubit list
    qb_list = []
    for sub in sublocals:
        for qb in sub:
            qb_list.append(qb)

    Hnull = build_hamiltonian(qb_list, {}, {})

    HL0 = Hnull
    HL1 = Hnull
    HN0 = Hnull
    HN1 = Hnull

    # Adding sigx configuration -> needs to be specified
    # for qb in qb_list:
    #     HL0 += build_hamiltonian(qb_list, {qb:('x', 1)}, {})
        
    # Adding local sigz, final hamiltonian 
    for subloc_spec in system_spec['local']:
        for (qbs, cpl, tm) in subloc_spec:
            Htemp = 0
            if len(qbs)==1:
                Htemp = build_hamiltonian(qb_list, {qbs[0]:cpl}, {} )
            elif len(qbs) == 2:
                Htemp = build_hamiltonian(qb_list, {} ,{qbs:cpl})
            
            if tm==0:
                HL0 += Htemp
            elif tm==1:
                HL1 += Htemp
            
    # Adding nonlocal 
    for qbs, cpl, tm in system_spec['nonlocal']:
        Htemp = 0
        if len(qbs)==1:
            Htemp = build_hamiltonian(qb_list, {qbs[0]:cpl}, {} )
        elif len(qbs) == 2:
            Htemp = build_hamiltonian(qb_list, {} ,{qbs:cpl})
        
        if tm==0:
            HN0 += Htemp
        elif tm==1:
            HN1 += Htemp

    HL_spec = {}
    HN_spec = {}

    L_idx = 0
    NL_idx = 0
    # print(sublocals)
    
    for sub, subloc_spec in zip(sublocals, system_spec['local']):
        # creating local hamiltonians
        # for this sublocal system
        HL_0 = 0
        HL_1 = 0
        for (qbs, cpl, tm) in subloc_spec:
            Htemp = 0
            if len(qbs)==1:
                Htemp = build_hamiltonian(sub, {qbs[0]:cpl}, {} )
            elif len(qbs) == 2:
                Htemp = build_hamiltonian(sub, {} ,{qbs:cpl})
            if tm==0:
                HL_0 += Htemp
            elif tm==1:
                HL_1 += Htemp
        
        HL_spec[f'HL{L_idx}'] = (sub, [[HL_0, lambda s, arg: 1-s], [HL_1, lambda s, arg: s]], 'complex')
        L_idx+=1
    
    for qbs, cpl, tm in system_spec['nonlocal']:
        Htemp = 0
        if len(qbs)==1:
            Htemp = build_hamiltonian(qbs, {qbs[0]:cpl}, {} )
        elif len(qbs) == 2:
            Htemp = build_hamiltonian(qbs, {} ,{qbs:cpl})
        if tm==0:
            HN_spec[f'HN{NL_idx}'] = (qbs, Htemp, 'antilinear')
        elif tm==1:
            HN_spec[f'HN{NL_idx}'] = (qbs, Htemp, 'linear')
                    
        NL_idx+=1
    
    return {'AT':(HL0, HL1, HN0, HN1), 'UT':(HL_spec, HN_spec)}

def make_spec_dense(Nq, sp, JID, Jabs=1):

    Jarr = JID_to_params_dense(JID, Nq, Jabs)
    
    qb_list = ([], [])

    for i in range(sp):
        qb_list[0].append(str(i))

    for i in range(sp, Nq):
        qb_list[1].append(str(i))

    system_spec = {'local':[[],[]], 'nonlocal':[]}
    idx = 0
    for i, qb0 in enumerate(qb_list[0]):
        for j, qb1 in enumerate(qb_list[0][i+1:]):
            # print(qb0, qb1)
            system_spec['local'][0].append( ((qb0, qb1), ('z', Jarr[idx])) )
            idx+=1

    for i, qb0 in enumerate(qb_list[1]):
        for j, qb1 in enumerate(qb_list[1][i+1:]):
            # print(qb0, qb1)
            system_spec['local'][1].append( ((qb0, qb1), ('z', Jarr[idx])) )
            idx+=1

    for i, qb0 in enumerate(qb_list[0]):
        for j, qb1 in enumerate(qb_list[1]):
            system_spec['nonlocal'].append( ((qb0, qb1), ('z', Jarr[idx])) )
            idx+=1
            
    # sublocals = qb_list
    return qb_list, system_spec

# Nq: total number of qubits 
# sp: qubit split: Nq = l1 + l2; l1 = sp; l2 = Nq-sp
# JID: sign configuration of the graph
# SID: sparse configuration (integer that indexes the connections) 
def make_spec_sparse(Nq, sp, JID, SID, Jabs=1):

    Jarr = JID_to_params_dense(JID, Nq, Jabs)
    qb_list = ([], [])

    for i in range(sp):
        qb_list[0].append(str(i))

    for i in range(sp, Nq):
        qb_list[1].append(str(i))

    # Generate all the local connections
    system_spec = {'local':[[],[]], 'nonlocal':[]}
    idx = 0
    for i, qb0 in enumerate(qb_list[0]):
        for j, qb1 in enumerate(qb_list[0][i+1:]):
            # print(qb0, qb1)
            system_spec['local'][0].append( ((qb0, qb1), ('z', Jarr[idx])) )
            idx+=1

    for i, qb0 in enumerate(qb_list[1]):
        for j, qb1 in enumerate(qb_list[1][i+1:]):
            # print(qb0, qb1)
            system_spec['local'][1].append( ((qb0, qb1), ('z', Jarr[idx])) )
            idx+=1


    # Generate all the sparse connections
    sparse_config_n = sp * (Nq-sp)

    # keeping the SID to the correct range
    SID = SID % sparse_config_n
    
    configs = []
    for i in range(sp):
        for j in range(Nq-sp):
            configs.append( (i,j) )

    sparse_config = configs[SID]

    for i, qb0 in enumerate(qb_list[0]):
        for j, qb1 in enumerate(qb_list[1]):
            # Have this additional condition to have only for the selected sparse connections
            # for example, in L1, connect all the qubits but the last 'sparse_config[0]' ones
            if (i < len(qb_list[0])-sparse_config[0]) and (j < len(qb_list[1])-sparse_config[1]):
                system_spec['nonlocal'].append( ((qb0, qb1), ('z', Jarr[idx])) )
            # Even if connection is taken out, the coupling stays the same
            idx+=1

    return qb_list, system_spec

def make_spec_cut(H_spec, JID, CID, Jabs=1):
    c_order = np.binary_repr(CID)
    for i in range(len(c_order), len(H_spec['qubits'])):
        c_order = '0' + c_order

    L1 = []
    L2 = []
    for c, qb in zip(c_order, H_spec['qubits']):
        if c=='0':
            L1.append(qb)
        elif c=='1':
            L2.append(qb)

    # qb_list = H_spec['qubits']
    J_arr = JID_to_params(JID, len(H_spec['couplings'])+1, Jabs)
    system_spec = {'local':[[], []], 'nonlocal':[]}
    
    for i,conn in enumerate(H_spec['couplings']):
        if ((conn[0] in L1) and (conn[1] in L1 )):
            # conn[0] and conn[1] are on the same sublocal
            system_spec['local'][0].append( (conn, ('z', J_arr[i])) )
        elif ((conn[0] in L2) and (conn[1] in L2 )):
            system_spec['local'][1].append( (conn, ('z', J_arr[i])) )
        elif ((conn[0] in L1) and (conn[1] in L2))\
             or ((conn[0] in L2) and (conn[1] in L1)):
            system_spec['nonlocal'].append(( conn, ('z', J_arr[i])))
        
    return (L1, L2), system_spec
