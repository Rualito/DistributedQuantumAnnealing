
import numpy as np
import ast
import inspect

class Hargs:
    metrics = ['E0', 'En', 'Emed', '||HL||', '||HN||', 'OE']
    extra_args = {'psi0': None, 'H_norm': ['tr', 'spectral']}
    def parse_parameters(param_string):
        param_names = inspect.getfullargspec(Hargs.h_spec_build).args
        list_literal = ast.literal_eval(param_string)
        temp = {}
        for i, el in enumerate(list_literal):
            temp[param_names[i]] = el
        return temp

    def hargs_from_file(file_str):
        param = Hargs.parse_parameters(file_str)
        return Hargs.h_spec_build(param['N'], param['s'],param['JID'], param['SID'], param['Jabs'])

    def h_spec_build(N, s, JID, SID, Jabs):
        # Trotter sparse
        
        def JID_to_params_dense(JID:int, NQubits:int,  Jabs=1):
            # By default |J_i| = 1
            J_arr = []
            for i in range(int(NQubits*(NQubits-1)/2)):
                J_arr.append( Jabs*(2*((JID & 1<<i) >> i)-1)  )

            return J_arr
        
        Jarr = JID_to_params_dense(JID, N, Jabs)
        qb_list = ([], [])

        for i in range(s):
            qb_list[0].append(str(i))

        for i in range(s, N): 
            qb_list[1].append(str(i))

        # Generate all the local connections
        system_spec = {'local':[[],[]], 'nonlocal':[]}
        idx = 0
        for i, qb0 in enumerate(qb_list[0]):
            system_spec['local'][0].append( ((qb0, ), ('x', 1), 0) )
            for j, qb1 in enumerate(qb_list[0][i+1:]):
                # print(qb0, qb1)
                system_spec['local'][0].append( ((qb0, qb1), ('z', Jarr[idx]), 1) )
                idx+=1

        for i, qb0 in enumerate(qb_list[1]):
            system_spec['local'][1].append( ((qb0, ), ('x', 1), 0) )
            for j, qb1 in enumerate(qb_list[1][i+1:]):
                # print(qb0, qb1)
                system_spec['local'][1].append( ((qb0, qb1), ('z', Jarr[idx]), 1) )
                idx+=1


        # Generate all the sparse connections
        sparse_config_n = s * (N-s)

        # keeping the SID to the correct range
        SID = SID % sparse_config_n
        
        configs = []
        for i in range(s):
            for j in range(N-s):
                configs.append( (i,j) )

        sparse_config = configs[SID]

        for i, qb0 in enumerate(qb_list[0]):
            for j, qb1 in enumerate(qb_list[1]):
                # Have this additional condition to have only for the selected sparse connections
                # for example, in L1, connect all the qubits but the last 'sparse_config[0]' ones
                if (i < len(qb_list[0])-sparse_config[0]) and (j < len(qb_list[1])-sparse_config[1]):
                    system_spec['nonlocal'].append( ((qb0, qb1), ('z', Jarr[idx]), 1) )
                # Even if connection is taken out, the coupling stays the same
                idx+=1

        return qb_list, system_spec

