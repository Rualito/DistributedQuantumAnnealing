
import numpy as np
import ast
import inspect

class Hargs:
    metrics = ['E0', 'En', 'Emed', '||HL||', '||HN||', 'OE', 'E0_reduced', 'En_reduced', 'Emed_reduced', 'OE_reduced']
    extra_args = {'psi0': 'mpc', 'H_norm': ['tr', 'spectral']}
    def parse_parameters(param_string):
        param_names = inspect.getfullargspec(Hargs.h_spec_build).args
        list_literal = ast.literal_eval(param_string)
        temp = {}
        for i, el in enumerate(list_literal):
            temp[param_names[i]] = el
        return temp

    def hargs_from_file(file_str):
        param = Hargs.parse_parameters(file_str)
        return Hargs.h_spec_build(param['ha'],param['hb'],param['hc'],param['hd'],param['jab'],param['jac'],param['jad'],param['jbc'],param['jbd'],param['jcd'])

    def h_spec_build(ha, hb, hc, hd, jab, jac, jad, jbc, jbd, jcd):
        # fully 4, D0
        # MPT
        
        sublocals = (['a0', 'b0', 'c0', 'd0'], ['a1', 'b1', 'c1', 'd1'])
        system_spec = {'local':[[], []], 'nonlocal':[]}   
        # local fields
        system_spec['local'][0].append( (('a0',), ('z', ha/2), 1)) 
        system_spec['local'][0].append( (('b0',), ('z', hb/2), 1)) 
        system_spec['local'][0].append( (('c0',), ('z', hc/2), 1)) 
        system_spec['local'][0].append( (('d0',), ('z', hd/2), 1))
        system_spec['local'][0].append( (('a0','b0'), ('z', jab/2), 1)) 
        system_spec['local'][0].append( (('a0','c0'), ('z', jac), 1))
        system_spec['local'][0].append( (('b0','d0'), ('z', jbd), 1))
        system_spec['local'][0].append( (('c0','d0'), ('z', jcd/2), 1))


        system_spec['local'][1].append( (('a1',), ('z', ha/2), 1)) 
        system_spec['local'][1].append( (('b1',), ('z', hb/2), 1)) 
        system_spec['local'][1].append( (('c1',), ('z', hc/2), 1)) 
        system_spec['local'][1].append( (('d1',), ('z', hd/2), 1))
        system_spec['local'][1].append( (('a1','b1'), ('z', jab/2), 1)) 
        system_spec['local'][1].append( (('a1','d1'), ('z', jad), 1))
        system_spec['local'][1].append( (('b1','c1'), ('z', jbc), 1))
        system_spec['local'][1].append( (('c1','d1'), ('z', jcd/2), 1))

        # nonlocal fields: x couplings
        system_spec['nonlocal'].append((('a0', 'a1'), ('x', 1), 0 ))
        system_spec['nonlocal'].append((('b0', 'b1'), ('x', 1), 0 ))
        system_spec['nonlocal'].append((('c0', 'c1'), ('x', 1), 0 ))
        system_spec['nonlocal'].append((('d0', 'd1'), ('x', 1), 0 ))
        
        return sublocals, system_spec

