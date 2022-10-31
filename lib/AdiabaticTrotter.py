import itertools
# %matplotlib inline
import matplotlib.pyplot as plt

from SimpleAdiabatic import SimpleAdiabatic

from qutip import *
qutip.settings.num_cpus = 4

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


default_sch_args = {'t max': 100}

class AdiabaticTrotter:
    # No approximation of non-local evolution
    def __init__(self, HL0, HLF, HN0, HNF,schedule_func=schedule_linear, schedule_args=default_sch_args, display_eigen=-1, qubit_names=[]):
        # HL0: starting local hamiltonian
        # HLF: final local hamiltonian
        # HN0: starting non-local hamiltonian
        # HN0: final non-local hamiltonian
        
        # self.trotter_steps = trotter_steps
        self.sch_args = schedule_args
        self.schedule_func = schedule_func

        self.HL_T = [ [HL0, lambda t, arg: 1-schedule_func(t, schedule_args)], [HLF, lambda t, arg: schedule_func(t, schedule_args)] ] 

        self.HN_T = [ [HN0, lambda t, arg: 1-schedule_func(t, schedule_args)], [HNF, lambda t, arg: schedule_func(t, schedule_args)] ] 

        self.H_T = [ [HL0+HN0, lambda t, arg: 1-schedule_func(t, schedule_args)], [HLF+HNF, lambda t, arg: schedule_func(t, schedule_args)] ] 

        self.HS = {'L':self.HL_T, 'N':self.HN_T, 'Tot':self.H_T}

        self.psi_0 = (HL0+HN0).eigenstates()[1][0]
        default_display_eigen = int(np.ceil(np.log2(len((HL0+HN0).eigenstates()[0]))))

        self.qb_names = qubit_names
        if len(qubit_names) < default_display_eigen:
            # give default names to qubits that don't have a name
            for i in range(len(qubit_names), default_display_eigen):
                self.qb_names.append(f'q{i}')
        
        self.total_eigen = min([default_display_eigen, display_eigen]) if display_eigen > 0 else default_display_eigen

        self.has_run_SA_benchmark = False
        # self.run_SA_benchmark()

    def run_SA_benchmark(self, schedule_args=None, time_res=15):
        # Run Simple Adiabatic first as a benchmark for later
        if not self.has_run_SA_benchmark: 
            qubit_dict = {}
            for name in self.qb_names:
                qubit_dict[name] = 0
            
            schedule_args = schedule_args if schedule_args is not None else self.sch_args

            self.sa = SimpleAdiabatic(qubit_dict, {}, time_resolution=time_res, schedule_args=schedule_args)

            self.sa.H_0 = self.H_T[0][0]
            self.sa.H_1 = self.H_T[1][0]

            # print("Solving SA")

            self.sa.reload_problem()
            self.sa.solve_main_eq()

            self.gs_SA = self.sa.get_final_state_energy()

            self.has_run_SA_benchmark = True

    def __process_rho__(self, t, psi):
        # input: 
        #   t - current time in the integration
        #   psi - current solution to the Hamiltonian (Lindbladian in general)
        
        # this function saves the results at each specified time in the timelist
        # global idx, min_gap, final_psi
        # evaluate the Hamiltonian with gradually switched on interaction 
        # essentially calculating the hamiltonian at time t: H(t) = f1(t) H_1 + f2(t) H_2
        H = qobj_list_evaluate(self.HS['Tot'], t, {}) 
        self.final_psi = psi
        # find the M=solver_args['eigenvals']
        #  lowest eigenvalues of the system
        evals, ekets = H.eigenstates(eigvals=self.total_eigen)
        
        self.evals_mat[self.__idx,:] = np.real(evals)
        
        # find the overlap between the eigenstates and psi 
        for n, eket in enumerate(ekets):
            self.P_mat[self.__idx,n] = abs((eket.dag().data * psi.data)[0,0])**2    
        
        # current 'process id' - time index in taulist
        self.__idx += 1

    def __process_rho_fast__(self, t, psi):
        # input: 
        #   t - current time in the integration
        #   psi - current solution to the Hamiltonian (Lindbladian in general)
        
        # this function saves the results at each specified time in the timelist
        # global idx, min_gap, final_psi
        # evaluate the Hamiltonian with gradually switched on interaction 
        # essentially calculating the hamiltonian at time t: H(t) = f1(t) H_1 + f2(t) H_2
        H = qobj_list_evaluate(self.H_TT, t, {}) 
        self.final_psi = psi
        # find the M=solver_args['eigenvals']
        #  lowest eigenvalues of the system
        evals, ekets = H.eigenstates(eigvals=self.total_eigen)
        
        self.evals_mat[self.__idx,:] = np.real(evals)
        
        # find the overlap between the eigenstates and psi 
        for n, eket in enumerate(ekets):
            self.P_mat[self.__idx,n] = abs((eket.dag().data * psi.data)[0,0])**2    
        
        # current 'process id' - time index in taulist
        self.__idx += 1


    def solve_main_eq(self, trotter_steps=10, interm_steps=10):
        # time steps ignore, only trotter steps are considered
        # (since time_list doesn't affect performance of mesolve)
        self.__idx = 0

        psi_0 = self.psi_0
        # total_time_segments = int(time_steps/trotter_steps) * trotter_steps

        self.time_list = np.linspace(0, self.sch_args['t max'], interm_steps*2*trotter_steps)

        self.evals_mat = np.zeros( (interm_steps*2*trotter_steps, self.total_eigen) )
        self.P_mat = np.zeros( (interm_steps*2*trotter_steps, self.total_eigen) )
        self.final_psi = psi_0

        for i in range(trotter_steps):
            # evolve system with a series of trotter local to non-local steps
            
            # start and end of evolution time of this step
            # print(f"At Trotter step {i}")
            t_now = i * self.sch_args['t max']/trotter_steps
            t_next = (i+1) * self.sch_args['t max']/trotter_steps
            
            # t_avg = (t_now + t_next)/2

            # the current psi is the psi at the end of the last evolution
            time_steps = np.linspace(t_now, t_next, interm_steps)
            # Trotterization Local step
            mesolve(self.HL_T, self.final_psi, time_steps, [], lambda t, psi: self.__process_rho__(t, psi))
            

            # Trotterization Non-Local step
            mesolve(self.HN_T, self.final_psi, time_steps, [], lambda t, psi: self.__process_rho__(t, psi))
   
    def solve_main_eq_fast(self, trotter_steps=10, t_per_trotter=3):
        # solves evolution of main equation by considering the whole adiabatic evolution instead of by trotter segments

        # time steps ignore, only trotter steps are considered
        # (since time_list doesn't affect performance of mesolve)
        self.__idx = 0

        psi_0 = self.psi_0
        # total_time_segments = int(time_steps/trotter_steps) * trotter_steps

        def s(t):
            return self.schedule_func(t, self.sch_args)

        Ds = 1.0/(trotter_steps)
        
        def f_map(s, order, id):
            i = int(2*s*trotter_steps) # linear trotter mapping
            if i >= 2*trotter_steps:
                i = 2*trotter_steps-1
            elif i<0:
                i=0
            return (order[i] == id)*1.0
        
        self.time_list = np.linspace(0, self.sch_args['t max'], int(2*t_per_trotter*trotter_steps))
        self.evals_mat = np.zeros( (int(t_per_trotter*2*trotter_steps), 
                                    self.total_eigen) )
        self.P_mat = np.zeros( (int(t_per_trotter*2*trotter_steps), 
                                self.total_eigen) )
        
        self.final_psi = psi_0

        order_arr = np.zeros( (2*trotter_steps,) ) 

        for i in range(trotter_steps):
            # evolve system with a series of trotter local to non-local steps
            
            # start and end of evolution time of this step
            # print(f"At Trotter step {i}")
            order_arr[2*i] = 0
            order_arr[2*i+1] = 1
             
        self.H_TT = [ [self.HL_T[0][0], # HL0
                  lambda t, arg: self.HL_T[0][1](t, arg) * f_map(s(t), order_arr, id=0) ],
                 [self.HL_T[1][0], # HLF
                  lambda t, arg: self.HL_T[1][1](t, arg) * f_map(s(t), order_arr, id=0) ],
                 [self.HN_T[0][0], # HN0
                 lambda t, arg: self.HN_T[0][1](t, arg) * f_map(s(t), order_arr, id=1) ],
                 [self.HN_T[1][0], # HNF
                 lambda t, arg: self.HN_T[1][1](t, arg) * f_map(s(t), order_arr, id=1) ]
                ]

        # Trotterization Local step
        mesolve(self.H_TT, psi_0, self.time_list, 
        [], lambda t, psi: self.__process_rho_fast__(t, psi))
            
    def solve_main_eq_fast_padded(self, trotter_steps=10, trotter_interm=3, time_padding=0.01):
        # Solves same way, but is more careful with what time steps are picked
        # time_padding is the fraction of Dt=t_F/trotter_steps. 
        # dt = time_padding * Dt
        # inserts essential time padding around discontinuities, at
        # (i+time_padding)*Dt, (i+0.5 +- time_padding)*Dt, ((i+ 1 - time_padding)*Dt)
        # adds aditional, intermediate, time steps in the middle of the padding intervals, given by trotter_interm

        self.__idx = 0

        psi_0 = self.psi_0
        # total_time_segments = int(time_steps/trotter_steps) * trotter_steps

        def s(t):
            return self.schedule_func(t, self.sch_args)

        Ds = 1.0/(trotter_steps)
        
        def f_map(s, order, id):
            i = int(2*s*trotter_steps) # linear trotter mapping
            if i >= 2*trotter_steps:
                i = 2*trotter_steps-1
            elif i<0:
                i=0
            return (order[i] == id)*1.0
        
        # how many time steps in a continuous hamiltonian segment
        half_trotter_segment_length = (2+trotter_interm) 
        
        time_segments = int(2*half_trotter_segment_length*trotter_steps) + 2

        self.time_list = np.zeros( (time_segments,) )

        self.time_list[0] = 0
        self.time_list[-1] = self.sch_args['t max']

        Delta_t = self.sch_args['t max']/trotter_steps

        for i in range(trotter_steps):
            
            for j in [0, 1]:
                start_h1 = i*(2*half_trotter_segment_length)+1 + (trotter_interm+2)*j

                # initial padding
                self.time_list[start_h1] = (i+time_padding + 0.5*j)* Delta_t
                
                # intermidiate steps for the jth continuous segment
                for k in range(trotter_interm):
                    self.time_list[start_h1+k+1] = \
                    (i+time_padding + (k+1)*(0.5-2*time_padding)/(trotter_interm+1))* Delta_t
                
                self.time_list[start_h1+trotter_interm+1] = (i+0.5*j+time_padding+(0.5-2*time_padding) )*Delta_t


        self.evals_mat = np.zeros( (time_segments, self.total_eigen) )
        self.P_mat = np.zeros( (time_segments, self.total_eigen) )
        

        order_arr = np.zeros( (2*trotter_steps,) ) 

        for i in range(trotter_steps):
            # evolve system with a series of trotter local to non-local steps
            # give the order at which they occur
            order_arr[2*i] = 0
            order_arr[2*i+1] = 1
        

        self.H_TT = [ [self.HL_T[0][0], # HL0
                  lambda t, arg: self.HL_T[0][1](t, arg) * f_map(s(t), order_arr, id=0) ],
                 [self.HL_T[1][0], # HLF
                  lambda t, arg: self.HL_T[1][1](t, arg) * f_map(s(t), order_arr, id=0) ],
                 [self.HN_T[0][0], # HN0
                 lambda t, arg: self.HN_T[0][1](t, arg) * f_map(s(t), order_arr, id=1) ],
                 [self.HN_T[1][0], # HNF
                 lambda t, arg: self.HN_T[1][1](t, arg) * f_map(s(t), order_arr, id=1) ]
                ]

        self.final_psi = psi_0

        # Trotterization Local step
        mesolve(self.H_TT, psi_0, self.time_list, 
        [], lambda t, psi: self.__process_rho_fast__(t, psi))
            
    def print_combinatorial_H1_highest_P(self, num=4):
        len_qbits = len(self.qb_names)
        print( ("{} "*len_qbits + " |\tE\tP ").format(*self.qb_names))
        # evals, _ = H_1.eigenstates()
        # iterate over all combinations
        # data = {}
        comb_list = []
        data_arr = np.zeros((2**len_qbits, 3))
        H_F = self.H_T[1][0] 
        for i, ev_c in enumerate(itertools.product([0,1], repeat=len_qbits)):
            psi = tensor([basis(2,i) for i in ev_c])
            state_comb = ("{} "*len_qbits).format(*ev_c)
            energy = (psi.dag()*H_F*psi).tr()+0.0
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
            print( state_comb + f" |\t{energy:.3} \t{prob:.3}")
    
    def find_trotter_steps(self, trotter_start=5,rtol=1e-3, schedule_args=None, trotter_geom=1.5, try_n_steps=10, trotter_alg=0):
        
        self.run_SA_benchmark(schedule_args=schedule_args)

        # print("Solving first trotter")

        # solving initial trotter
        trotter_now = trotter_start
        self.solve_main_eq(int(trotter_now))

        final_gs_gap = np.abs(self.gs_SA - self.get_final_state_energy())
        idx = 0
        # stop_call = False
        while final_gs_gap > rtol:
            if idx > try_n_steps:
                # stop_call = True
                print(f"Algorithm wasn't able to find optimal number of Trotter steps. \nLast attempt: {int(trotter_now)}")
                break
            # trotter_now *= trotter_geom # geometric progression
            
            # The algebraic factor increases in a geometric progression
            trotter_alg *= trotter_geom
            
            trotter_now += trotter_alg # algebraic progression
            # print(f"Step: {idx}, TrotterSteps: {trotter_now:.1}")

            self.solve_main_eq(int(trotter_now))

            final_gs_gap = np.abs(self.gs_SA - self.get_final_state_energy())
            idx += 1 

        return trotter_now

    def find_trotter_steps_advanced(self, trotter_start=10,rtol=1e-3, schedule_args=None, exp_coef=2.5, exp_steps=5, refine_steps=5):
        
        # applies an exponential step, where the trotter_steps increase exponentially
        # then refines the trotter step to get the error close to rtol 
        # via a bisection algorithm


        self.run_SA_benchmark(schedule_args=schedule_args)

        # print("Solving first trotter")

        trotter_now = trotter_start
        self.solve_main_eq(int(trotter_now))

        final_gs_gap = np.abs(self.gs_SA - self.get_final_state_energy())
        idx = 0
        stop_call = False
        for i in range(exp_steps):
            trotter_now *= exp_coef
            self.solve_main_eq(int(trotter_now))

            final_gs_gap = np.abs(self.gs_SA - self.get_final_state_energy())
            
            if final_gs_gap < rtol:
                # found a number of trotter steps that overshot the treshold 
                break

        if final_gs_gap > rtol:
            print("Algorithm didn't find trotter steps, returning ... ")
            return trotter_now

        # The exponential trotter tries certainly overshot
        # so next we do aditional bisection steps to get better accuracy on the 
        # least trotter steps necessary to achieve error convergence 
        trotter0 = trotter_now/exp_coef
        trotter1 = trotter_now
        
        for i in range(refine_steps):
            trotter_now = (trotter1 + trotter0)/2

            if trotter_now-trotter0<=1:
                return trotter_now

            self.solve_main_eq(int(trotter_now))
            final_gs_gap = np.abs(self.gs_SA - self.get_final_state_energy())


            if final_gs_gap < rtol:
                # accuracy is too big, means we overshot
                # the upper limit is now the current position
                trotter1 = trotter_now 
            elif final_gs_gap > rtol:
                # accuracy is too low, we undershot
                # the lower limit is now the current position
                trotter0 = trotter_now
            else:
                return trotter_now

        
        trotter_now = (trotter1 + trotter0)/2

        return trotter_now


    def get_final_state_energy(self):
        H_F = self.H_T[1][0] 
        return (self.final_psi.dag() * H_F * self.final_psi).tr()+0.0 

    def get_state_error(self):
        H_F = self.H_T[1][0] 
        psi_gs = H_F.eigenstates()[1][0]
        psi_temp = psi_gs - self.final_psi
        return psi_temp.norm()

    def get_energy_error(self):
        H_F = self.H_T[1][0] 
        egs = H_F.eigenstates()
        psi_gs = egs[1][0]
        psi_temp = psi_gs - self.final_psi
        et = (self.final_psi.dag() * H_F * self.final_psi).tr()+0.0 - egs[0][0] # trotter energy error
        return et

    def show_eigenvalue_plot(self):

        # first draw thin lines outlining the energy spectrum
        for n in range(len(self.evals_mat[0,:])):
            ls,lw = ('b',1) if n == 0 else ('k', 0.25)
            plt.plot(self.time_list/max(self.time_list), self.evals_mat[:,n], ls, lw=lw)

        # second, draw line that encode the occupation probability of each state in 
        # its linewidth. thicker line => high occupation probability.
        for idx in range(len(self.time_list)-1):
            for n in range(len(self.P_mat[0,:])):
                lw = 0.5 + 4*self.P_mat[idx,n]    
                if lw > 0.55:
                    plt.plot(np.array([self.time_list[idx], self.time_list[idx+1]])/self.sch_args['t max'], 
                        np.array([self.evals_mat[idx,n], self.evals_mat[idx+1,n]]), 'r', linewidth=lw)    
                
        plt.xlabel(r'$t$', fontsize=25)
        plt.ylabel(r'$\varepsilon_i$', fontsize=25)
        plt.title("Energyspectrum of a chain of %d spins.\n " % (len(self.qb_names))
                        + "The occupation probabilities are encoded in the red line widths.")
        
        plt.show()




def build_hamiltonian(qbits, linear_args={}, quadratic_args={}):
    sx_i, sy_i, sz_i = spin_tensors(len(qbits))

    pauli = {'I':tensor([si]*len(qbits)),
    'x': sx_i, 'y':sy_i, 'z':sz_i}

    qb_idx = {}
    for i,qb in enumerate(qbits):
        qb_idx[qb] = i

    H = 0*sx_i[0] # 0  matrix
    
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
    




