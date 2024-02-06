import save_in_file as sif
from problems.evaluation_functions import h2, vqls_1, sudoku2x2, vqls_0

eval_func = [h2]
N_ITER = 10

BUDGET = [1000, 2000, 5000, 10000, 50000, 100000, 200000]

BF = [False]
ROTYPE = 'classic'
ROSTEPS = [0]
p = {'a': 50, 'd': 10, 's': 20, 'c': 20}
MAX_DEPTH = 20      # Chosen by the hardware

plot = [True, True]
run = True

for r in ROSTEPS:
    for f in eval_func:
        for m in BF:
            for b in BUDGET:
                for i in range(N_ITER):
                    if run:
                        if f == sudoku2x2:
                            r = 0
                            sif.data(evaluation_function=f, variable_qubits=5, ancilla_qubits=0, gate_set='continuous',
                                     budget=b, branches=m, max_depth=MAX_DEPTH, roll_out_steps=r, iteration=i, choices=p,
                                     verbose=True)
                        else:
                            sif.data(evaluation_function=f, variable_qubits=4, ancilla_qubits=0, gate_set='continuous',
                                     rollout_type=ROTYPE, budget=b, branches=m, roll_out_steps=r, iteration=i, max_depth=MAX_DEPTH,
                                     choices=p, verbose=True)
                        print('Iteration ', i, ' has been saved')
                if plot[0]:
                    sif.plot_cost(evaluation_function=f, branches=m, budget=b, roll_out_steps=r,
                                  rollout_type=ROTYPE, n_iter=N_ITER)
            if plot[1]:
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE,
                            n_iter=N_ITER)
