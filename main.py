import save_in_file as sif
from save_in_file import check_file_exist
from problems.evaluation_functions import h2, vqls_1, sudoku2x2, vqls_0

eval_func = [h2]
N_ITER = 10

BUDGET = [1000, 2000, 5000, 10000, 50000, 100000, 200000]

BF = [False]
ROTYPE = 'classic'
ROSTEPS = [0]
p = {'a': 50, 'd': 10, 's': 20, 'c': 20}
EPS = 0.2
STOP = True
MAX_DEPTH = 20      # Chosen by the hardware

plot = [False, True, False]
run = False
add_column = False

# Run and Cost plots
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
                                     epsilon=EPS, stop_deterministic=STOP, verbose=True)
                        else:
                            sif.data(evaluation_function=f, variable_qubits=4, ancilla_qubits=0, gate_set='continuous',
                                     rollout_type=ROTYPE, budget=b, branches=m, roll_out_steps=r, iteration=i, max_depth=MAX_DEPTH,
                                     choices=p, epsilon=EPS, stop_deterministic=STOP, verbose=True)
                        print('Iteration ', i, ' has been saved')

                if plot[0]:
                    sif.plot_cost(evaluation_function=f, branches=m, budget=b, roll_out_steps=r,
                                  rollout_type=ROTYPE, n_iter=N_ITER, epsilon=EPS)

# Boxplots
for r in ROSTEPS:
    for f in eval_func:
        for m in BF:
            for b in BUDGET:
                if add_column:
                    if check_file_exist(evaluation_function=f, budget=b, n_iter=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE, gradient=False):

                        sif.add_gradient_descent_column(evaluation_function=f, budget=b, iteration=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE)

            if plot[1]:
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, best=False, gradient=False)
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, best=True, gradient=False)
            if plot[2]:
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, best=False, gradient=True)
