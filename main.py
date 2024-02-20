import save_in_file as sif
from save_in_file import check_file_exist
from problems.evaluation_functions import h2, lih, h2o, vqls_0, vqls_1, sudoku2x2, qc_regeneration

eval_func = [lih]
N_ITER = 10


BUDGET = [1000, 2000, 5000, 10000, 50000, 100000]

BF = [False]
ROTYPE = 'classic'
ROSTEPS = [1]
p = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
EPS = None
STOP = False
MAX_DEPTH = 20      # Chosen by the hardware

plot = [False, False, True]
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
                        elif f == lih:
                            sif.data(evaluation_function=f, variable_qubits=10, ancilla_qubits=0, gate_set='continuous',
                                     rollout_type=ROTYPE, budget=b, branches=m, roll_out_steps=r, iteration=i, max_depth=MAX_DEPTH,
                                     choices=p, epsilon=EPS, stop_deterministic=STOP, verbose=True)
                        elif f == h2o:
                            sif.data(evaluation_function=f, variable_qubits=8, ancilla_qubits=0, gate_set='continuous',
                                     rollout_type=ROTYPE, budget=b, branches=m, roll_out_steps=r, iteration=i, max_depth=MAX_DEPTH,
                                     choices=p, epsilon=EPS, stop_deterministic=STOP, verbose=True)
                        else:
                            sif.data(evaluation_function=f, variable_qubits=4, ancilla_qubits=0, gate_set='continuous',
                                     rollout_type=ROTYPE, budget=b, branches=m, roll_out_steps=r, iteration=i, max_depth=MAX_DEPTH,
                                     choices=p, epsilon=EPS, stop_deterministic=STOP, verbose=True)
                        print('Iteration ', i, ' has been saved')

                if plot[0]:
                    sif.plot_cost(evaluation_function=f, branches=m, budget=b, roll_out_steps=r,
                                  rollout_type=ROTYPE, n_iter=N_ITER, epsilon=EPS, stop_deterministic=STOP)

# Boxplots
for r in ROSTEPS:
    for f in eval_func:
        for m in BF:
            for b in BUDGET:
                if add_column:
                    if check_file_exist(evaluation_function=f, budget=b, n_iter=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE, gradient=False, stop_deterministic=STOP):

                        sif.add_gradient_descent_column(evaluation_function=f, budget=b, iteration=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE, stop_deterministic=STOP)

            if plot[1]:
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, best=False, gradient=False, stop_deterministic=STOP)
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, best=True, gradient=False, stop_deterministic=STOP)
            if plot[2]:
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, best=False, gradient=True, stop_deterministic=STOP)
