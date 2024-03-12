import save_in_file as sif
from save_in_file import check_file_exist
from evaluation_functions import h2, lih, h2o, vqls_0, vqls_1, sudoku2x2, fidelity_easy, fidelity_hard, fidelity_5, h2o_full

eval_func = [h2o]
N_ITER = 10
BUDGET = [1000, 2000, 5000, 10000, 50000, 100000, 200000, 300000]

BF = [False]

ROTYPE = 'classic'
ROSTEPS = [1]
p = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
EPS = None
STOP = False
MAX_DEPTH = 20      # Chosen by the hardware
qubits = {'h2': 4, 'lih': 10, 'h2o': 8, 'vqls_1': 4, 'sudoku2x2': 5, 'fidelity_easy': 4, 'fidelity_hard': 4, 'fidelity_5': 4,  'h2o_full': 8}

# Cost plot: convergence via mcts, boxplot of the best via mcts, boxplot after classical optimizer, convergence via classical optimizer
plot = [False, False, True, False]
run = False
apply_gradient_descent = [False, False]

# Run Experiments
for r in ROSTEPS:
    for f in eval_func:
        for m in BF:
            for b in BUDGET:
                for i in range(N_ITER):
                    if run:
                        sif.run_and_savepkl(evaluation_function=f, variable_qubits=qubits[f.__name__], ancilla_qubits=0, gate_set='continuous',
                                            rollout_type=ROTYPE, budget=b, branches=m, roll_out_steps=r, iteration=i, max_depth=MAX_DEPTH,
                                            choices=p, epsilon=EPS, stop_deterministic=STOP, verbose=True)

# Plots
for r in ROSTEPS:
    for f in eval_func:
        for m in BF:
            for b in BUDGET:
                if apply_gradient_descent[0]:
                    if check_file_exist(evaluation_function=f, budget=b, n_iter=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE, stop_deterministic=STOP):
                        # Add columns of the cost along the mcts path and the gradient descent on the best quantum circuit found
                        sif.add_columns(evaluation_function=f, budget=b, n_iter=N_ITER, branches=False, epsilon=EPS, roll_out_steps=r, rollout_type=ROTYPE, stop_deterministic=STOP, gradient=apply_gradient_descent[1])

                if plot[0]:
                    # plot the cost along the mcts path
                    sif.plot_cost(evaluation_function=f, branches=m, budget=b, roll_out_steps=r, rollout_type=ROTYPE, n_iter=N_ITER, epsilon=EPS, stop_deterministic=STOP)
            if plot[1]:
                # Boxplot with the results of the best circuits at different budget on n_iter independent run
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, gradient=False, stop_deterministic=STOP)
            if plot[2]:
                # Boxplot with the results after the fine-tuning at different budget on n_iter independent run
                sif.boxplot(evaluation_function=f, branches=m, roll_out_steps=r, rollout_type=ROTYPE, epsilon=EPS,
                            n_iter=N_ITER, gradient=True, stop_deterministic=STOP)
            if plot[3]:
                # Plot of the gradient descent on the best run for different budgets
                sif.plot_gradient_descent(evaluation_function=f, branches=m, budget=BUDGET, roll_out_steps=r,
                                          rollout_type=ROTYPE, n_iter=N_ITER, epsilon=EPS, stop_deterministic=STOP)
