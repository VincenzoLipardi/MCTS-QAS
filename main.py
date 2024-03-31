import save_in_file as sif
from save_in_file import check_file_exist
import evaluation_functions as evf


eval_func = [evf.fidelity_4_5_easy, evf.fidelity_4_5_hard, evf.fidelity_4_10_easy, evf.fidelity_4_10_hard,
             evf.fidelity_4_15_easy, evf.fidelity_4_15_hard, evf.fidelity_4_20_easy, evf.fidelity_4_20_hard,
             evf.fidelity_4_30_easy, evf.fidelity_4_30_hard,
            evf.fidelity_6_5_easy, evf.fidelity_6_5_hard, evf.fidelity_6_10_easy, evf.fidelity_6_10_hard,
             evf.fidelity_6_15_easy, evf.fidelity_6_15_hard, evf.fidelity_6_20_easy, evf.fidelity_6_20_hard,
             evf.fidelity_6_30_easy, evf.fidelity_6_30_hard,
            evf.fidelity_8_5_easy, evf.fidelity_8_5_hard, evf.fidelity_8_10_easy, evf.fidelity_8_10_hard,
             evf.fidelity_8_15_easy, evf.fidelity_8_15_hard, evf.fidelity_8_20_easy, evf.fidelity_8_20_hard,
             ]
N_ITER = 10
BUDGET = [1000, 2000, 5000, 10000, 50000, 100000]#, 200000, 300000]#, 400000, 600000]
#BUDGET = [1000]
BF = [False]

ROTYPE = 'classic'
ROSTEPS = [1,2]
p = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
EPS = None
STOP = False
MAX_DEPTH = 20      # Chosen by the hardware
qubits = {'h2': 4, 'lih': 10, 'h2o': 8, 'vqls_1': 4, 'sudoku2x2': 5, 'h2o_full': 8}

# Cost plot: convergence via mcts, boxplot of the best via mcts, boxplot after classical optimizer, convergence via classical optimizer
plot = [True, True, False, False]
run = False
apply_gradient_descent = [True, False]

# Run Experiments
for r in ROSTEPS:
    for f in eval_func:
        for m in BF:
            for b in BUDGET:
                for i in range(N_ITER):
                    if run:
                        if f.__name__[0] == 'f':
                            n_qubits = int(f.__name__[9])
                        else:
                            n_qubits = qubits[f.__name__]
                        sif.run_and_savepkl(evaluation_function=f, variable_qubits=n_qubits, ancilla_qubits=0, gate_set='continuous',
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
