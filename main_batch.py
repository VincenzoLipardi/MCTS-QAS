import save_in_file as sif
from save_in_file import check_file_exist
import evaluation_functions as evf
from multiprocessing import Pool

N_ITER = 10
ROTYPE = 'classic'
UCB = [0.7]
p = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
EPS = None
STOP = False
MAX_DEPTH = 20      # Chosen by the hardware
CRITERIA = ['average_value', 'value']
N_QUBITS = {'h2': 4, 'lih': 10, 'h2o': 8, 'vqls_1': 4, 'sudoku': 5, 'sudoku2x2': 5, 'h2o_full': 8}


apply_gradient_descent = False
PLOT = [True, True, True, True]
plot_oracle = False
process_pool = True

#BUDGET = [1000, 2000, 5000, 10000, 50000, 100000]#, 200000, 300000
BUDGET = [100000]

start = 'run'
roll = [0, 1, 2]
gates = [5, 10, 15, 20, 30]
iterations = list(range(10))
functions = [evf.h2, evf.h2o, evf.lih, evf.vqls_1,
evf.fidelity_4_5_easy, evf.fidelity_4_10_easy, evf.fidelity_4_15_easy, evf.fidelity_4_20_easy, evf.fidelity_4_30_easy,
evf.fidelity_4_5_hard, evf.fidelity_4_10_hard, evf.fidelity_4_15_hard, evf.fidelity_4_20_hard, evf.fidelity_4_30_hard,
evf.fidelity_6_5_easy, evf.fidelity_6_10_easy, evf.fidelity_6_15_easy, evf.fidelity_6_20_easy, evf.fidelity_6_30_easy,
evf.fidelity_6_5_hard, evf.fidelity_6_10_hard, evf.fidelity_6_15_hard, evf.fidelity_6_20_hard,evf.fidelity_6_30_hard,
evf.fidelity_8_5_hard, evf.fidelity_8_10_hard, evf.fidelity_8_15_hard, evf.fidelity_8_20_hard, evf.fidelity_8_30_hard,
evf.fidelity_8_5_easy, evf.fidelity_8_10_easy, evf.fidelity_8_15_easy, evf.fidelity_8_20_easy,  evf.fidelity_8_30_easy,]


def main(function, budget, rollout_steps, iter, ucb, criteria):
    if function.__name__[0] == 'f':
        n_qubits = int(function.__name__[9])
    else:
        n_qubits = N_QUBITS[function.__name__]

    sif.run_and_savepkl(evaluation_function=function, criteria=criteria, variable_qubits=n_qubits, ancilla_qubits=0, gate_set='continuous', rollout_type=ROTYPE, budget=budget, branches=False, roll_out_steps=rollout_steps, iteration=iter, max_depth=MAX_DEPTH, choices=p, epsilon=EPS, stop_deterministic=STOP, ucb=ucb, verbose=False)


def parameter_opt(function, budget, rollout_steps, ucb, criteria):

    sif.add_columns(evaluation_function=function, criteria=criteria, budget=budget, n_iter=N_ITER, branches=False, epsilon=EPS, roll_out_steps=rollout_steps, rollout_type=ROTYPE, stop_deterministic=STOP, gradient=apply_gradient_descent, ucb=ucb)


def plot(function, roll_out_steps):

    if PLOT[1]:
        # Boxplot with the results of the best circuits at different budget on n_iter independent run
        sif.boxplot(evaluation_function=function, budget=BUDGET, criteria=CRITERIA, branches=False, roll_out_steps=roll_out_steps,  rollout_type=ROTYPE, epsilon=EPS, n_iter=N_ITER, gradient=False, stop_deterministic=STOP, ucb=UCB)
    if PLOT[2]:
        # Boxplot with the results after the fine-tuning at different budget on n_iter independent run
        sif.boxplot(evaluation_function=function, budget=BUDGET, criteria=CRITERIA, branches=False, roll_out_steps=roll_out_steps,  rollout_type=ROTYPE, epsilon=EPS, n_iter=N_ITER, gradient=True, stop_deterministic=STOP, ucb=UCB)

    if PLOT[0]:
        # plot the cost along the mcts path
        for budget in BUDGET:
            sif.plot_cost(evaluation_function=function, criteria=CRITERIA, branches=False, budget=budget, roll_out_steps=roll_out_steps, rollout_type=ROTYPE, n_iter=N_ITER, epsilon=EPS, stop_deterministic=STOP, ucb=UCB)

    if PLOT[3]:
        # Plot of the gradient descent on the best run for different budgets
        sif.plot_gradient_descent(evaluation_function=function, criteria=CRITERIA, branches=False, budget=BUDGET, roll_out_steps=roll_out_steps, rollout_type=ROTYPE, n_iter=N_ITER, epsilon=EPS, stop_deterministic=STOP, ucb=UCB)


"""
evf.h2, evf.h2o, evf.lih, evf.vqls_1, evf.sudoku,
evf.fidelity_4_5_easy, evf.fidelity_4_10_easy, evf.fidelity_4_15_easy, evf.fidelity_4_20_easy, evf.fidelity_4_30_easy,
evf.fidelity_4_5_hard, evf.fidelity_4_10_hard, evf.fidelity_4_15_hard, evf.fidelity_4_20_hard, evf.fidelity_4_30_hard,
evf.fidelity_6_5_easy, evf.fidelity_6_10_easy, evf.fidelity_6_15_easy, evf.fidelity_6_20_easy, evf.fidelity_6_30_easy,
evf.fidelity_6_5_hard, evf.fidelity_6_10_hard, evf.fidelity_6_15_hard, evf.fidelity_6_20_hard, evf.fidelity_6_30_hard,
evf.fidelity_8_5_hard, evf.fidelity_8_10_hard, evf.fidelity_8_15_hard, evf.fidelity_8_20_hard, evf.fidelity_8_30_hard,
evf.fidelity_8_5_easy, evf.fidelity_8_10_easy, evf.fidelity_8_15_easy, evf.fidelity_8_20_easy,  evf.fidelity_8_30_easy,
 
"""


combinations = []
if start == 'run':
    for k in functions:
        for b in BUDGET:
            for i in roll:
                for j in iterations:
                    for u in UCB:
                        for c in CRITERIA:
                            combinations.append((k, b, i, j, u, c))

elif start == 'gd':
    for k in functions:
        for b in BUDGET:
            for i in roll:
                for u in UCB:
                    for c in CRITERIA:
                        combinations.append((k, b, i, u, c))

elif start == 'plot':
    for k in functions:
        for i in roll:
            combinations.append((k, i))


if process_pool:
    with Pool() as processing_pool:
        if start == 'run':
            processing_pool.starmap(main, combinations)
        elif start == 'gd':
            processing_pool.starmap(parameter_opt, combinations)
        elif start == 'plot':
            processing_pool.starmap(plot, combinations)
if plot_oracle:
    for s in roll:
        for u in UCB:
            for c in CRITERIA:
                for a in [0.01, 0.05, 0.1, 0.2]:
                    for q in [4, 6]:
                        sif.colorplot_oracle(criteria=c, qubits=q, gates=gates, accuracy=a, simulations=100000, steps=s, ucb=u)



