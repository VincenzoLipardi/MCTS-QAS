import mcts
import pandas as pd
import math
import os.path
import numpy as np
from structure import Circuit
import matplotlib.pyplot as plt
from evaluation_functions import h2, vqls_1, sudoku2x2, h2o, lih, fidelity_easy, fidelity_hard, fidelity_5, fidelity_10, fidelity_15, fidelity_20, h2o_full
from problems.oracles.grover.grover import grover_algo



def get_filename(evaluation_function, budget, branches, iteration, epsilon, stop_deterministic, rollout_type, image, gradient=False, gate_set='continuous', roll_out_steps=None):
    """ it creates the string of the file name that have to be saved or read"""

    ro = 'rollout_' + rollout_type + '/'
    ros = '_rsteps_' + str(roll_out_steps)
    stop = ''
    if stop_deterministic:
        stop = '_stop'
    if isinstance(branches, bool):
        if branches:
            branch = "dpw"
        else:
            branch = "pw"
    elif isinstance(branches, int):
        branch = 'bf_' + str(branches)
    else:
        raise TypeError
    grad, eps = '', ''
    if epsilon is not None:
        eps = '_eps_'+str(epsilon)
    if gradient:
        grad = '_gd'
    if image:
        filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '/' + ro+'images/' + branch + eps + ros + grad + stop
    else:
        filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '/' + ro + branch + eps + '_budget_' + str(budget) + ros + '_run_' + str(iteration)+grad+stop
    return filename


def run_and_savepkl(evaluation_function, variable_qubits, ancilla_qubits, budget, max_depth, iteration, branches, choices, epsilon, stop_deterministic, gate_set='continuous', rollout_type="classic", roll_out_steps=None, verbose=True):
    """
    It runs the mcts on the indicated problem and saves the result (the best path) in a .pkl file
    :param stop_deterministic: If True each node expanded will be also taken into account as terminal node.
    :param epsilon: float. probability to go random
    :param choices: dict. Probability distribution over all the possible class actions
    :param evaluation_function: func. It defines the problem, then the reward function for the mcts agent
    :param variable_qubits:int.  Number of qubits required for the problem
    :param ancilla_qubits: int. Number of ancilla qubits required, as in the case of the oracle problem (Hyperparameter)
    :param gate_set: str. Use 'continuous' (CX+single-qubit rotations) or 'discrete'( Clifford generator + t).
    :param budget: int. Resources allocated for the mcts search. MCTS iterations (or simulations)
    :param max_depth: int. Max depth of the quantum circuit
    :param iteration: int. Number of the independent run.
    :param branches: bool or int. If True progressive widening implemented. If int the number of maximum branches is fixed.
    :param rollout_type: str. classic evaluates the final quantum circuit got after rollout. rollout_max takes the best reward get from all the states in the rollout path
    :param roll_out_steps: int Number of moves for the rollout.
    :param verbose: bool. True if you want to print out the algorithm results online.
    """
    if isinstance(choices, dict):
        pass
    elif isinstance(choices, list):
        choices = {'a': choices[0], 'd': choices[1], 's': choices[2], 'c': choices[3], 'p': choices[4]}
    else:
        raise TypeError
    # Define the root note
    root = mcts.Node(Circuit(variable_qubits=variable_qubits, ancilla_qubits=ancilla_qubits), max_depth=max_depth)
    # Run the mcts algorithm
    final_state = mcts.mcts(root, budget=budget, branches=branches, evaluation_function=evaluation_function, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                            choices=choices, epsilon=epsilon, stop_deterministic=stop_deterministic, verbose=verbose)
    # Create the name of the pickle file where the results will be saved in
    filename = get_filename(evaluation_function, budget=budget, branches=branches, iteration=iteration, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps, epsilon=epsilon, stop_deterministic=stop_deterministic, image=False)
    df = pd.DataFrame(final_state)
    df.to_pickle(os.path.join(filename + '.pkl'))

    print("files saved in experiments/", evaluation_function.__name__, 'as ', filename)


def add_columns(evaluation_function, budget, n_iter, branches, epsilon, stop_deterministic, roll_out_steps, rollout_type, gradient, gate_set='continuous'):
    """Adds the column of the cost function during the search, and apply the gradient descent on the best circuit and save it the column Adam"""

    for i in range(n_iter):
        filename = get_filename(evaluation_function, budget, branches, iteration=i, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                                epsilon=epsilon, stop_deterministic=stop_deterministic, image=False)
        qc_path = get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter)[0]
        df = pd.read_pickle(filename + '.pkl')
        # Get last circuit in the tree path
        quantum_circuit_last = qc_path[i][-1]
        # Apply gradient on teh last circuit and create a column to save it
        final_result = evaluation_function(quantum_circuit_last, ansatz='', cost=False, gradient=True)
        column_adam = [[None]]*df.shape[0]
        column_adam[-1] = final_result

        # Create column of the cost values along the tree path
        column_cost = list(map(lambda x: evaluation_function(x, cost=True), qc_path[i]))
        # Add the columns to the pickle file
        df['cost'] = column_cost

        # Apply gradient on the best circuit if the best is not the last in the path
        if gradient:
            index = column_cost.index(min(column_cost))
            if index != len(qc_path[i]):
                quantum_circuit_best = qc_path[i][index]
                best_result = evaluation_function(quantum_circuit_best, ansatz='', cost=False, gradient=True)
                column_adam[index] = best_result
            df["Adam"] = column_adam

        df.to_pickle(os.path.join(filename + '.pkl'))
        print('Columns added to: ', filename)



def get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter=10):
    """ It opens the .pkl files and returns quantum circuits along the best path for all the independent run
    :return: four list of lists
    """
    qc_along_path = []
    children, visits, value = [], [], []
    for i in range(n_iter):
        filename = get_filename(evaluation_function, budget, branches, iteration=i, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, roll_out_steps=roll_out_steps, image=False)

        if os.path.isfile(filename+'.pkl'):
            df = pd.read_pickle(filename+'.pkl')
            qc_along_path.append([circuit for circuit in df['qc']])
            children.append(df['children'].tolist())
            value.append(df['value'].tolist())
            visits. append(df['visits'].tolist())
        else:
            return FileNotFoundError
    return qc_along_path, children, visits, value


def best_in_path(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter):
    """ It returns the list of the costs of best solution for all the independent runs right after mcts"""
    cost_overall, best_index = [], []
    for i in range(n_iter):
        filename = get_filename(evaluation_function, budget, branches, iteration=i, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, roll_out_steps=roll_out_steps, image=False)
        df = pd.read_pickle(filename + '.pkl')
        cost = df['cost'].tolist()
        if isinstance(cost[0], list):
            best = min(cost)[0]
        else:
            best = min(cost)
        cost_overall.append(best)
        best_index.append(cost.index(best))
    return cost_overall, best_index


def get_best_overall(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter):
    """Given an experiment with fixed hyperparameters, it returns the index of the best run and its convergence via classical optimizer"""
    best = []
    for i in range(n_iter):
        filename = get_filename(evaluation_function=evaluation_function, budget=budget, iteration=i, branches=branches, epsilon=epsilon, stop_deterministic=stop_deterministic, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                                image=False)
        df = pd.read_pickle(filename + '.pkl')
        column = df['Adam']
        final = [column[j][-1] for j in range(df.shape[0]) if column[j][0] is not None]
        best.append(min(k for k in final if not math.isnan(k)))
    best_run = best.index(min(best))
    return best, best_run


# PLOTS
def plot_cost(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, benchmark=None):
    """It saves the convergence plot of the cost vs tree depth"""

    plt.xlabel('Tree Depth')
    plt.ylabel('Cost')
    max_tree_depth = 0
    for i in range(n_iter):
        filename = get_filename(evaluation_function, budget, branches, i, epsilon, stop_deterministic, rollout_type, roll_out_steps=roll_out_steps, image=False)
        df = pd.read_pickle(filename+'.pkl')
        cost = df['cost']
        tree_depth = len(cost)
        if tree_depth > max_tree_depth:
            max_tree_depth = tree_depth
        plt.plot(list(range(len(cost))), cost, marker='o', linestyle='-', label=str(i+1))

    # Set x-ticks
    indices = list(range(max_tree_depth + 2))
    if max_tree_depth > 20:
        indices = indices[::2]
    plt.xticks(indices)
    # Get benchmark value for the problem
    if evaluation_function == h2 or evaluation_function == h2o or evaluation_function == lih:
        benchmark = get_benchmark(evaluation_function)
        plt.ylabel('Energy (Ha)')
    if evaluation_function == h2:
        plt.yticks(np.arange(-1.2, 0.1, 0.1))

    if benchmark is not None:
        if isinstance(benchmark, list) or isinstance(benchmark, tuple):
            plt.axhline(y=benchmark[0], color='r', linestyle='--', label=f'bench_SCF({round(benchmark[0], 3)})')
            plt.axhline(y=benchmark[1], color='g', linestyle='--', label=f'bench_FCI({round(benchmark[1], 3)})')

        else:
            plt.axhline(y=benchmark, color='r', linestyle='--', label=f'ADAPT-VQE({round(benchmark, 3)})')
    filename = get_filename(evaluation_function=evaluation_function, branches=branches, image=True, roll_out_steps=roll_out_steps, rollout_type=rollout_type, iteration=0, budget=budget, epsilon=epsilon, stop_deterministic=stop_deterministic) + '_budget_'+str(budget)
    plt.legend(loc='best')
    plt.title(evaluation_function.__name__ + ' - Budget  '+str(budget))

    plt.savefig(filename + '_cost_along_path.png')
    print('Plot of the cost along the path saved in image', filename)
    plt.clf()


def boxplot(evaluation_function, branches, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, gradient):
    """ Save a boxplot image, with the stats on the n_iter independent runs vs the budget of mcts"""
    solutions = []
    BUDGET = [1000, 2000, 5000, 10000, 50000, 100000, 200000, 300000]#, 400000, 600000]

    for budget in BUDGET:
        if not check_file_exist(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter):
            index = BUDGET.index(budget)
            BUDGET = BUDGET[:index]
            break

        if gradient:
            sol = get_best_overall(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter)
            if isinstance(sol, tuple):
                sol = sol[0]
            solutions.append(sol)
        else:
            sol = best_in_path(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter)[0]
            solutions.append(sol)

    # Plotting
    budget_effective = [str(b) for b in BUDGET]
    print(budget_effective)

    plt.boxplot(solutions, patch_artist=True, labels=budget_effective, meanline=True, showmeans=True, showfliers=True)
    print([min(a) for a in solutions])
    benchmark = get_benchmark(evaluation_function)
    if benchmark is not None:
        if evaluation_function == h2 or evaluation_function == h2o or evaluation_function ==h2o_full:
            plt.ylabel('Energy (Ha)')
            benchmark = round(benchmark[1], 3)
            label = 'bench_FCI'
        elif evaluation_function == lih:
            plt.ylabel('Energy (Ha)')
            benchmark = round(benchmark, 3)
            label = 'ADAPT-VQE'
        elif evaluation_function == sudoku2x2:
            plt.ylabel('Counts')
            label = 'exact_oracle'
        else:
            plt.ylabel('Cost')
            label = 'benchmark'
        plt.axhline(y=benchmark, color='r', linestyle='--', label=label)

    filename = get_filename(evaluation_function=evaluation_function, branches=branches, image=True, roll_out_steps=roll_out_steps, rollout_type=rollout_type, iteration=0, epsilon=epsilon, stop_deterministic=stop_deterministic,  gradient=gradient, budget=0)
    plt.title(evaluation_function.__name__)
    plt.xlabel('MCTS Simulations')
    plt.legend()
    plt.savefig(filename + '_boxplot.png')

    plt.clf()
    print('boxplot image saved in ', filename)
    return solutions


def plot_gradient_descent(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter):

    plt.xlabel('Steps')
    plt.ylabel('Cost')
    for b in budget:
        index = get_best_overall(evaluation_function, branches, b, roll_out_steps, rollout_type, epsilon,
                                 stop_deterministic, n_iter)[1]
        filename = get_filename(evaluation_function=evaluation_function, budget=b, iteration=index, branches=branches,
                                epsilon=epsilon, stop_deterministic=stop_deterministic, rollout_type=rollout_type,
                                roll_out_steps=roll_out_steps,
                                image=False)
        df = pd.read_pickle(filename + '.pkl')
        filtered_column = df[df['Adam'].apply(lambda x: x != [None])]['Adam']

        gd_values = filtered_column.tolist()[0]
        plt.plot(range(len(gd_values)), gd_values, marker='o', linestyle='-', label=str(b))
    plt.title(evaluation_function.__name__ + ' - Adam Optimizer')

    benchmark_value = get_benchmark(evaluation_function)
    if evaluation_function == h2 or evaluation_function == h2o or evaluation_function == lih:
        plt.ylabel('Energy (Ha)')

        if isinstance(benchmark_value, list) or isinstance(benchmark_value, tuple):
            plt.axhline(y=benchmark_value[0], color='r', linestyle='--',
                        label=f'bench_SCF({round(benchmark_value[0], 3)})')
            plt.axhline(y=benchmark_value[1], color='g', linestyle='--',
                        label=f'bench_FCI({round(benchmark_value[1], 3)})')

        else:
            plt.axhline(y=benchmark_value, color='r', linestyle='--', label=f'ADAPT-VQE({round(benchmark_value, 3)})')
    else:
        plt.axhline(y=benchmark_value, color='r', linestyle='--', label=f'bench({benchmark_value})')
    plt.legend()


    filename = get_filename(evaluation_function=evaluation_function, budget=budget, iteration=0, branches=branches,
                            epsilon=epsilon, stop_deterministic=stop_deterministic, rollout_type=rollout_type,
                            roll_out_steps=roll_out_steps, image=True)
    plt.savefig(filename+'_gd.png')
    plt.clf()
    return print('Gradient descent image saved in ', filename)


# Utils
def check_file_exist(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter=10, gate_set='continuous'):
    """ :return: bool. True if all the files are stored, false otherwise"""
    check = True
    for i in range(n_iter):
        filename = get_filename(evaluation_function, budget, branches, iteration=i, gate_set=gate_set, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, roll_out_steps=roll_out_steps, image=False)
        if not os.path.isfile(filename+'.pkl'):
            check = False
    return check


def get_benchmark(evaluation_function):
    """ It returns the classical benchmark value of the problems in input"""
    if evaluation_function == h2:
        # List
        # sol_scf = H2().benchmark()
        sol_scf = -1.115
        sol_fci = -1.136189454088     # Full configuration Interaction
        return sol_scf, sol_fci
    elif evaluation_function == sudoku2x2:
        counts_exact = grover_algo(oracle='exact', iterations=2, ancilla=1)

        if '1001' not in counts_exact:
            counts_exact['1001'] = 0
        elif '0110' not in counts_exact:
            counts_exact['0110'] = 0
        else:
            pass
        right_counts = counts_exact['1001'] + counts_exact['0110']
        return right_counts
    elif evaluation_function == lih:
        return -7.972
    elif evaluation_function == h2o or evaluation_function == h2o_full:
        return -75.16, -75.49
    elif evaluation_function == fidelity_easy or evaluation_function == fidelity_hard or \
        evaluation_function == fidelity_5 or evaluation_function == fidelity_10 or fidelity_15:
        return .0

