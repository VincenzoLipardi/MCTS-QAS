import mcts
import pandas as pd
import math
import os.path
import numpy as np
from structure import Circuit
import matplotlib.pyplot as plt
from problems.evaluation_functions import h2, vqls_1, sudoku2x2, h2o, lih, qc_regeneration
from problems.oracles.grover.grover import grover_algo
from qiskit.visualization import plot_histogram
from problems.vqe import H2



def get_filename(evaluation_function, budget, branches, iteration, epsilon, stop_deterministic, gradient, image, gate_set='continuous', rollout_type="classic", roll_out_steps=None):
    """ it creates the string of the file name that have to be saved or read"""

    ro = ''
    ros = ''
    stop = ''
    if stop_deterministic:
        stop = '_stop'
    if evaluation_function != sudoku2x2:
        ro = 'rollout_' + rollout_type + '/'
        ros = '_rsteps_' + str(roll_out_steps)
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
        grad = '_comp'

    if image:
        filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '/' + ro+'images/' + branch + eps + ros + grad + stop
    else:
        filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '/' + ro + branch + eps + '_budget_' + str(budget) + ros + '_run_' + str(iteration)+grad+stop
    return filename


def data(evaluation_function, variable_qubits, ancilla_qubits, budget, max_depth, iteration, branches, choices, epsilon, stop_deterministic, gate_set='continuous', rollout_type="classic", roll_out_steps=None, verbose=True):
    """
    It runs the mcts on the indicated problem and saves the result (the best path) in a .pkl file
    :param stop_deterministic:
    :param epsilon: float. probability to go random
    :param choices: dict. Probability over all the possible choices
    :param evaluation_function: func. It defines the problem, then the reward function for the mcts agent
    :param variable_qubits:int.  Number of qubits required for the problem
    :param ancilla_qubits: int. Number of ancilla qubits required, as in the case of the oracle problem (Hyperparameter)
    :param gate_set: str.
    :param budget: int. resources allocated for the mcts search. MCTS iterations
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

    root = mcts.Node(Circuit(variable_qubits=variable_qubits, ancilla_qubits=ancilla_qubits), max_depth=max_depth)
    final_state = mcts.mcts(root, budget=budget, branches=branches, evaluation_function=evaluation_function, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                            choices=choices, epsilon=epsilon, stop_deterministic=stop_deterministic, verbose=verbose)

    filename = get_filename(evaluation_function, budget=budget, branches=branches, iteration=iteration, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps, epsilon=epsilon, stop_deterministic=stop_deterministic, gradient=False, image=False)

    df = pd.DataFrame(final_state)
    df.to_pickle(os.path.join(filename + '.pkl'))
    return print("files saved in experiments/", evaluation_function.__name__, 'as ', filename)


def get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter=10, gate_set='continuous'):
    """ It opens the .pkl files and returns quantum circuits along the best path """
    qc_along_path = []
    children, visits, value = [], [], []
    for i in range(n_iter):
        filename = get_filename(evaluation_function, budget, branches, iteration=i, gate_set=gate_set, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, gradient=False, roll_out_steps=roll_out_steps, image=False)

        if os.path.isfile(filename+'.pkl'):
            df = pd.read_pickle(filename+'.pkl')
            qc_along_path.append([circuit for circuit in df['qc']])
            children = df['children'].tolist()
            value = df['value'].tolist()
            visits = df['visits'].tolist()
        else:
            return FileNotFoundError
    return qc_along_path, children, visits, value


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
    elif evaluation_function == h2o:
        return -75.16, -75.49



def best_in_path(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter):
    d = get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon,stop_deterministic,  n_iter)[0]

    cost_overall, best_index = [], []
    for i in range(n_iter):
        cost = list(map(lambda x: evaluation_function(x, cost=True), d[i]))
        minimum = min(cost)
        """if isinstance(minimum, float):
            cost_overall.append(minimum)
        else:
            cost_overall.append(minimum.numpy())"""
        cost_overall.append(minimum)

        best_index.append(cost.index(minimum))
    return cost_overall, best_index


def plot_cost(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, gradient=False):
    """It saves the convergence plot of the cost vs tree depth"""
    d = get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic,  n_iter=n_iter)[0]
    max_tree_depth = len(max(d, key=len))
    indices = list(range(max_tree_depth+2))
    if max_tree_depth > 20:
        indices = indices[::2]

    plt.xlabel('Tree Depth')
    plt.ylabel('Cost')
    plt.xticks(indices)
    for i in range(n_iter):
        cost = list(map(lambda x: evaluation_function(x, cost=True), d[i]))
        plt.plot(list(range(len(cost))), cost, marker='o', linestyle='-', label=str(i+1))
    benchmark_value = None
    if evaluation_function == h2 or evaluation_function == h2o or evaluation_function == lih:
        benchmark_value = get_benchmark(evaluation_function)
        plt.ylabel('Energy (Ha)')
    if evaluation_function == h2:
        plt.yticks(np.arange(-1.2, 0.1, 0.1))

    if benchmark_value is not None:
        if isinstance(benchmark_value, list) or isinstance(benchmark_value, tuple):
            plt.axhline(y=benchmark_value[0], color='r', linestyle='--', label=f'bench_SCF({round(benchmark_value[0], 3)})')
            plt.axhline(y=benchmark_value[1], color='g', linestyle='--', label=f'bench_FCI({round(benchmark_value[1], 3)})')

        else:
            plt.axhline(y=benchmark_value, color='r', linestyle='--', label=f'ADAPT-VQE({round(benchmark_value, 3)})')
    filename = get_filename(evaluation_function=evaluation_function, branches=branches, image=True, roll_out_steps=roll_out_steps, rollout_type=rollout_type, iteration=0, budget=budget, epsilon=epsilon, stop_deterministic=stop_deterministic, gradient=gradient) + '_budget_'+str(budget)
    plt.legend(loc='best')
    plt.title(evaluation_function.__name__ + ' - Budget  '+str(budget))

    plt.savefig(filename + '_cost.png')
    print('Cost Plot image saved in ', filename)
    plt.clf()


def boxplot(evaluation_function, branches, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, best, gradient):
    """ Save a boxplot image, with the stats on the n_iter independent runs vs the budget of mcts"""
    solutions = []
    BUDGET = [1000, 2000, 5000, 10000, 50000, 100000, 200000]

    if gradient:
        best = False
    # Gate Data

    for budget in BUDGET:
        if not check_file_exist(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, gradient, n_iter):
            index = BUDGET.index(budget)
            BUDGET.pop(index)
            continue
        qc_solutions = []
        if not gradient:
            d = get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter)[0]
            qc_solutions = [d[i][-1] for i in range(n_iter)]   # leaf nodes
        if evaluation_function == h2 or evaluation_function == vqls_1 or evaluation_function == lih or evaluation_function == h2o or evaluation_function == qc_regeneration:
            if best:
                sol = best_in_path(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter)[0]
                solutions.append(sol)
            elif gradient:
                sol = get_best_overall(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter)
                solutions.append(sol)
            else:
                sol = list(map(lambda x: evaluation_function(x, cost=True), qc_solutions))
                """if evaluation_function == qc_regeneration:
                    solutions.append([x for x in sol])
                else:
                    solutions.append([x.numpy() for x in sol])"""
                solutions.append([x for x in sol])


        elif evaluation_function == sudoku2x2:
            gates = [qc.to_gate(label='Oracle Approx') for qc in qc_solutions]

            counts_approx = [grover_algo(oracle='approximation', oracle_gate=gates[i], iterations=2, ancilla=1) for i in
                             range(len(gates))]
            for i in range(n_iter):
                if '1001' not in counts_approx[i]:
                    counts_approx[i]['1001'] = 0
                elif '0110' not in counts_approx[i]:
                    counts_approx[i]['0110'] = 0
                else:
                    pass

            solutions.append([x['1001'] + x['0110'] for x in counts_approx])
        print('Boxplot created at budget: ', budget)
    # Plotting
    lab = [str(b) for b in BUDGET]
    print(lab)
    plt.boxplot(solutions, patch_artist=True, labels=lab, meanline=True, showmeans=True)

    benchmark = get_benchmark(evaluation_function)
    if evaluation_function == h2:
        plt.ylabel('Energy (Ha)')
        flat_solutions = [x for xs in solutions for x in xs]
        maximum = round(max(flat_solutions), 1)
        plt.yticks(np.arange(-1.2, maximum+0.01, step=0.05))
        benchmark = round(benchmark[1], 3)
        label = 'bench_FCI'
    elif evaluation_function == h2o:
        plt.ylabel('Energy (Ha)')
        benchmark = round(benchmark[1], 3)
        label = 'bench_FCI'
    elif evaluation_function == lih:
        plt.ylabel('Energy (Ha)')
        benchmark = round(benchmark, 3)
        label = 'ADAPT-VQE'
    elif evaluation_function == qc_regeneration:
        plt.ylabel('Distance')
        benchmark = 0
        label = 'Benchmark'
    elif evaluation_function == vqls_1:
        label = 'benchmark'
        plt.ylabel('Cost')
    elif evaluation_function == sudoku2x2:
        plt.ylabel('Right Counts')
        label = 'exact_oracle'
    else:
        raise NotImplementedError

    plt.axhline(y=benchmark, color='r', linestyle='--', label=label)
    filename = get_filename(evaluation_function=evaluation_function, branches=branches, image=True, roll_out_steps=roll_out_steps, rollout_type=rollout_type, iteration=0, epsilon=epsilon, stop_deterministic=stop_deterministic, gradient=gradient, budget=0)
    plt.title(evaluation_function.__name__)
    plt.xlabel('MCTS Simulations')
    plt.legend()
    if best:
        filename = filename+'_best'
    plt.savefig(filename + '_boxplot.png')

    plt.clf()
    print('boxplot image saved in ', filename)
    return solutions


def get_qc_depth(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type, n_iter):
    d = get_paths(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type, n_iter)[1]
    qc_solutions = [d[0][i][-1] for i in range(n_iter)]  # leaf nodes
    depth = [qc.depth() for qc in qc_solutions]
    return depth


def add_gradient_descent_column(evaluation_function, budget, iteration, branches, epsilon, stop_deterministic, roll_out_steps, gate_set='continuous', rollout_type="classic"):
    indices = best_in_path(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter=iteration)[1]
    for i in range(iteration):
        filename = get_filename(evaluation_function, budget, branches, iteration=i, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                                epsilon=epsilon, stop_deterministic=stop_deterministic, gradient=False, image=False)
        d = get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, iteration)[0]
        df = pd.read_pickle(filename + '.pkl')

        quantum_circuit_last = d[i][-1]
        if evaluation_function == qc_regeneration:
            final_result = evaluation_function(quantum_circuit_last, cost=False, gradient=True)
        else:
            final_result = evaluation_function(quantum_circuit_last, ansatz='', cost=False, gradient=True)

        final_result = [x for x in final_result]
        column = [[None]]*df.shape[0]
        column[-1] = final_result

        index = indices[i]

        if index != len(d[i]):
            quantum_circuit_best = d[i][index]
            if evaluation_function == qc_regeneration:
                best_result = evaluation_function(quantum_circuit_last, cost=False, gradient=True)
            else:
                best_result = evaluation_function(quantum_circuit_best, ansatz='', cost=False, gradient=True)
            best_result = [x for x in best_result]
            column[index] = best_result
        df["Adam"] = column
        df.to_pickle(os.path.join(filename + '_comp.pkl'))


def get_best_overall(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter):
    best = []
    for i in range(n_iter):
        filename = get_filename(evaluation_function=evaluation_function, budget=budget, iteration=i, branches=branches, epsilon=epsilon, stop_deterministic=stop_deterministic, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                                gradient=True, image=False)
        df = pd.read_pickle(filename + '.pkl')
        column = df['Adam']
        final = [column[j][-1] for j in range(df.shape[0]) if column[j][0] is not None]
        best.append(min(k for k in final if not math.isnan(k)))

    return best

def check_file_exist(evaluation_function, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, gradient, n_iter=10, gate_set='continuous'):
    """ :return: bool. True if all the files are stored, false otherwise"""
    check = True
    for i in range(n_iter):
        filename = get_filename(evaluation_function, budget, branches, iteration=i, gate_set=gate_set, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, gradient=gradient, roll_out_steps=roll_out_steps, image=False)
        if not os.path.isfile(filename+'.pkl'):
            check = False
    return check
