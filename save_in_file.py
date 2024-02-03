import mcts
import pandas as pd
import os.path
import numpy as np
from structure import Circuit
import matplotlib.pyplot as plt
from problems.evaluation_functions import h2, vqls_1, sudoku2x2
from problems.vqe import H2
from problems.oracles.grover.grover import grover_algo
from qiskit.visualization import plot_histogram


def get_filename(evaluation_function, budget, branches, iteration, image, gate_set='continuous', rollout_type="classic", roll_out_steps=None):
    """ it creates the string of the file name that have to be saved or read"""

    ro = ''
    ros = ''
    if evaluation_function == h2 or evaluation_function == vqls_1:
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
    if image:
        filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '/' + ro+branch + ros
    else:
        filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '/' + ro + branch + '_budget_' + str(
            budget) + ros + '_run_' + str(iteration)
    return filename


def data(evaluation_function, variable_qubits, ancilla_qubits, budget, max_depth, iteration, branches, choices, gate_set='continuous', rollout_type="classic", roll_out_steps=None, verbose=True):
    """
    It runs the mcts on the indicated problem and saves the result (the best path) in a .pkl file
    :param choices: dict. Probability over all the possible choices
    :param evaluation_function: func. It defines the problem, then the reward function for the mcts agent
    :param variable_qubits:int.  Number of qubits required for the problem
    :param ancilla_qubits: int. Number of ancilla qubits required, as in the case of the oracle problem (Hyperparameter)
    :param gate_set: str.
    :param budget: int. resources allocated for the mcts search. MCTS iterations
    :param max_depth: int. Max depth of the quantum circuit
    :param iteration: int. Number of the indipendent run.
    :param branches: bool or int. If True progressive widening implemented. If int the number of maximum branches is fixed.
    :param rollout_type: str. classic evaluates the final quantum circuit got after rollout. max takes the best reward get from all the states in the rollout path
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
    final_state = mcts.mcts(root, budget=budget, branches=branches, evaluation_function=evaluation_function, rollout_type=rollout_type, roll_out_steps=roll_out_steps, choices=choices, verbose=verbose)

    filename = get_filename(evaluation_function, budget, branches, iteration, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps, image=False)

    df = pd.DataFrame(final_state)
    df.to_pickle(os.path.join(filename + '.pkl'))
    return print("files saved in experiments/", evaluation_function.__name__, 'as ', filename)


def get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, n_iter=10, gate_set='continuous'):
    """ It opens the .pkl files and returns quantum circuits along the best path """
    qc_along_path = []
    children, visits, value = [], [], []
    for i in range(n_iter):
        print(i)
        filename = get_filename(evaluation_function, budget, branches, iteration=i, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps, image=False)
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
        problem = H2().benchmark()
        classical_sol = -1.136189454088     # Full configuration Interaction
        return problem, classical_sol
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
    else:
        return 0



def plot_cost(evaluation_function, branches, budget, roll_out_steps, rollout_type, n_iter):
    """It saves the convergence plot of the cost vs tree depth"""
    d = get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type, n_iter)[0]
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
    if evaluation_function == h2:
        benchmark_value = get_benchmark(evaluation_function)[1]
        plt.yticks(np.arange(-1.2, 0.1, 0.1))
    if benchmark_value is not None:
        plt.axhline(y=benchmark_value, color='r', linestyle='--', label=f'bench_FCI({round(benchmark_value, 3)})')
    if isinstance(branches, bool) and branches:
        branch = "_pw_aba"
    elif isinstance(branches, int):
        branch = '_bf_' + str(branches)
    else:
        raise TypeError
    filename = get_filename(evaluation_function=evaluation_function, branches=branches, image=True, roll_out_steps=roll_out_steps, iteration=0, budget=budget)
    plt.legend(loc='best')
    plt.title(evaluation_function.__name__)

    plt.savefig(filename + '_cost.png')
    print('image saved')
    plt.clf()


def plot_oracle(evaluation_function, max_branches, gate_set, budget, roll_out_steps=0, rollout_type=None):
    d = get_paths(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type)
    gate = d[2][-1].to_gate(label='Oracle Approx')
    counts_approx = grover_algo(oracle='approximation', oracle_gate=gate, iterations=2, ancilla=1)
    counts_exact = grover_algo(oracle='exact', iterations=2)

    legend = ['Exact', 'Approx']

    filename = 'experiments/sudoku2x2/Istogram' + gate_set + '_bf_' + str(max_branches) + '_budget_' + str(budget)

    plot_histogram([counts_exact, counts_approx], legend=legend, color=['crimson', 'midnightblue'],
                   figsize=(15, 10), filename=filename)
    print('image saved')
    plt.clf()


def boxplot(evaluation_function, branches, roll_out_steps, rollout_type, n_iter):
    """ Save a boxplot image, with the stats on the n_iter independent runs vs the budget of mcts"""
    solutions = []
    BUDGET = [1000, 2000, 5000, 10000, 50000]

    # Gate Data
    for budget in BUDGET:
        d = get_paths(evaluation_function, branches, budget, roll_out_steps, rollout_type)[0]
        if d is None:
            break
        qc_solutions = [d[i][-1] for i in range(n_iter)]   # leaf nodes
        if evaluation_function == h2 or evaluation_function == vqls_1:
            sol = list(map(lambda x: evaluation_function(x, cost=True), qc_solutions))
            solutions.append([x.numpy() for x in sol])
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
        print(budget)

    # Plotting
    lab = [str(b) for b in BUDGET]
    plt.boxplot(solutions, patch_artist=True, labels=lab, meanline=True, showmeans=True)

    benchmark = get_benchmark(evaluation_function)
    if evaluation_function == h2:
        plt.ylabel('Cost')
        flat_solutions = [x for xs in solutions for x in xs]
        maximum = round(max(flat_solutions), 1)
        plt.yticks(np.arange(-1.2, maximum+0.01, step=0.05))
        benchmark = round(benchmark[1], 3)
        label = 'bench_FCI'
    elif evaluation_function == vqls_1:
        label = 'benchmark'
        plt.ylabel('Cost')
    elif evaluation_function == sudoku2x2:
        plt.ylabel('Right Counts')
        label = 'exact_oracle'

    else:
        raise NotImplementedError

    plt.axhline(y=benchmark, color='r', linestyle='--', label=label)
    filename = get_filename(evaluation_function=evaluation_function, branches=branches, image=True, roll_out_steps=roll_out_steps, iteration=0, budget=0)
    plt.title(evaluation_function.__name__)
    plt.xlabel('MCTS Simulations')
    plt.legend()
    plt.savefig(filename + '_boxplot.png')

    plt.clf()
    print('boxplot image saved')
    return solutions


def get_qc_depth(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type, n_iter):
    d = get_paths(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type, n_iter)[1]
    qc_solutions = [d[0][i][-1] for i in range(n_iter)]  # leaf nodes
    depth = [qc.depth() for qc in qc_solutions]
    return depth


# boxplot(roll_out_steps=1, rollout_type='classic', n_iter=10)
plot_cost(evaluation_function=sudoku2x2, branches=False, budget=1000, roll_out_steps=1, rollout_type='classic', n_iter=10)