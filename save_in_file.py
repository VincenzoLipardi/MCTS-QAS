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


def data(evaluation_function, variable_qubits, ancilla_qubits, gate_set, budget, max_depth, iteration, verbose, branches, rollout_type="classic", roll_out_steps=None, ):
    root = mcts.Node(Circuit(variable_qubits=variable_qubits, ancilla_qubits=ancilla_qubits), max_depth=max_depth)

    final_state = mcts.mcts(root, budget=budget, branches=branches, evaluation_function=evaluation_function, rollout_type=rollout_type, roll_out_steps=roll_out_steps, verbose=verbose)
    if verbose:
        print("Value best node overall: ", final_state[0].value)
    ro = 'rollout_' + rollout_type + '/'
    if isinstance(branches, bool) and branches:
        branch = "_pw"
    elif isinstance(branches, int):
        branch = '_bf_' + str(branches)
    else:
        raise TypeError
    filename = 'experiments/' + evaluation_function.__name__ + '/' + ro + gate_set + branch + '_budget_' + str(budget)+'_ro_'+str(roll_out_steps)+'_run_'+str(iteration)

    df = pd.DataFrame(final_state[1], columns=['path'])

    df.to_pickle(os.path.join(filename + '.pkl'))
    return print("files saved in experiments/", evaluation_function.__name__)


def get_pkl(evaluation_function, branches, gate_set, budget, roll_out_steps, rollout_type, verbose=False):
    ro = ''
    if rollout_type is not None:
        ro = 'rollout_' + rollout_type + '/'
    if isinstance(branches, bool) and branches:
        branch = "_pw"
    elif isinstance(branches, int):
        branch = '_bf_' + str(branches)
    else:
        raise TypeError
    filename = 'experiments/' + evaluation_function.__name__ + '/' + ro + gate_set + branch + "_budget_" + str(budget)+'_ro_'+str(roll_out_steps)+'_run_'+str(0)
    df = pd.read_pickle(filename+'.pkl')

    path = df['path']
    values_along_path = [node.value for node in path]
    visits_along_path = [node.visits for node in path]
    qc_along_path = [node.state.circuit for node in path]
    path_list = list(path)
    if verbose:
        for node in path:
            print("node number:", path_list.index(node))
            # print(node.circuit.qc)
            print('node_visit:', node.visits)
            print('value', node.value)
    return values_along_path, visits_along_path, qc_along_path


def get_paths(evaluation_function, branches, gate_set, budget, roll_out_steps, rollout_type, n_iter):
    values_along_path = []
    visits_along_path = []
    qc_along_path = []
    ro = '/rollout_'+rollout_type
    if evaluation_function == sudoku2x2:
        ro = ''
    if isinstance(branches, bool) and branches:
        branch = "_pw"
    elif isinstance(branches, int):
        branch = '_bf_' + str(branches)
    else:
        raise TypeError
    for i in range(n_iter):
        filename = 'experiments/' + evaluation_function.__name__ + ro+'/' + gate_set + branch + "_budget_" + str(budget)+'_ro_'+str(roll_out_steps)+'_run_'+str(i)
        df = pd.read_pickle(filename+'.pkl')
        path = df['path']

        qc_along_path.append([node.state.circuit for node in path])
        if evaluation_function != sudoku2x2:
            values_along_path.append([node.value.numpy() for node in path])
            visits_along_path.append([node.visits for node in path])

    return values_along_path, qc_along_path


def get_benchmark(evaluation_function):
    if evaluation_function == h2:
        # List
        problem = H2().benchmark()
        classical_sol = -1.136189454088     # Full configuration Interaction
        return problem, classical_sol


def plot_Cost_Func(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type=None):
    d = get_pkl(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type)[2]
    indices = list(range(len(d)))
    cost = list(map(lambda x: evaluation_function(x, cost=True), d))
    color = 'tab:blue'

    plt.xlabel('Tree Depth')
    plt.ylabel('Cost', color=color)
    plt.xticks(indices)
    plt.plot(indices, cost, color=color, marker='o', linestyle='-', label='MCTS')
    benchmark_value = None
    if evaluation_function == h2:
        benchmark_value = get_benchmark(evaluation_function)[1]
    if benchmark_value is not None:
        plt.axhline(y=benchmark_value, color='r', linestyle='--', label=f'bench_FCI({benchmark_value})')
    title = evaluation_function.__name__ + '_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)+'_ro_'+str(roll_out_steps)
    plt.legend()
    plt.title(title)
    ro = ''
    if rollout_type is not None:
        ro = '/rollout_' + rollout_type
    plt.savefig('experiments/' + evaluation_function.__name__ + ro + '/C_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)+'_ro_'+str(roll_out_steps) + '.png')
    print('image saved')
    plt.clf()


def plot_cost_all_iterations(evaluation_function, branches, gate_set, budget, roll_out_steps, rollout_type, n_iter=10):
    d = get_paths(evaluation_function, branches, gate_set, budget, roll_out_steps, rollout_type, n_iter)[1]

    indices = list(range(len(max(d, key=len))))
    plt.xlabel('Tree Depth')
    plt.ylabel('Cost')
    plt.xticks(indices)

    for i in range(n_iter):
        cost = list(map(lambda x: evaluation_function(x, cost=True), d[i]))
        plt.plot(list(range(len(cost))), cost, marker='o', linestyle='-', label=str(i+1))
    benchmark_value = None
    if evaluation_function == h2:
        benchmark_value = get_benchmark(evaluation_function)[1]
        plt.yticks(np.arange(-1.2, 0, 0.1))
    if benchmark_value is not None:
        plt.axhline(y=benchmark_value, color='r', linestyle='--', label=f'bench_FCI({round(benchmark_value, 3)})')
    if isinstance(branches, bool) and branches:
        branch = "_pw"
    elif isinstance(branches, int):
        branch = '_bf_' + str(branches)
    else:
        raise TypeError
    title = evaluation_function.__name__ + '_' + gate_set + branch + "_budget_" + str(
        budget)
    plt.legend(loc='best')
    plt.title(title)
    ro = ''
    if rollout_type is not None:
        ro = '/rollout_' + rollout_type
    plt.savefig('experiments/' + evaluation_function.__name__ + ro + '/C_' + gate_set + branch + "_budget_" + str(budget) + '_ro_' + str(roll_out_steps) + '.png')
    print('image saved')
    plt.clf()


def plot_Reward_Func(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type=None):
    d = get_pkl(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type)[2]
    indices = list(range(len(d)))
    reward = list(map(evaluation_function, d))
    plt.plot(indices, reward, marker='o', linestyle='-', label='Reward')
    plt.xlabel('Tree Depth')
    plt.ylabel('Reward')
    plt.xticks(indices)
    ro = ''
    if rollout_type is not None:
        ro = '/rollout_' + rollout_type
    title = evaluation_function.__name__ + '_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)
    plt.title(title)
    plt.savefig('experiments/'+evaluation_function.__name__ + ro + '/R_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)+'.png')

    print('image saved')
    plt.clf()


def plot_oracle(evaluation_function, max_branches, gate_set, budget):
    d = get_pkl(evaluation_function, max_branches, gate_set, budget, roll_out_steps=0, rollout_type=None)
    gate = d[2][-1].to_gate(label='Oracle Approx')
    counts_approx = grover_algo(oracle='approximation', oracle_gate=gate, iterations=2, ancilla=1)
    counts_exact = grover_algo(oracle='exact', iterations=2)

    legend = ['Exact', 'Approx']

    filename = 'experiments/sudoku2x2/Istogram' + gate_set + '_bf_' + str(max_branches) + '_budget_' + str(budget)

    plot_histogram([counts_exact, counts_approx], legend=legend, color=['crimson', 'midnightblue'],
                   figsize=(15, 10), filename=filename)
    print('image saved')
    plt.clf()


def plot_sudoku(evaluation_function, max_branches, gate_set, roll_out_steps=0, rollout_type='', n_iter=10):
    BUDGET = [1000, 2000, 5000, 10000]
    # counts_exact = grover_algo(oracle='exact', iterations=2)
    counts_approx_good = []
    for budget in BUDGET:
        d = get_paths(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type, n_iter)[1]
        qc_solutions = [d[i][-1] for i in range(n_iter)]  # leaf nodes
        gates = [qc.to_gate(label='Oracle Approx') for qc in qc_solutions]

        counts_approx = [grover_algo(oracle='approximation', oracle_gate=gates[i], iterations=2, ancilla=1) for i in range(len(gates))]
        print(counts_approx)
        counts_approx_good = [sub['0110']+sub['1001'] for sub in counts_approx]

    return counts_approx_good


def h2_boxplot(evaluation_function, max_branches, gate_set, roll_out_steps, rollout_type, n_iter=10):
    solutions = []
    BUDGET = [1000, 2000, 5000, 10000, 50000]
    for budget in BUDGET:
        d = get_paths(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type, n_iter)[1]
        qc_solutions = [d[i][-1] for i in range(n_iter)]   # leaf nodes
        sol = list(map(lambda x: evaluation_function(x, cost=True), qc_solutions))
        solutions.append([x.numpy() for x in sol])

    # plt.figure(figsize=(10, 7))
    # Creating plot
    lab = [str(b) for b in BUDGET]
    plt.boxplot(solutions, patch_artist=True, labels=lab, meanline=True)

    plt.xlabel('MCTS Simulations')
    plt.ylabel('Cost')

    flat_solutions = [x for xs in solutions for x in xs]
    maximum = round(max(flat_solutions), 1)
    plt.yticks(np.arange(-1.2, maximum+0.1, step=0.1))
    plt.axhline(y=-1.136, color='r', linestyle='--', label='bench_FCI')

    ro = '/rollout_' + rollout_type
    title = evaluation_function.__name__ + '_' + gate_set + '_bf_' + str(max_branches)+'_ro_' + str(roll_out_steps)
    plt.title(title)

    plt.savefig('experiments/' + evaluation_function.__name__ + ro+'/boxplot_' + gate_set + '_bf_' + str(
        max_branches) + '_ro_' + str(roll_out_steps) + '.png')

    return solutions


def get_qc_depth(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type, n_iter=10):
    d = get_paths(evaluation_function, max_branches, gate_set, budget, roll_out_steps, rollout_type, n_iter)[1]
    qc_solutions = [d[i][-1] for i in range(n_iter)]  # leaf nodes
    depth = [qc.depth() for qc in qc_solutions]
    return depth
