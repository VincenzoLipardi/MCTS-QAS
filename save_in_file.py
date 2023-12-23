import mcts
import pandas as pd
import os.path
from structure import Circuit
import matplotlib.pyplot as plt
from problems.evaluation_functions import h2, vqls_1, sudoku2x2
from problems.vqe import H2
from problems.oracles.grover.grover import grover_algo
from qiskit.visualization import plot_histogram


def data(evaluation_function, variable_qubits, ancilla_qubits, gate_set, budget, max_branches, max_depth, roll_out_steps, iteration, verbose):
    root = mcts.Node(Circuit(variable_qubits=variable_qubits, ancilla_qubits=ancilla_qubits), max_depth=max_depth)

    final_state = mcts.mcts(root, budget=budget, max_branches=max_branches, evaluation_function=evaluation_function, roll_out_steps=roll_out_steps, verbose=verbose)
    if verbose:
        print("Value best node overall: ", final_state[0].value)
    filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '_bf_' + str(max_branches) + '_budget_' + str(budget)+'run_'+str(iteration)

    df = pd.DataFrame(final_state[1], columns=['path'])
    df.to_pickle(os.path.join(filename + '.pkl'))
    return print("files saved in experiments/", evaluation_function.__name__)


def get_pkl(evaluation_function, max_branches, gate_set, budget, verbose=False):

    filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)+'run_'+str(0)
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


def get_benchmark(evaluation_function):
    if evaluation_function == h2:
        # List
        problem = H2().benchmark()
        classical_sol = -1.136189454088     # Full configuration Interaction
        return problem, classical_sol


def plot_Cost_Func(evaluation_function, max_branches, gate_set, budget):
    data = get_pkl(evaluation_function, max_branches, gate_set, budget)[2]
    indices = list(range(len(data)))
    cost = list(map(lambda x: evaluation_function(x, cost=True), data))
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
    title = evaluation_function.__name__ + '_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)
    plt.legend()
    plt.title(title)
    plt.savefig('experiments/' + evaluation_function.__name__ + '/C_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(
            budget) + '.png')
    print('image saved')
    plt.clf()


def plot_Reward_Func(evaluation_function, max_branches, gate_set, budget):
    data = get_pkl(evaluation_function, max_branches, gate_set, budget)[2]
    indices = list(range(len(data)))
    reward = list(map(evaluation_function, data))
    plt.plot(indices, reward, marker='o', linestyle='-', label='Reward')
    plt.xlabel('Tree Depth')
    plt.ylabel('Reward')
    plt.xticks(indices)

    title = evaluation_function.__name__ + '_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)
    plt.title(title)
    plt.savefig('experiments/'+evaluation_function.__name__ + '/R_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)+'.png')

    print('image saved')
    plt.clf()

def plot_oracle(evaluation_function, max_branches, gate_set, budget):
    data = get_pkl(evaluation_function, max_branches, gate_set, budget)
    gate = data[2][-1].to_gate(label='Oracle Approx')
    counts_approx = grover_algo(oracle='approximation', oracle_gate=gate, iterations=2, ancilla=1)
    counts_exact = grover_algo(oracle='exact', iterations=2)

    legend = ['Exact', 'Approx']
    filename = 'experiments/sudoku2x2/Istogram' + gate_set + '_bf_' + str(max_branches) + '_budget_' + str(budget)

    plot_histogram([counts_exact, counts_approx], legend=legend, color=['crimson', 'midnightblue'],
                   figsize=(15, 10), filename=filename)
    print('image saved')
    plt.clf()

