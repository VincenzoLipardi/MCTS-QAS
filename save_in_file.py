import mcts
import pandas as pd
import os.path
from structure import Circuit
import matplotlib.pyplot as plt


def data(evaluation_function, variable_qubits, ancilla_qubits, gate_set, budget, max_branches, verbose):
    root = mcts.Node(Circuit(variable_qubits=variable_qubits, ancilla_qubits=ancilla_qubits))

    final_state = mcts.mcts(root, budget=budget, max_branches=max_branches, evaluation_function=evaluation_function, verbose=verbose)
    if verbose:
        print("Value best node overall: ", final_state[0].value)
    filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '_bf_' + str(max_branches) + '_budget_' + str(budget)

    df = pd.DataFrame(final_state[1], columns=['path'])
    df.to_pickle(os.path.join(filename + '.pkl'))
    return print("files saved in experiments/", evaluation_function.__name__)


def get_pkl(evaluation_function, max_branches, gate_set, budget, verbose=False):

    filename = 'experiments/' + evaluation_function.__name__ + '/' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)
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

def plot_Cost_Func(evaluation_function, max_branches, gate_set, budget):
    data = get_pkl(evaluation_function, max_branches, gate_set, budget)[2]

    indices = list(range(len(data)))
    reward = list(map(evaluation_function, data))
    cost = list(map(lambda x: evaluation_function(x, cost=True), data))
    plt.scatter(indices, reward, label='Reward')
    plt.scatter(indices, reward, label='Cost')

    title = evaluation_function.__name__ + '_' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Tree Depth')
    ax1.set_ylabel('Reward', color=color)
    ax1.scatter(indices, reward, label='Reward', color='blue')
    ax1.plot(indices, reward, color=color, linestyle='-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Cost', color=color)
    ax2.scatter(indices, cost, color=color, label='Cost')
    ax2.plot(indices, cost, color=color, linestyle='-', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    #plt.xlabel('Tree depth')
    #plt.ylabel('Cost')

    plt.title(title)
    #ax1.legend()
    #ax2.legend()
    plt.savefig('experiments/'+evaluation_function.__name__ + '/' + gate_set + '_bf_' + str(max_branches) + "_budget_" + str(budget)+'.png')
    print('image saved')
    plt.clf()



