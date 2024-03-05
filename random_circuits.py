import pandas as pd
import matplotlib.pyplot as plt
import pennylane.numpy as np
from problems.utils import optimize_qc, get_matrix, create_random_qc, stabilizer_renyi_entropy


def save_pkl(qubits: int, gates: int, samples=10):
    filename = 'problems/oracles/dataset/random_circuit_qubit_'+str(qubits)+'_gates_'+str(gates)
    circuits, n_qubits, unitary, t_gate, sre = [], [], [], [], [],
    for _ in range(samples):
        n_qubits.append(qubits)
        qc = optimize_qc(create_random_qc(qubits, gates))
        circuits.append(qc)
        unitary.append(get_matrix(qc))
        t_gate.append(qc.count_ops().get('t', 0))
        sre.append(stabilizer_renyi_entropy(qc))
    # Raw Data
    data = {'n_qubits': n_qubits, 'quantum_circuit': circuits, 'operator': unitary, 't_gate': t_gate,  'sre': sre}
    df = pd.DataFrame(data)
    df.to_pickle(filename + '.pkl')
    print('.pkl file saved as ', filename)




def stats(filename):
    df = pd.read_pickle(filename+'.pkl')

    data = df.groupby(df.iloc[:, 0]).mean()
    data.drop(['n_gates', 'n_qubits'], axis=1)

    df = pd.DataFrame(data)
    filename = filename+'_stats'+'.pkl'
    df.to_pickle(filename)
    print('.pkl saved as ', filename)


def plot_histogram(filename):
    df = pd.read_pickle(filename+'.pkl')
    species = tuple(range(2, 11))
    gates = {'cx': tuple(df['cx_gate']), 'h': tuple(df['h_gate']), 's': tuple(df['s_gate']), 't': tuple(df['t_gate'])}
    x = np.arange(len(species))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in gates.items():
        offset = width * multiplier
        ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Counts')
    ax.set_xlabel('Qubits')
    ax.set_title('Circuit Average Gates Composition')
    ax.set_xticks(x + width, species)
    ax.set_yticks(range(0, 11, 1))
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 11)
    plt.savefig(filename+'.png')


def plot_sre(qubits: int, gates: int):
    filename = 'problems/oracles/dataset/random_circuit_qubit_' + str(qubits) + '_gates_' + str(gates)
    df = pd.read_pickle(filename + '.pkl')


    x = range(0, 10, 1)

    fig, ax1 = plt.subplots()

    # Plotting the first dataset with primary y-axis
    ax1.plot(x, df['t_gate'], linestyle="-", marker='o', c='b', label='T gates')
    ax1.set_xlabel('Circuit Index')
    ax1.set_ylabel('T gates', color='b')

    # Creating a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, df['sre'], linestyle="-", marker='o', label='SRE', c='r')
    ax2.set_ylabel('SRE', color='r')

    # Adding labels and legend
    plt.xticks(x)
    plt.title('random Circuit Generation')
    plt.legend()

    plt.savefig(filename+'.png')
