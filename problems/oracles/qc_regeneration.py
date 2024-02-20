import pandas as pd
from qiskit import Aer, transpile, execute
from structure import GateSet, Circuit, actions_on_circuit
import matplotlib.pyplot as plt
import numpy as np



def create_random_qc(n_qubits, gate_number):
    circuit = Circuit(variable_qubits=n_qubits, ancilla_qubits=0, initialization=None)
    circuit_qiskit = circuit.circuit
    for i in range(gate_number):
        action = actions_on_circuit(action_chosen='a', gate_set=GateSet('discrete'))
        circuit_qiskit = action(circuit_qiskit)
    return circuit_qiskit


def optimize_qc(circuit):
    # Transpile the circuit for the target backend
    optimized_circuit = transpile(circuit, basis_gates=['cx', 'h', 't', 's'])
    return optimized_circuit


def matrix(quantum_circuit):
    simulator = Aer.get_backend("unitary_simulator")
    job = execute(quantum_circuit, backend=simulator)
    result = job.result()
    unitary = result.get_unitary()      # .data to get the numpy array object
    return unitary


def save_pkl(qubits: int, gates: int):
    filename = 'dataset/regeneration_clifford'
    circuits, n_qubits, n_gates, unitary, cx_gate, t_gate, h_gate, s_gate = [], [], [], [], [], [], [], []
    for q in range(2, qubits):
        for g in range(q, gates):
            print(q, g)
            n_qubits.append(q)
            n_gates.append(g)
            qc = optimize_qc(create_random_qc(q, g))
            circuits.append(qc)
            unitary.append(matrix(qc))
            cx_gate.append(qc.count_ops().get("cx", 0))
            t_gate.append(qc.count_ops().get('t', 0))
            h_gate.append(qc.count_ops().get('h', 0))
            s_gate.append(qc.count_ops().get('s', 0))



    data = {'n_qubits': n_qubits, 'n_gates': n_gates, 'quantum_circuit': circuits, 'operator': unitary, 'cx_gate': cx_gate, 't_gate': t_gate, 's_gate': s_gate, 'h_gate': h_gate}
    df = pd.DataFrame(data)
    df.to_pickle(filename + '.pkl')
    print('.pkl saved as ', filename)


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


def evaluation(matrix1, matrix2):
    l = 0
    n = len(matrix1[:][0])
    for i in range(n):
        for j in range(n):
            l += np.abs(matrix1[i][j]-matrix2[i][j])
    return l


def quantum_circuit_regeneration(quantum_circuit, filename):
    df = pd.read_pickle(filename+'.pkl')
    hardest = max(df['t_gate'])
    hardest_index = df[df['t_gate'] == hardest].index[0]

    matrix_goal = df['operator'][hardest_index].data
    matrix_test = matrix(quantum_circuit).data
    normalization_factor = 4*(2**len(quantum_circuit.qubits))**2
    distance = evaluation(matrix_test, matrix_goal)/normalization_factor
    return 1-distance

