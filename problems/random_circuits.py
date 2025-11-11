import pandas as pd

from problems.utils import optimize_qc, get_matrix, create_random_qc, stabilizer_renyi_entropy


def save_pkl(qubits: int, gates: int, samples: int):
    filename = 'oracles/dataset/random_circuit_qubit_'+str(qubits)+'_gates_'+str(gates)
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

