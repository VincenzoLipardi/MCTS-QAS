from problems.oracles.grover import grover
from problems.vqe import H2
from problems.vqls import VQLS
import heapq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile


# Evaluation Function for the Grover Oracle
def sudoku2x2(quantum_circuit, n_solutions=2):
    """ Oracle Approximation
    :param quantum_circuit: quantum circuit approximating a quantum oracle
    :param n_solutions: We are supposed to know the number of solutions of the problem
    :return: float. Reward
    """
    gate = quantum_circuit.to_gate(label='Oracle Approx')
    # n_iteration have to be generalised has np.pi/4*(np.sqrt(N/M))
    counts = grover.grover_algo(oracle='approximation', oracle_gate=gate, iterations=2, ancilla=1)
    tot_counts = sum(counts.values())
    best_counts = heapq.nlargest(n_solutions, counts.values())
    solution = [i for i in counts if counts[i] in best_counts]
    reward = 0
    for sol in solution:
        if check_constraints(sol):
            if counts[sol] <= tot_counts / 2:
                reward += counts[sol] / tot_counts
            else:
                reward += tot_counts/2
    return reward


# Sudoku 2x2
def check_constraints(solution):
    values = [int(i) for i in solution]
    return values[0] != values[1] and values[1] != values[3] and values[2] != values[3]


def oracle_function(quantum_circuit, function, x, shots):
    """
    It evaluates the approximation of qc to be a quantum gate W such that W|x>|0>= |x>(sqrt(R(x))|0> + sqrt(1-R(x))|1>)
    :param quantum_circuit: Quantum circuit to be evaluated
    :param function: Function R(x) to be reproduced by the quantum gate W. 0<=R(x)<=1.
    :param x: string. Input value. It is a computational basis state
    :param shots: Number of shots to evaluate the modulo squared of the amplitude of the last qubit in |0>
    :return: Reward between 0 and 1
    """
    n = len(quantum_circuit.qubits)
    gate = quantum_circuit.to_gate(label='Oracle Approx')
    x_register = QuantumRegister(n-1, name='x')
    y_register = QuantumRegister(1,  name='y')
    cbit = ClassicalRegister(1)
    qc = QuantumCircuit(x_register, y_register, cbit)
    for i in range(len(x)):
        if x[len(x)-1-i] == '1':
            qc.x(i)

    qc.append(gate, [x_register, y_register])
    qc.measure(y_register, cbit)

    # Simulate and plot results
    qasm_simulator = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, qasm_simulator)
    result = qasm_simulator.run(transpiled_qc, shots=shots).result()
    counts = result.get_counts()
    print(counts)
    r_x = counts['0']/shots
    reward = abs(function(x)-r_x)
    return reward


def h2(quantum_circuit, ansatz='all', cost=False):
    if cost:
        return H2().costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return H2().getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def vqls_1(quantum_circuit, ansatz='all', cost=False):

    # Define the problem A = c_0 I + c_1 X_1 + c_2 X_2 + c_3 Z_3 Z_4
    #problem = VQLS(c=[1, 0.1, 0.1, 0.2])
    problem = VQLS(c=[1, 0.5, 0.4, 0.7])
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1, 0.2], quantum_circuit=quantum_circuit, ansatz=ansatz)



