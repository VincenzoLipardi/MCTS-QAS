from problems.oracles.grover import grover
from problems.vqe import h2_class, lih_class, h2o_class, h2o_full_class
from problems.combinatorial import maxcut_class
from problems.vqls import vqls_demo, vqls_paper
from problems.oracles.oracle_approximation import Fidelity, CircuitRegeneration
import heapq
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile


def maxcut(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = maxcut_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit)


# ORACLE APPROXIMATION
def qc_regeneration(quantum_circuit, cost=False, gradient=False):
    problem = CircuitRegeneration()
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return -problem.reward(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_20_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_20_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_20_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_20_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_20_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_20_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_15_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=15, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_15_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=15, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_15_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=15, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_15_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=15, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_15_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=15, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_15_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=15, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_10_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=10, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_10_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=10, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_10_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=10, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_10_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=10, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_10_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=10, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_10_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=10, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_5_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=5, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_5_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=5, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_5_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=5, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_5_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=5, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_5_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=5, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_5_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=5, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_30_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=30, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_30_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=30, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)

def fidelity_6_30_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=30, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_30_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=30, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)

def fidelity_8_30_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=30, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_30_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=30, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


# Evaluation Function for the Grover's Oracle (Sudoku 2x2)
def sudoku2x2(quantum_circuit, ansatz='', n_solutions=2, cost=False, gradient=False):
    """ Oracle Approximation
    :param gradient: bool. If True it applies gradient descent over the parameters of the quantum circuit in input
    :param quantum_circuit: quantum circuit approximating a quantum oracle
    :param n_solutions: We are supposed to know the number of solutions of the problem
    :return: float. Reward
    """
    def reward_func():
        gate = quantum_circuit.to_gate(label='Oracle Approx')
        # n_iteration have to be generalised has np.pi/4*(np.sqrt(N/M))
        counts = grover.grover_algo(oracle='approximation', oracle_gate=gate, iterations=2, ancilla=1)
        tot_counts = sum(counts.values())
        best_counts = heapq.nlargest(n_solutions, counts.values())
        solution = [i for i in counts if counts[i] in best_counts]

        def check_constraints(solution):
            values = [int(i) for i in solution]
            return values[0] != values[1] and values[1] != values[3] and values[2] != values[3]
        reward = 0
        for sol in solution:
            if check_constraints(sol):
                if counts[sol] <= tot_counts / 2:
                    reward += counts[sol] / tot_counts
                else:
                    reward += tot_counts/2
        return reward

    def gradient_descent():
        return 0

    if gradient:
        return gradient_descent()
    else:
        if cost:
            return 1000-reward_func()
        else:
            return reward_func()


def sudoku(quantum_circuit, ansatz='', n_solutions=2, cost=False, gradient=False):
    """ Oracle Approximation
    :param gradient: bool. If True it applies gradient descent over the parameters of the quantum circuit in input
    :param quantum_circuit: quantum circuit approximating a quantum oracle
    :param n_solutions: We are supposed to know the number of solutions of the problem
    :return: float. Reward
    """
    def reward_func():
        gate = quantum_circuit.to_gate(label='Oracle Approx')
        # n_iteration have to be generalised has np.pi/4*(np.sqrt(N/M))
        counts = grover.grover_algo(oracle='approximation', oracle_gate=gate, iterations=2, ancilla=1)
        tot_counts = sum(counts.values())

        def check_constraints(solution):
            values = [int(i) for i in solution]
            return values[0] != values[1] and values[1] != values[3] and values[2] != values[3]

        reward = 0
        for sol in counts.keys():
            if check_constraints(sol):
                reward += counts[sol] / tot_counts
        return reward

    def gradient_descent():
        return 0

    if gradient:
        return gradient_descent()
    else:
        if cost:
            return 1 - reward_func()
        else:
            return reward_func()

# Oracle for A. Montanaro Quantum Monte Carlo
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


def h2(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def lih(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = lih_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def h2o(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2o_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def h2o_full(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2o_full_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def vqls_0(quantum_circuit, ansatz='all', cost=False):
    # Instance shown in pennylane demo: https://pennylane.ai/qml/demos/tutorial_vqls/
    problem = vqls_demo
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def vqls_1(quantum_circuit, ansatz='all', cost=False, gradient=False):
    # Define the problem A = c_0 I + c_1 X_1 + c_2 X_2 + c_3 Z_3 Z_4
    problem = vqls_paper

    if cost and gradient:
        raise ValueError('Cannot return both cost and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
