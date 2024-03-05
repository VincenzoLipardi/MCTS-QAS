import pennylane as qml
from qiskit.quantum_info import state_fidelity, Statevector
import numpy as np
from qiskit_algorithms import optimizers, gradients
from qiskit import QuantumCircuit, Aer, execute
import pandas as pd
from scipy import optimize


def naive(matrix1, matrix2):
    l = 0
    n = len(matrix1[:][0])
    for i in range(n):
        for j in range(n):
            l += np.abs(matrix1[i][j]-matrix2[i][j])
    normalization_factor = 4 * n**2
    return l/normalization_factor


class Fidelity:
    def __init__(self, qubits, gates, difficulty):
        filename = 'problems/oracles/dataset/random_circuit_qubit_' + str(qubits) + '_gates_' + str(gates)
        df = pd.read_pickle(filename + '.pkl')
        if difficulty == "easy":
            choice = min(df['sre'])
        elif difficulty == "hard":
            choice = min(df['sre'])
        else:
            raise NotImplementedError
        self.index = df[df['sre'] == choice].index[0]
        self.circuit_bench = df['quantum_circuit'][self.index]



    def reward(self, quantum_circuit):
        return state_fidelity(Statevector(quantum_circuit), Statevector(self.circuit_bench))


    def cost(self, quantum_circuit):
        return -self.reward(quantum_circuit)


    def gradient_descent(self, quantum_circuit):
        parameters = get_parameters(quantum_circuit)

        def costFunc(params):
            qc = QuantumCircuit(len(quantum_circuit.qubits))
            i = 0
            for instr, qubits, clbits in quantum_circuit.data:
                name = instr.name.lower()
                if name == "rx":
                    qc.rx(params[i], qubits[0].index)
                    i += 1
                elif name == "ry":
                    qc.ry(params[i], qubits[0].index)
                    i += 1
                elif name == "rz":
                    qc.rz(params[i], qubits[0].index)
                    i += 1
                elif name == "h":
                    qc.h(qubits[0].index)
                elif name == "cx":
                    qc.cx(qubits[0].index, qubits[1].index)
            return state_fidelity(Statevector(qc), Statevector(self.circuit_bench))

        print(costFunc(parameters))

        a = optimize.minimize(costFunc, np.array(parameters))
        parameters = a.jac


        return parameters, costFunc(parameters)


class CircuitRegeneration:
    def __init__(self):
        df = pd.read_pickle('regeneration_clifford_stats.pkl')
        self.circuits = df['quantum_circuit']
        self.operators = df['operator']
        hardest = max(df['t_gate'])
        self.hardest_index = df[df['t_gate'] == hardest].index[0]
        self.matrix_goal = df['operator'][self.hardest_index].data


    def reward(self, quantum_circuit):
        matrix_test = get_matrix(quantum_circuit).data
        distance = naive(matrix_test, self.matrix_goal)
        return 1 - distance

    def costFunc(self, params, quantum_circuit):
        """
        Energy of the molecule that we have to minimize
        """

        def circuit_input(parameters):

            i = 0
            for instr, qubits, clbits in quantum_circuit.data:
                name = instr.name.lower()
                if name == "rx":
                    qml.RX(parameters[i], wires=qubits[0].index)
                    i += 1
                elif name == "ry":
                    qml.RY(parameters[i], wires=qubits[0].index)
                    i += 1
                elif name == "rz":
                    qml.RZ(parameters[i], wires=qubits[0].index)
                    i += 1
                elif name == "h":
                    qml.Hadamard(wires=qubits[0].index)
                elif name == "cx":
                    qml.CNOT(wires=[qubits[0].index, qubits[1].index])
        n_qubits = len(quantum_circuit.qubits)
        dev = qml.device('default.qubit', wires=range(n_qubits))
        @qml.qnode(dev)
        def cost_fn(parameters):
            circuit_input(parameters)
            return qml.density_matrix(wires=range(n_qubits))

        return naive(cost_fn(parameters=params), self.matrix_goal)



    def gradient_descent(self, quantum_circuit):
        opt = qml.AdamOptimizer()
        parameters = get_parameters(quantum_circuit)
        theta = np.array(parameters, requires_grad=True)

        # store the values of the cost function

        def prova(params):
            return self.costFunc(params=params, quantum_circuit=quantum_circuit)

        energy = [prova(theta)]

        # store the values of the circuit parameter
        angle = [theta]

        max_iterations = 200
        conv_tol = 1e-08  # default -06

        for n in range(max_iterations):
            theta, prev_energy = opt.step_and_cost(prova, theta)
            energy.append(prova(theta))
            angle.append(theta)

            conv = np.abs(energy[-1] - prev_energy)

            if n % 2 == 0:
                print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

            if conv <= conv_tol:
                print('Landscape is flat')
                break

        # print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
        # print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")
        return energy


def get_parameters(quantum_circuit):
    parameters = []
    # Iterate over all gates in the circuit
    for instr, qargs, cargs in quantum_circuit.data:

        # Extract parameters from gate instructions
        if len(instr.params) > 0:
            parameters.append(instr.params[0])
    return parameters


def get_matrix(quantum_circuit):
    simulator = Aer.get_backend("unitary_simulator")
    job = execute(quantum_circuit, backend=simulator)
    result = job.result()
    unitary = result.get_unitary()      # .data to get the numpy array object
    return unitary
