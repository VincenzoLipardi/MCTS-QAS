import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit


class Maxcut:
    # Modified from https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html
    def __init__(self):
        """
        The cost Hamiltonian:
        C_alpha = 1/2 * (1 - Z_j Z_k)
        (j, k) is an edge of the graph
        """
        self.n_qubits = 7
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=1)
        self.dev_train = qml.device("default.qubit", wires=self.n_qubits)
        self.n_samples = 100
        self.pauli_z = [[1, 0], [0, -1]]
        self.pauli_z_2 = np.kron(self.pauli_z, self.pauli_z, requires_grad=False)
        self.graph = [(0, 1), (0, 2),  (2, 3), (1, 4), (2, 4), (0, 5),  (3, 6), (1, 6)]

    def costFunc(self, params, quantum_circuit=None, ansatz=''):
        @qml.qnode(self.dev_train)
        def circuit_input(parameters, edge=None):
            i = 0
            for instr, qubits, clbits in quantum_circuit.data:
                name = instr.name.lower()
                if name == "rx":
                    if ansatz == 'all':
                        qml.RX(instr.params[0], wires=qubits[0].index)
                    else:
                        qml.RX(parameters[i], wires=qubits[0].index)
                        i += 1
                elif name == "ry":
                    if ansatz == 'all':
                        qml.RY(instr.params[0], wires=qubits[0].index)
                    else:
                        qml.RY(parameters[i], wires=qubits[0].index)
                        i += 1
                elif name == "rz":
                    if ansatz == 'all':
                        qml.RZ(instr.params[0], wires=qubits[0].index)
                    else:
                        qml.RZ(parameters[i], wires=qubits[0].index)
                        i += 1
                elif name == "h":
                    qml.Hadamard(wires=qubits[0].index)
                elif name == "cx":
                    qml.CNOT(wires=[qubits[0].index, qubits[1].index])

            if edge is None:
                return qml.sample()

            return qml.expval(qml.Hermitian(self.pauli_z_2, wires=edge))
        neg_obj = 0
        for edge in self.graph:
            neg_obj -= 0.5*(1-circuit_input(params, edge))
        return neg_obj

    def getReward(self, params, quantum_circuit=None, ansatz=''):
        return -self.costFunc(params, quantum_circuit, ansatz)
    
    def gradient_descent(self, quantum_circuit):
        opt = qml.AdamOptimizer()
        parameters = get_parameters(quantum_circuit)
        theta = np.array(parameters, requires_grad=True)
        # store the values of the cost function

        def prova(params):
            return self.costFunc(params=params, quantum_circuit=quantum_circuit, ansatz='')
        energy = [prova(theta)]


        # store the values of the circuit parameter
        angle = [theta]

        max_iterations = 500
        conv_tol = 1e-08    # default -06

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

maxcut_class = Maxcut()

