import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit


class QAOA:
    # Modified from https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut.html
    def __init__(self):
        """
        The cost Hamiltonian:
        C_alpha = 1/2 * (1 - Z_j Z_k)
        (j, k) is an edge of the graph
        """
        self.n_qubits = 7
        self.dev = qml.device("lightning.qubit", wires=self.n_qubits, shots=1)
        self.dev_train = qml.device("lightning.qubit", wires=self.n_qubits)
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

qaoa_class = QAOA()
# Class works - Implement Gradient Descent on the parameters and check if it return the right solution
"""qc = QuantumCircuit(7)

for i in range(len(qc.qubits)):
    qc.h(i)
    if i == 3:
        qc.cx(2, 3)
        qc.u(0.3, 0.8, 0.3, 2)
        qc.u(1, 0.2, 1, 3)
        qc.cx(3, 2)
        qc.cx(2, 3)
    else:
        qc.u(2, 0.9, 1, i)
print(qc)
print(qaoa_class.costFunc(params=[0.1], quantum_circuit=qc))"""
