import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit


class H2:
    # https://pennylane.ai/qml/demos/tutorial_vqe/
    def __init__(self, name='', geometry=None):
        # Atoms
        self.symbols = ["H", "H"]
        self.wires = [0, 1, 2, 3]
        self.dev = qml.device('default.qubit', wires=4)

        # Position of the Hydrogen atoms
        if geometry is None:
            self.geometry = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
        else:
            self.geometry = geometry

        def hamiltonian_preparation(name):
            if name == 'pyscf':
                h, q = qml.qchem.molecular_hamiltonian(
                    self.symbols, self.geometry, charge=0, mult=1, basis='sto-3g', method='pyscf', active_electrons=2, active_orbitals=2)
            else:
                h, q = qml.qchem.molecular_hamiltonian(self.symbols, self.geometry)
            return h, q

        # Hamiltonian of the molecule represented
        # Number of qubits needed to perform the quantum simulation
        self.hamiltonian, self.qubits = hamiltonian_preparation(name)
        self.hf = qml.qchem.hf_state(len(self.symbols), self.qubits)

    def costFunc(self, params, quantum_circuit=None, ansatz=''):
        """
        Energy of the molecule that we have to minimize
        """

        def circuit(parameters):
            # Standard Ansatz for h2 molecule (Hatree-Fock state when parameter =0)
            assert len(parameters) == 1

            qml.BasisState(self.hf, wires=self.wires)
            qml.DoubleExcitation(parameters[0], wires=[0, 1, 2, 3])

        def circuit_input(parameters):
            qml.BasisState(self.hf, wires=self.wires)
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
                        qml.RZ(parameters[i], wires=qubits[0].index)
                    else:
                        qml.RZ(parameters[i], wires=qubits[0].index)
                        i += 1
                elif name == "h":
                    qml.Hadamard(wires=qubits[0].index)
                elif name == "cx":
                    qml.CNOT(wires=[qubits[0].index, qubits[1].index])

        @qml.qnode(self.dev, interface="autograd")
        def cost_fn(parameters):
            if quantum_circuit is None:
                circuit(parameters, wires=range(self.qubits))
            else:
                circuit_input(parameters)
            return qml.expval(self.hamiltonian)
        return cost_fn(parameters=params)

    def getReward(self, params, quantum_circuit=None, ansatz=''):
        return -self.costFunc(params, quantum_circuit, ansatz)


