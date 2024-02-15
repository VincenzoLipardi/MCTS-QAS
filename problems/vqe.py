import pennylane as qml
from pennylane import numpy as np

# All the models are retrieved from Pennylane


class H2O:
    def __init__(self):
        # Atoms

        self.geometry = np.array([0., 0., 0., 1.63234543, 0.86417176, 0., 3.36087791, 0., 0.]) * 1.88973  # Angstrom to Bohr
        self.symbols = ['H', 'O', 'H']
        self.wires = list(range(0, 8))
        self.dev = qml.device('default.qubit', wires=8)
        self.active_electrons = 4


        # Hamiltonian of the molecule represented
        # Number of qubits needed to perform the quantum simulation
        self.hamiltonian, self.qubits = qml.qchem.molecular_hamiltonian(self.symbols, self.geometry, charge=0, mult=1, basis="sto-6g", active_electrons=4, active_orbitals=4, load_data=True)

        self.hf = qml.qchem.hf_state(self.active_electrons, self.qubits)
    def costFunc(self, params, quantum_circuit=None, ansatz=''):
        """
        Energy of the molecule that we have to minimize
        """
        def circuit(parameters):
            # Standard Ansatz for h2 molecule (Hatree-Fock state when parameter =0)

            # assert len(parameters) == 1

            qml.BasisState(self.hf, wires=self.wires)
            qml.DoubleExcitation(parameters, wires=[0, 1, 2, 3])

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
                circuit(parameters)
            else:
                circuit_input(parameters)
            return qml.expval(self.hamiltonian)
        return cost_fn(parameters=params)

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

        max_iterations = 200
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


class LiH:


    def __init__(self):
        # Atoms

        self.geometry = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.969280527])  # units in Bohr
        self.symbols = ["Li", "H"]
        self.wires = list(range(0, 10))
        self.dev = qml.device('default.qubit', wires=10)
        self.active_electrons = 2


        # Hamiltonian of the molecule represented
        # Number of qubits needed to perform the quantum simulation
        self.hamiltonian, self.qubits = qml.qchem.molecular_hamiltonian(self.symbols, self.geometry, active_electrons=2,
                                                                        active_orbitals=5, basis='sto-6g', load_data=True)
        self.hf = qml.qchem.hf_state(self.active_electrons, self.qubits)

    def costFunc(self, params, quantum_circuit=None, ansatz=''):
        """
        Energy of the molecule that we have to minimize
        """
        def circuit(parameters):
            # Standard Ansatz for h2 molecule (Hatree-Fock state when parameter =0)

            # assert len(parameters) == 1

            qml.BasisState(self.hf, wires=self.wires)
            qml.DoubleExcitation(parameters, wires=[0, 1, 2, 3])

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
                circuit(parameters)
            else:
                circuit_input(parameters)
            return qml.expval(self.hamiltonian)
        return cost_fn(parameters=params)

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

        max_iterations = 200
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

            # assert len(parameters) == 1

            qml.BasisState(self.hf, wires=self.wires)
            qml.DoubleExcitation(parameters, wires=[0, 1, 2, 3])

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
                circuit(parameters)
            else:
                circuit_input(parameters)
            return qml.expval(self.hamiltonian)
        return cost_fn(parameters=params)

    def getReward(self, params, quantum_circuit=None, ansatz=''):
        return -self.costFunc(params, quantum_circuit, ansatz)

    def benchmark(self):
        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        theta = np.array(0.0, requires_grad=True)
        # store the values of the cost function
        energy = [self.costFunc(theta)]

        # store the values of the circuit parameter
        angle = [theta]

        max_iterations = 100
        conv_tol = 1e-06

        for n in range(max_iterations):
            theta, prev_energy = opt.step_and_cost(self.costFunc, theta)

            energy.append(self.costFunc(theta))
            angle.append(theta)

            conv = np.abs(energy[-1] - prev_energy)

            """if n % 2 == 0:
                print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")"""

            if conv <= conv_tol:
                break

        # print("\n" f"Final value of the ground-state energy = {energy[-1]:.8f} Ha")
        # print("\n" f"Optimal value of the circuit parameter = {angle[-1]:.4f}")
        return energy

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

        max_iterations = 200
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
