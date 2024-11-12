import pennylane as qml
from pennylane import numpy as np
import math



class VQLS:
    # https://pennylane.ai/qml/demos/tutorial_vqls/
    def __init__(self, c):
        """ Solving linear equation through a quantum variational approach A|x> = |b>.
        :param c: list. List of coefficients of the linear combination A = c_0 A_0 + c_1 A_1 ...
        """
        self.n_qubits = 4  # Number of system qubits.
        self.n_shots = 10 ** 6  # Number of quantum measurements.
        self.tot_qubits = self.n_qubits + 1  # Addition of an ancillary qubit.
        self.ancilla_idx = self.n_qubits  # Index of the ancillary qubit (last position).

        # Hyperparameters of the standard classical optimization technique
        self.steps = 30  # Number of optimization steps
        self.eta = 0.8  # Learning rate
        self.q_delta = 0.001  # Initial spread of random quantum params
        self.rng_seed = 0  # Seed for random number generator

        # Array c of the coefficients of the linear combination: A = c_0 A_0 + c_1 A_1 + ... + c_l A_l
        self.c = np.array(c)
        # Inizialization of the quantum devices in pennylane for the evaluation of mu's and the solution x
        self.dev_mu = qml.device("lightning.qubit", wires=self.tot_qubits)
        self.dev_x = qml.device("lightning.qubit", wires=self.n_qubits, shots=self.n_shots)

    def U_b(self):
        """Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>."""
        for idx in range(self.n_qubits):
            qml.Hadamard(wires=idx)

    def CA(self, idx):
        """Controlled versions of the unitary components A_l of the problem matrix A (for the Hadamard test)."""
        if idx == 0:
            # Identity operation
            pass
        elif idx == 1:
            # X_0
            qml.CNOT(wires=[self.ancilla_idx, 0])
            if len(self.c) == 3:
                qml.CZ(wires=[self.ancilla_idx, 1])

        elif idx == 2:
            # X_1
            if len(self.c) == 4:
                qml.CNOT(wires=[self.ancilla_idx, 1])
            elif len(self.c) == 3:
                qml.CNOT(wires=[self.ancilla_idx, 0])


        elif idx == 3:
            if len(self.c) == 4:
                # Z_2 Z_3
                qml.CZ(wires=[self.ancilla_idx, 2])
                qml.CZ(wires=[self.ancilla_idx, 3])
            else:
                pass

    def variational_block(self, params, quantum_circuit, ansatz):
        """Variational circuit mapping the ground state |0> to the ansatz state |x>."""
        if quantum_circuit is None:
            # We first prepare an equal superposition of all the states of the computational basis.
            for idx in range(self.n_qubits):
                qml.Hadamard(wires=idx)
            # Let's define our guess with a minimal variational circuit.
            # A single rotation qubit is added on each qubit on which angles are0 provided len(angles)<n_qubits
            for idx, element in enumerate(params):
                qml.RY(element, wires=idx)
            qml.CNOT(wires=[1, 0])
            qml.CNOT(wires=[2, 3])

        else:

            def circuit(parameters):
                parameters = parameters
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
                return qml

            return circuit(parameters=params)

    def constructCirc(self, quantum_circuit, ansatz):
        @qml.qnode(self.dev_mu, interface="autograd")
        def local_hadamard_test(params, l=None, lp=None, j=None, part=None):
            # First Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.ancilla_idx)

            # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
            # phase gate.
            if part == "Im" or part == "im":
                qml.PhaseShift(-math.pi / 2, wires=self.ancilla_idx)

            # Variational circuit generating a guess for the solution vector |x>
            self.variational_block(params=params, quantum_circuit=quantum_circuit, ansatz=ansatz)

            # Controlled application of the unitary component A_l of the problem matrix A.
            self.CA(l)

            # Adjoint of the unitary U_b associated to the problem vector |b>.
            # In this specific example Adjoint(U_b) = U_b.
            self.U_b()

            # Controlled Z operator at position j. If j = -1, apply the identity.
            if j != -1:
                qml.CZ(wires=[self.ancilla_idx, j])

            # Unitary U_b associated to the problem vector |b>.
            self.U_b()

            # Controlled application of Adjoint(A_lp).
            # In this specific example Adjoint(A_lp) = A_lp.
            self.CA(lp)

            # Second Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=self.ancilla_idx)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=self.ancilla_idx))

        def mu(params, l=None, lp=None, j=None):
            """Generates the coefficients to compute the "local" cost function C_L."""
            mu_real = local_hadamard_test(params, l=l, lp=lp, j=j, part="Re")
            mu_imag = local_hadamard_test(params, l=l, lp=lp, j=j, part="Im")
            return mu_real + 1.0j * mu_imag

        def psi_norm(params):
            """Returns the normalization constant <psi|psi>, where |psi> = A |x>."""
            norm = 0.0
            for l in range(0, len(self.c)):
                for lp in range(0, len(self.c)):
                    norm = norm + self.c[l] * np.conj(self.c[lp]) * mu(params, l, lp, -1)
            return abs(norm)

        def cost_loc(params):
            """Local version of the cost function. Tends to zero when A|x> is proportional to |b>."""
            mu_sum = 0.0

            for l in range(0, len(self.c)):
                for lp in range(0, len(self.c)):
                    for j in range(0, self.n_qubits):
                        mu_sum = mu_sum + self.c[l] * np.conj(self.c[lp]) * mu(params, l, lp, j)

            mu_sum = abs(mu_sum)
            # Cost function C_L
            return 0.5 - 0.5 * mu_sum / (self.n_qubits * psi_norm(params))

        return cost_loc

    def costFunc(self, params, quantum_circuit=None, ansatz=''):
        cost = self.constructCirc(quantum_circuit, ansatz=ansatz)
        return cost(params)

    def getReward(self, params, quantum_circuit=None, ansatz=''):
        return np.exp(-10*self.costFunc(params, quantum_circuit, ansatz))

    def gradient_descent(self, quantum_circuit):
        opt = qml.AdamOptimizer()
        parameters = get_parameters(quantum_circuit)
        theta = np.array(parameters, requires_grad=True)

        # store the values of the cost function

        def prova(params):
            return self.costFunc(params=params, quantum_circuit=quantum_circuit, ansatz='')

        cost = [prova(theta)]

        # store the values of the circuit parameter
        angle = [theta]

        max_iterations = 200
        conv_tol = 1e-08  # default -06

        for n in range(max_iterations):
            theta, prev_energy = opt.step_and_cost(prova, theta)
            cost.append(prova(theta))
            angle.append(theta)

            conv = np.abs(cost[-1] - prev_energy)

            """if n % 2 == 0:
                print(f"Step = {n},  Cost = {cost[-1]:.8f}")"""

            if conv <= conv_tol:
                print('Landscape is flat')
                break
        print("\n" f"Final value of the cost function is  {cost[-1]:.4f} ")
        return cost


    def getClassicalSolution(self):
        Id = np.identity(2)
        Z = np.array([[1, 0], [0, -1]])
        X = np.array([[0, 1], [1, 0]])

        A_0 = np.identity(2 ** self.n_qubits)
        A_1 = np.kron(X, np.kron(Id, np.kron(Id, Id)))
        A_2 = np.kron(Id, np.kron(X, np.kron(Id, Id)))
        A_3 = np.kron(Id, np.kron(Id, np.kron(Z, Z)))

        A_num = self.c[0] * A_0 + self.c[1] * A_1 + self.c[2] * A_2 + self.c[3] * A_3
        b = np.ones(2 ** self.n_qubits) / np.sqrt(2 ** self.n_qubits)
        # print("A = \n", A_num)
        # print("b = \n", b)

        A_inv = np.linalg.inv(A_num)
        x = np.dot(A_inv, b)

        c_probs = (x / np.linalg.norm(x)) ** 2

        return c_probs

    def getQuantumSolution(self, parameters, quantum_circuit=None, ansatz=''):

        @qml.qnode(self.dev_x)
        def prepare_and_sample(weights):
            # Variational circuit generating a guess for the solution vector |x>
            self.variational_block(weights, quantum_circuit=quantum_circuit, ansatz=ansatz)

            # We assume that the system is measured in the computational basis.
            # then sampling the device will give us a value of 0 or 1 for each qubit (n_qubits)
            # this will be repeated for the total number of shots provided (n_shots)
            return qml.sample()

        raw_samples = prepare_and_sample(parameters)
        # convert the raw samples (bit strings) into integers and count them
        samples = []
        for sam in raw_samples:
            samples.append(int("".join(str(bs) for bs in sam), base=2))

        q_probs = np.bincount(samples) / self.n_shots

        return q_probs


def get_parameters(quantum_circuit):
    parameters = []
    # Iterate over all gates in the circuit
    for instr, qargs, cargs in quantum_circuit.data:

        # Extract parameters from gate instructions
        if len(instr.params) > 0:
            parameters.append(instr.params[0])
    return parameters


# SYSTEMS OF LINEAR EQUATIONS
vqls_demo = VQLS(c=[1, 0.2, 0.2])
vqls_paper = VQLS(c=[1, 0.1, 0.1, 0.2])
