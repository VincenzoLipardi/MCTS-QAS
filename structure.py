import random
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, RXGate, RZGate, HGate, CXGate
from qiskit.quantum_info import Operator


class Circuit:
    def __init__(self, variable_qubits, ancilla_qubits, initialization='h'):
        """
        It builds the first quantum circuit regarding the target problem
        :param variable_qubits: integer. Number of qubits necessary to encode the problem variables
        :param ancilla_qubits: integer. Number of ancilla qubits (hyperparameter)
        """
        # self.variable_qubits = variable_qubits
        # self.ancilla_qubits = ancilla_qubits
        # Initialization of the circuit
        variable_qubits = QuantumRegister(variable_qubits, name='v')
        ancilla_qubits = QuantumRegister(ancilla_qubits, name='a')
        qc = QuantumCircuit(variable_qubits, ancilla_qubits)
        # Initialization of the quantum circuit all qubits in the 0 states (by default) or equal superposition
        if initialization == 'h' or initialization == 'equal_superposition' or initialization == 'hadamard':
            qc.h([qubits for qubits in qc.qubits])

        # Qiskit object
        self.circuit = qc
        # NISQ CONTROL
        self.is_nisq = None

    def building_state(self, quantum_circuit):
        """Given a quantum circuit in qiskit, it creates an instance into the Circuit class"""
        self.circuit = quantum_circuit
        return self

    def nisq_control(self, max_depth):
        """ Check if it is executable on a nisq device. Our definition comes from IBM devices
        :param max_depth: integer. Max quantum circuit depth due to the hardware constraint (NISQ).
        :return: False if the depth is beyond teh Rollout_max depth
        """
        if self.circuit.depth() >= max_depth:
            nisq_control = False
        else:
            nisq_control = True
        self.is_nisq = nisq_control
        return nisq_control

    def evaluation(self, evaluation_function):
        """ Evaluate the circuit through an evaluation function to be given in input together with its variables
        :return: float. Reward
        """
        reward = evaluation_function(self.circuit)
        return reward

    def get_legal_action(self, gate_set, max_depth, prob_choice, stop):
        if stop:
            prob_choice['p'] = 0
        if self.is_nisq is None:
            self.nisq_control(max_depth)
        if not self.is_nisq:
            prob_choice['a'] = 0
            prob_choice['d'] = 50

        keys = list(prob_choice.keys())
        probabilities = list(prob_choice.values())
        probabilities = np.array(probabilities) / sum(probabilities)
        action_str = np.random.choice(keys, p=probabilities)
        action = actions_on_circuit(action_chosen=action_str, gate_set=gate_set)
        if action is not None and callable(action):
            return action, action_str
        else:
            raise NotImplementedError


class GateSet:
    def __init__(self, gate_type='continuous'):
        self.gate_type = gate_type
        if self.gate_type == 'discrete':
            gates = ['s', 'cx', 'h', 't']
        elif self.gate_type == 'continuous':
            gates = ['cx', 'ry', 'rx', 'rz']

        else:
            raise NotImplementedError
        self.pool = gates


def actions_on_circuit(action_chosen, gate_set):
    """
    It modifies the quantum circuit depending on the action required in
    :param gate_set: Universal Gate Set chosen for the application
    :param action_chosen: str. Action chosen to apply on the circuit
    :return: new quantum circuit
    """

    def add_gate(quantum_circuit):
        """ Pick a random one-qubit (two-qubit) gate to add on random qubit(s) """
        qc = quantum_circuit.copy()
        qubits = random.sample([i for i in range(len(qc.qubits))], k=2)
        angle = 2 * math.pi * random.random()
        choice = random.choice(gate_set.pool)
        if choice == 'cx':
            qc.cx(qubits[0], qubits[1])
        elif choice == 'ry':
            qc.ry(angle, qubits[0])
        elif choice == 'rx':
            qc.rx(angle, qubits[0])
        elif choice == 'rz':
            qc.rz(angle, qubits[0])
        elif choice == 'x':
            qc.x(qubits[0])
        elif choice == 'y':
            qc.y(qubits[0])
        elif choice == 'z':
            qc.z(qubits[0])
        elif choice == 'h':
            qc.h(qubits[0])
        elif choice == 't':
            qc.t(qubits[0])
        elif choice == 's':
            qc.s(qubits[0])
        return qc

    def delete_gate(quantum_circuit):
        """ It removes a random gate from the input quantum circuit  """
        qc = quantum_circuit.copy()
        if len(qc.data) < 4:
            return None
        position = random.randint(0, len(qc.data) - 2)
        qc.data.remove(qc.data[position])
        return qc

    def swap(quantum_circuit):
        """ It removes a gate in a random position and replace it with a new gate randomly chosen """

        angle = random.random() * 2 * math.pi
        if len(quantum_circuit.data) > 1:
            position = random.randint(0, len(quantum_circuit.data) - 2)
        else:
            return None

        gate_to_remove = quantum_circuit.data[position]
        gate_to_add_str = random.choice(gate_set.pool[1:])
        gate_to_add = get_gate(gate_to_add_str, angle=angle)
        n_qubits = len(quantum_circuit.qubits)
        qr = QuantumRegister(n_qubits, 'v')
        qc = QuantumCircuit(qr)

        instructions = []
        pos = 0
        two_qubit_gate = 0
        if gate_to_add_str == 'cx':
            two_qubit_gate = 1
        delta = len(gate_to_remove[1]) - 1 - two_qubit_gate     # difference of qubit the new gate is applied to
        for instruction, qargs, cargs in quantum_circuit:
            if pos == position:
                if delta == 1:
                    qargs = [qargs[0]]
                if delta == -1:
                    qargs.append(random.choice(quantum_circuit.qubits))
                instruction = gate_to_add

            instructions.append((instruction, qargs, cargs))
            pos += 1
        qc.data = instructions
        return qc

    def change(quantum_circuit):
        """ It changes the parameter of a gate randomly chosen"""

        qc = quantum_circuit.copy()
        n = len(qc.data)
        position = random.choice([i for i in range(n)])

        check = 0
        while len(qc.data[position][0].params) == 0:
            position = random.choice([i for i in range(n)])
            check += 1
            if check > 2*n:
                return None
        gate_to_change = qc.data[position][0]
        qc.data[position][0].params[0] = gate_to_change.params[0] + random.uniform(0, 0.2)
        return qc

    def stop():
        """ It marks the node as terminal"""
        return 'stop'

    # Define a mapping between input strings and methods
    dict_actions = {'a': add_gate, 'd': delete_gate, 's': swap, 'c': change, 'p': stop}
    action = dict_actions.get(action_chosen, None)
    return action


def get_gate(gate_str, angle=None):
    """
    Get the qiskit object representing the specified gate.
    Returns: qiskit object: Qiskit gate object.
    """

    if gate_str == 'h':
        return HGate()
    elif gate_str == 'cx':
        return CXGate()
    elif gate_str == 'rx':
        return RXGate(theta=angle)
    elif gate_str == 'ry':
        return RYGate(theta=angle)
    elif gate_str == 'rz':
        return RZGate(phi=angle)


def get_action_from_str(input_string, gate_set):
    method_mapping = {
        'a': gate_set.add_gate,
        'd': gate_set.delete_gate,
        's': gate_set.swap,
        'c': gate_set.change,
        'p': gate_set.stop}

    # Choose the method based on the input string
    chosen_method = method_mapping.get(input_string, None)
    if chosen_method is not None and callable(chosen_method):
        return chosen_method
    else:
        return "Invalid method name"


def check_equivalence(qc1, qc2):
    """ It returns a boolean variable. True if the two input quantum circuits are equivalent (same matrix)eqi"""
    Op1 = Operator(qc1)
    Op2 = Operator(qc2)
    return Op1.equiv(Op2)
