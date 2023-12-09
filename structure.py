import random
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate, RXGate, RZGate, HGate, CXGate

class Circuit:
    def __init__(self, variable_qubits, ancilla_qubits, initialization=None):
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
        if initialization == 'h' or 'equal_superposition' or 'hadamard':
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
        :return: False if the depth is beyond teh max depth
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

    def get_legal_action(self, gate_set, max_depth, prob_choice):
        if self.is_nisq is None:
            self.nisq_control(max_depth)
        if not self.is_nisq:
            prob_choice['a'] = 0
        keys = list(prob_choice.keys())
        probabilities = list(prob_choice.values())
        probabilities = np.array(probabilities) / sum(probabilities)
        action_str = np.random.choice(keys, p=probabilities)
        action = get_action_from_str(action_str, gate_set=gate_set)
        return action


def get_action_from_str(input_string, gate_set):

    # Define a mapping between input strings and methods
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


class GateSet:
    def __init__(self, gate_type='continuous'):
        self.gate_type = gate_type
        if self.gate_type == 'discrete':
            gates = ['s', 'cx', 'h', "t"]
        elif self.gate_type == 'continuous':
            gates = ['cx', 'ry', 'rx', 'rz']

        else:
            raise NotImplementedError
        self.pool = gates

    def add_gate(self, quantum_circuit):
        """
        Pick a random one-qubit (two-qubit) gate to add on random qubit(s)
        :param quantum_circuit: quantum circuit to modify
        :return: new quantum circuit
        """
        qc = quantum_circuit.copy()
        qubits = random.sample([i for i in range(len(qc.qubits))], k=2)
        angle = 2 * math.pi * random.random()
        choice = random.choice(self.pool)
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

    @staticmethod
    def delete_gate(quantum_circuit):
        """
        It removes a random gate from the input quantum circuit
        :param quantum_circuit: quantum circuit to modify
        :return: new quantum circuit
        """
        qc = quantum_circuit.copy()
        position = random.randint(0, len(qc.data) - 1)
        qc.data.remove(qc.data[position])
        return qc

    def swap(self, quantum_circuit):
        " It removes a gate in a random position and replace it with a new gate randomly chosen"
        qc = quantum_circuit.copy()
        angle = random.random() * 2 * math.pi
        if len(qc.data) - 1 > 0:
            position = random.randint(0, len(qc.data) - 2)
        gate_to_remove = qc.data[position][0]
        gate_to_add = random.choice(list(map(get_gate, self.pool)))(angle)
        while gate_to_add.name == gate_to_remove.name:
            gate_to_add = random.choice(list(map(get_gate, self.pool)))(angle)
        if gate_to_add.name == 'cx':
            n_qubits = 2
        else:
            n_qubits = 1
        lenght = len(qc.data[position][1])
        if lenght == n_qubits:
            element_to_remove = list(qc.data[position])
            element_to_remove[0] = gate_to_add
            element_to_add = tuple(element_to_remove)
            qc.data[position] = element_to_add
        elif lenght > n_qubits:
            element_to_remove = list(qc.data[position])
            element_to_remove[0] = gate_to_add
            element_to_remove[1] = [random.choice(qc.data[position][1])]
            element_to_add = tuple(element_to_remove)
            qc.data[position] = element_to_add
        elif lenght < n_qubits:
            element_to_remove = list(qc.data[position])
            element_to_remove[0] = gate_to_add
            qubits_available = []
            for q in qc.qubits:
                if [q] != qc.data[position][1]:
                    qubits_available.append(q)
            qubits_ = [qc.data[position][1], random.choice(qubits_available)]
            random.shuffle(qubits_)
            element_to_remove[1] = qubits_
            element_to_add = tuple(element_to_remove)
            qc.data[position] = element_to_add
        return quantum_circuit

    def change(self, quantum_circuit):
        qc = quantum_circuit.copy()
        position = random.choice([i for i in range(len(quantum_circuit.data))])
        while len(quantum_circuit.data[position][0].params) > 0:
            position = random.choice([i for i in range(len(quantum_circuit.data))])
        gate_to_mute = quantum_circuit.data[position][0]

        qc.data[position][0].params[0] = gate_to_mute.params[0] + random.uniform(0, 0.2)

        return qc

    def stop(self, quantum_circuit):
        return 'stop'


def get_gate(gate_str):
    """
    Get the qiskit object representing the specified gate.
    Returns: qiskit object: Qiskit gate object.
    """

    if gate_str == 'h':
        return HGate
    elif gate_str == 'cx':
        return CXGate
    elif gate_str == 'rx':
        return RXGate
    elif gate_str == 'ry':
        return RYGate
    elif gate_str == 'rz':
        return RZGate




