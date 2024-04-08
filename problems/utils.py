from qiskit import Aer, transpile, execute
from qiskit.quantum_info import Operator, Statevector
from structure import GateSet, Circuit, actions_on_circuit
from qiskit import QuantumCircuit, Aer, execute
from itertools import product
#from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
import numpy as np


def expectation_value_qiskit(state, operator):
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.least_busy(operational=True, simulator=False)
    session = Session(service=service, backend=backend.name)
    estimator = Estimator(session=session)
    exp_val = estimator.run(state, operator).result().values
    return exp_val


def stabilizer_renyi_entropy(circuit, alpha=2):
    n = len(circuit.qubits)     # Number of qubits
    d = 2**n
    pauli_gates = ['I', 'X', 'Y', 'Z']
    gate_combinations = product(pauli_gates, repeat=n)      # All the pauli combination on all the qubits
    A = 0
    op = QuantumCircuit(n)

    for combination in gate_combinations:
        # Apply Pauli gates according to the combination
        for qubit, gate in enumerate(combination):
            if gate == 'X':
                op.x(qubit)
            elif gate == 'Y':
                op.y(qubit)
            elif gate == 'Z':
                op.z(qubit)
            else:
                pass

        # Calculate expectation value
        op = Operator(op)
        exp_val = Statevector(circuit).expectation_value(op).real
        A += ((1/d) * exp_val**2)**alpha
        # Recalculate the operator
        op = QuantumCircuit(n)

    entropy = (1 / (1 - alpha)) * np.log(A) - np.log(d)

    return entropy


def create_random_qc(n_qubits, gate_number):
    circuit = Circuit(variable_qubits=n_qubits, ancilla_qubits=0, initialization=None)
    circuit_qiskit = circuit.circuit
    for i in range(gate_number):
        action = actions_on_circuit(action_chosen='a', gate_set=GateSet('discrete'))
        circuit_qiskit = action(circuit_qiskit)
    return circuit_qiskit


def optimize_qc(circuit):
    # Transpile the circuit for the target backend
    optimized_circuit = transpile(circuit, basis_gates=['cx', 'h', 't', 's'])
    return optimized_circuit


def get_matrix(quantum_circuit):
    simulator = Aer.get_backend("unitary_simulator")
    job = execute(quantum_circuit, backend=simulator)
    result = job.result()
    unitary = result.get_unitary()      # .data to get the numpy array object
    return unitary
