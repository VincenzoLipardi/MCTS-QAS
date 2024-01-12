import numpy as np
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from problems.oracles.grover import exact_oracles


def diffuser(nqubits):
    quantum_circuit = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-pool)
    for qubit in range(nqubits):
        quantum_circuit.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-pool)
    for qubit in range(nqubits):
        quantum_circuit.x(qubit)
    # Do multi-controlled-Z gate
    quantum_circuit.h(nqubits - 1)
    # multi-controlled-toffoli
    quantum_circuit.mct(list(range(nqubits - 1)), nqubits - 1)
    quantum_circuit.h(nqubits - 1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        quantum_circuit.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        quantum_circuit.h(qubit)
    # We will return the diffuser as a gate
    U_s = quantum_circuit.to_gate()
    U_s.name = "Diffuser"
    return U_s


def grover_algo(oracle, iterations, ancilla=1, oracle_gate=None):
    clause_list = [[0, 1], [0, 2], [1, 3], [2, 3]]
    # Create separate registers to name bits:
    # vi: assignement in the i-th position
    var_qubits = QuantumRegister(4, name='v')
    # ci: check of the i-th contraint condition
    clause_qubits = QuantumRegister(4, name='c')
    # Output qubit will be 1 only if all the constraint are satisfied
    output_qubit = QuantumRegister(1, name='out')
    # The output qubit will be called ancilla in the case of approximate oracle, and it can be higher than one
    ancilla_qubits = QuantumRegister(ancilla, name='a')
    # Classical bits storing the results
    cbits = ClassicalRegister(4, name='cbits')

    if oracle == 'exact':
        # Quantum Circuit with n_v + n_c + 1 qubits
        qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit, cbits)
        # Initialize 'out0' in circuit |->
        qc.initialize([1 / np.sqrt(2), -1 / np.sqrt(2)], output_qubit)

    elif oracle == 'approximation':
        # Quantum Circuit with n_variables + n_ancilla qubits. n_ancilla>=1
        qc = QuantumCircuit(var_qubits, ancilla_qubits, cbits)
    else:
        raise 'ERROR: the oracle chosen is not valid'

    # Initialize qubits in circuit |s> equal superposition
    qc.h(var_qubits)
    qc.barrier()  # for visual separation
    # Iterative reflection through the oracle and the diffuser
    # optimal number of iteration : sqrt(number_of_items/number_of_solutions)*pi/4
    for i in range(iterations):
        # Apply our oracle
        if oracle == 'exact':
            exact_oracles.sudoku_oracle(qc, clause_list, clause_qubits, output_qubit)
        elif oracle == 'approximation':
            qc.append(oracle_gate, range(len(qc.qubits)))
        qc.barrier()  # for visual separation
        # Apply the diffuser
        qc.append(diffuser(4), [0, 1, 2, 3])

    # Measure the variable qubits
    qc.measure(var_qubits, cbits)
    '''print('Grover Circuit:\n', qc.draw(fold=-1), '\nGrover - Depth:', qc.depth(),
          '\nGrover - Number of pool:', len(qc.data), '\nGrover - Number of qubits:', len(qc.qubits))'''

    # Simulate results
    qasm_simulator = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, qasm_simulator)
    result = qasm_simulator.run(transpiled_qc, shots=1000).result()
    counts = result.get_counts()
    # print(Rollout_max(result.get_counts(), key=counts.get))
    return counts

# print(grover_algo(oracle='exact', oracle_gate=None, iterations=2, ancilla=1))