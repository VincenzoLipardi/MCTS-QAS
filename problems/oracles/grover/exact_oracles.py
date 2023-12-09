import qiskit.quantum_info as qi


def XOR(qc, a, b, output):
    """
    It checks the values of the variable qubits a and b and output 1 only if they satisfy the sudoku constraints
    (different values over the same row and column)
    :param qc: quantum circuit
    :param a: qubit. First variable we are comparing
    :param b: qubit. Second variable we are comparing
    :param output: qubit. Ancilla qubit on which we record the result of the comparison between a and b
    """
    qc.cx(a, output)
    qc.cx(b, output)


def sudoku_oracle(qc, clause_list, clause_qubits, output_qubit, verbose=False):
    """
    :param qc: Initial qc implementing the Grover's algo. Then encoding an equal superposition quantum state.
    :param clause_list: list of lists. List of the variables that have to be different in the sudoku
    :param clause_qubits: qubits encoding the local constraint information (1 if satisfied 0 otherwise).
    :param output_qubit: Qubit encoding the global constraint information. It is one if all of them are satisfied 0 ow)
    :param verbose: Boolean. True in case you need to check what the function does by printing information
    """
    # Compute clauses
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[clause_list.index(clause)])

    # Flip 'output' bit if all clauses are satisfied: multi-controlled Toffoli gate
    qc.mct(clause_qubits, output_qubit)
    # Uncompute clauses to reset clause-checking bits to 0
    for clause in clause_list:
        XOR(qc, clause[0], clause[1], clause_qubits[clause_list.index(clause)])
    if verbose:
        print('Exact Oracle:\n', qc.draw(), '\nExact Oracle - Depth:', qc.depth(), '\nExact Oracle - Number of pool:',
              len(qc.data), '\nExact Oracle - Number of qubits:', len(qc.qubits))
        print('Exact Oracle - Matrix:\n', qi.Operator(qc).to_instruction())



