from qiskit import QuantumCircuit
from evaluation_functions import vqls_1, sudoku2x2
from vqe import LiH, H2O
qc = QuantumCircuit(8, 0)
for i in range(8):
    qc.h(i)
    qc.ry(0.2, i)


value = H2O().gradient_descent(quantum_circuit=qc)


print(value)


