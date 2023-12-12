import save_in_file as sif
from problems.evaluation_functions import h2, vqls_1, sudoku2x2

BUDGET = [1000]
M = [3, 4, 5]
eval_func = [sudoku2x2]

for f in eval_func:
    for b in BUDGET:
        for m in M:
            sif.data(evaluation_function=f, variable_qubits=5, ancilla_qubits=0, gate_set='continuous', budget=b, max_branches=m, verbose=True)
            sif.plot_Cost_Func(evaluation_function=f, gate_set='continuous', budget=b, max_branches=m)
