import save_in_file as sif
from evaluation_functions import h2, vqls_1

BUDGET = [1000]
M = [3, 4, 5]
eval_func = [h2, vqls_1]

for f in eval_func:
    for b in BUDGET:
        for m in M:
            sif.data(evaluation_function=f, variable_qubits=4, ancilla_qubits=0, gate_set='continuous', budget=b, max_branches=m, verbose=True)
            sif.plot_Cost_Func(evaluation_function=f, gate_set='continuous', budget=b, max_branches=m)
