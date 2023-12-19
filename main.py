import save_in_file as sif
from problems.evaluation_functions import h2, vqls_1, sudoku2x2

BUDGET = [1000, 2000, 5000, 10000, 50000, 100000]
M = [3, 5, 7, 10, 15]
eval_func = [h2, sudoku2x2, vqls_1]
iterations = 10


for f in eval_func:
    for b in BUDGET:
        for m in M:
            for i in range(iterations):
                sif.data(evaluation_function=f, variable_qubits=4, ancilla_qubits=0, gate_set='continuous', budget=b, max_branches=m, roll_out_steps=2, iteration=i, verbose=True)
                if f == sudoku2x2:
                    sif.plot_oracle(evaluation_function=f, gate_set='continuous', budget=b, max_branches=m)
                else:
                    sif.plot_Cost_Func(evaluation_function=f, gate_set='continuous', budget=b, max_branches=m)
                    sif.plot_Reward_Func(evaluation_function=f, gate_set='continuous', budget=b, max_branches=m)
