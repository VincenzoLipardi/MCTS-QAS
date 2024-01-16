import save_in_file as sif
from problems.evaluation_functions import h2, vqls_1, sudoku2x2

BUDGET = [1000, 2000, 5000, 10000, 50000]
M = [3, 5, 7, 10, 15]
eval_func = [h2, vqls_1, sudoku2x2]
iterations = 3
RT = 'classic'

for f in eval_func:
    for b in BUDGET:
        for m in M:
            for i in range(iterations):
                if f == sudoku2x2:
                    sif.data(evaluation_function=f, variable_qubits=5, ancilla_qubits=0, gate_set='continuous',
                             budget=b, max_branches=m, max_depth=20, roll_out_steps=0, iteration=i, verbose=True)
                else:
                    sif.data(evaluation_function=f, variable_qubits=4, ancilla_qubits=0, gate_set='continuous', rollout_type=RT, budget=b, max_branches=m, roll_out_steps=2, iteration=i, max_depth=20, verbose=True)

            if f == sudoku2x2:
                sif.plot_oracle(evaluation_function=f, gate_set='continuous', budget=b, max_branches=m)
            else:
                # sif.plot_Cost_Func(evaluation_function=f, gate_set='continuous', budget=b, max_branches=m,roll_out_steps=2, rollout_type=RT)
                sif.plot_cost_all_iterations(evaluation_function=f, gate_set='continuous', budget=b, max_branches=m, roll_out_steps=2,
                                                rollout_type=RT, n_iter=iterations)
