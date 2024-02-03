import save_in_file as sif
from problems.evaluation_functions import h2, vqls_1, sudoku2x2

BUDGET = [2000, 5000, 10000, 50000]
M = [False]
eval_func = [h2]
iterations = 10
RT = 'classic'
ROS = 1
p = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
for f in eval_func:
    for b in BUDGET:
        for m in M:
            for i in range(iterations):
                if f == sudoku2x2:
                    sif.data(evaluation_function=f, variable_qubits=5, ancilla_qubits=0, gate_set='continuous',
                             budget=b, branches=m, max_depth=20, roll_out_steps=0, iteration=i, choices=p, verbose=True)
                else:
                    sif.data(evaluation_function=f, variable_qubits=4, ancilla_qubits=0, gate_set='continuous', rollout_type=RT,
                             budget=b, branches=m, roll_out_steps=ROS, iteration=i, max_depth=20,  choices=p, verbose=True)
                    print('done')