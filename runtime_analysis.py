import mcts
from structure import Circuit
import evaluation_functions as evf
import time
import numpy as np
import os
import json
import matplotlib.pyplot as plt
variable_qubits = 4
ancilla_qubits = 0
max_depth = 20
budget = 1000
evaluation_function = evf.h2
criteria = 'value'
rollout_type = 'classic'
roll_out_steps = 0
simulation = False
choices = {'a': 50, 'd': 10, 's': 20, 'c': 20, 'p': 0}
epsilon = None
stop_deterministic = False
ucb = 0.4
verbose = False
branch_values = [False, 10, 20]
"""os.makedirs('experiments/paper/runtime', exist_ok=True)

n_runs = 1000
results = {}

for branch_val in branch_values:
    print(f"\nStarting experiments with branches={branch_val}")
    execution_times = []
    
    for i in range(n_runs):
        if i % 100 == 0:
            print(f"Run {i}/{n_runs}")
            
        root = mcts.Node(Circuit(variable_qubits=variable_qubits, ancilla_qubits=ancilla_qubits), max_depth=max_depth)
        start_time = time.time()
        final_state = mcts.mcts(root, budget=budget, branches=branch_val, evaluation_function=evaluation_function, 
                               criteria=criteria, rollout_type=rollout_type, roll_out_steps=roll_out_steps, 
                               simulation=simulation, choices=choices, epsilon=epsilon, 
                               stop_deterministic=stop_deterministic, ucb_value=ucb, verbose=verbose)
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
    
    avg_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    results[str(branch_val)] = {
        'times': execution_times,
        'average': avg_time,
        'std': std_time
    }
    print(f"\nResults for branches={branch_val}:")
    print(f"Average execution time: {avg_time:.2f} seconds")
    print(f"Standard deviation: {std_time:.2f} seconds")

# Save detailed results
with open('experiments/paper/runtime/detailed_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# Save summary
summary = {str(k): {'average': v['average'], 'std': v['std']} for k, v in results.items()}
with open('experiments/paper/runtime/summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
print("\nResults have been saved to experiments/paper/runtime/")
"""

# Load and plot the results
def plot_runtime_results():
    # Load the detailed results
    with open('experiments/paper/runtime/detailed_results.json', 'r') as f:
        results = json.load(f)

    # Calculate percentage difference between branch=10 and Progressive Widening
    pw_avg = results['False']['average']  # Progressive Widening
    branch_10_avg = results['10']['average']  # branch=10
    percent_diff_10 = ((pw_avg - branch_10_avg) / branch_10_avg) * 100
    
    # Calculate percentage difference between branch=20 and Progressive Widening
    branch_20_avg = results['20']['average']  # branch=20
    percent_diff_20 = ((pw_avg - branch_20_avg) / branch_20_avg) * 100
    
    print(f"\nComparison between Branch=10 and Progressive Widening:")
    print(f"Progressive Widening average: {pw_avg:.4f} seconds")
    print(f"Branch=10 average: {branch_10_avg:.4f} seconds")
    print(f"Percentage difference: {percent_diff_10:.2f}% ({'increase' if percent_diff_10 > 0 else 'decrease'})")
    
    print(f"\nComparison between Branch=20 and Progressive Widening:")
    print(f"Progressive Widening average: {pw_avg:.4f} seconds")
    print(f"Branch=20 average: {branch_20_avg:.4f} seconds")
    print(f"Percentage difference: {percent_diff_20:.2f}% ({'increase' if percent_diff_20 > 0 else 'decrease'})")

    # Prepare data for box plot
    data = [results[str(branch)]['times'] for branch in branch_values]
    labels = [f'branches={branch}' if branch else 'Progressive Widening' for branch in branch_values]

    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig('experiments/paper/runtime/runtime_analysis_boxplot.png')
    plt.close()

    # Create a bar plot with error bars
    averages = [results[str(branch)]['average'] for branch in branch_values]
    stds = [results[str(branch)]['std'] for branch in branch_values]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(branch_values)), averages, yerr=stds, capsize=5)
    plt.xticks(range(len(branch_values)), labels)
    plt.ylabel('Average Execution Time (seconds)')
    plt.grid(True, alpha=0.3)

    # Save the plot
    plt.savefig('experiments/paper/runtime/runtime_analysis_barplot.png')
    plt.close()

# Call the plotting function if results exist
if os.path.exists('experiments/paper/runtime/detailed_results.json'):
    plot_runtime_results()
    print("Plots have been saved to experiments/paper/runtime/")
else:
    print("No results found. Please run the experiments first.")




