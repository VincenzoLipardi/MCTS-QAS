import mcts
import pandas as pd
import math
import os.path
import numpy as np
from structure import Circuit
import matplotlib.pyplot as plt
import evaluation_functions as evf
from problems.oracles.grover.grover import grover_algo
from evaluation_functions import h2, vqls_1, sudoku, sudoku2x2, h2o, lih, h2o_full


def get_filename(evaluation_function, criteria, budget, branches, iteration, epsilon, stop_deterministic, rollout_type,
                 roll_out_steps, ucb, image, gradient=False, gate_set='continuous'):
    """Creates the filename string for saving or reading files."""
    ro = f'rollout_{rollout_type}/'
    ros = f'_rsteps_{roll_out_steps}'
    stop = '_stop' if stop_deterministic else ''

    if isinstance(branches, bool):
        branch = "dpw" if branches else "pw"
    elif isinstance(branches, int):
        branch = f'bf_{branches}'
    else:
        raise TypeError("Variable branches only accepts boolean and integer types")

    eps = f'_eps_{epsilon}' if epsilon is not None else ''
    grad = '_gd' if gradient else ''

    if image:
        filename = f"{branch}{eps}{ros}{grad}{stop}"
        ro += 'images/'
    else:
        filename = f"{branch}{eps}_budget_{budget}{ros}_run_{iteration}{grad}{stop}"

    ucb_dir = f'ucb{ucb}/'
    directory = os.path.join('experiments', criteria, ucb_dir, evaluation_function.__name__, gate_set, ro)
    return directory, filename


def run_and_savepkl(evaluation_function, criteria, variable_qubits, ancilla_qubits, budget, max_depth, iteration, branches, choices, epsilon, stop_deterministic, ucb, gate_set='continuous', rollout_type="classic", roll_out_steps=None, verbose=False):
    """
    It runs the mcts on the indicated problem and saves the result (the best path) in a .pkl file
    :param criteria: string. Criteria to choose the best children node.
    :param stop_deterministic: If True each node expanded will be also taken into account as terminal node.
    :param epsilon: float. probability to go random
    :param choices: dict. Probability distribution over all the possible class actions
    :param evaluation_function: func. It defines the problem, then the reward function for the mcts agent
    :param variable_qubits:int.  Number of qubits required for the problem
    :param ancilla_qubits: int. Number of ancilla qubits required, as in the case of the oracle problem (Hyperparameter)
    :param gate_set: str. Use 'continuous' (CX+single-qubit rotations) or 'discrete'( Clifford generator + t).
    :param budget: int. Resources allocated for the mcts search. MCTS iterations (or simulations)
    :param max_depth: int. Max depth of the quantum circuit
    :param iteration: int. Number of the independent run.
    :param branches: bool or int. If True progressive widening implemented. If int the number of maximum branches is fixed.
    :param rollout_type: str. classic evaluates the final quantum circuit got after rollout. rollout_max takes the best reward get from all the states in the rollout path
    :param roll_out_steps: int Number of moves for the rollout.
    :param verbose: bool. True if you want to print out the algorithm results online.
    """
    directory, filename = get_filename(evaluation_function, criteria=criteria, budget=budget, branches=branches, iteration=iteration,
                                       gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                                       epsilon=epsilon, stop_deterministic=stop_deterministic, image=False, ucb=ucb)
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose:
            print("Directory created successfully!")


    if not os.path.exists(directory+filename+'.pkl'):
        if isinstance(choices, dict):
            pass
        elif isinstance(choices, list):
            choices = {'a': choices[0], 'd': choices[1], 's': choices[2], 'c': choices[3], 'p': choices[4]}
        else:
            raise TypeError("The variable choices have to be defines as type dictionary or list")

        # Define the root note
        root = mcts.Node(Circuit(variable_qubits=variable_qubits, ancilla_qubits=ancilla_qubits), max_depth=max_depth)
        # Run the mcts algorithm
        final_state = mcts.mcts(root, budget=budget, branches=branches, evaluation_function=evaluation_function, criteria=criteria, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                                choices=choices, epsilon=epsilon, stop_deterministic=stop_deterministic, ucb_value=ucb, verbose=verbose)
        # Create the name of the pickle file where the results will be saved in
        df = pd.DataFrame(final_state)
        df.to_pickle(os.path.join(directory+filename + '.pkl'))

        print("File has been saved as:", directory+filename)
    else:

        print('File already exists:', directory+filename)


def add_columns(evaluation_function, criteria, budget, n_iter, branches, epsilon, stop_deterministic, roll_out_steps, rollout_type, gradient, ucb, gate_set='continuous', verbose=False):
    """Adds the column of the cost function during the search, and apply the gradient descent on the best circuit and save it the column Adam"""
    # Get best paths
    qc_path = get_paths(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, ucb, n_iter)[0]

    for i in range(n_iter):
        directory, filename = get_filename(evaluation_function, criteria, budget, branches, ucb=ucb, iteration=i, gate_set=gate_set, rollout_type=rollout_type, roll_out_steps=roll_out_steps, epsilon=epsilon, stop_deterministic=stop_deterministic, image=False)
        df = pd.read_pickle(directory + filename + '.pkl')
        if 'cost' in df.columns:
            column_cost = df['cost']
            if verbose:
                print('Cost column already created')
        else:
            # Create column of the cost values along the tree path
            column_cost = list(map(lambda x: evaluation_function(x, cost=True), qc_path[i]))
            # Add the columns to the pickle file
            df['cost'] = column_cost
        if gradient:
            if 'Adam' in df.columns:
                if verbose:
                    print('Angle parameter optimization already performed')
            else:
                # Get last circuit in the tree path
                quantum_circuit_last = qc_path[i][-1]

                # Apply gradient on the last circuit and create a column to save it
                final_result = evaluation_function(quantum_circuit_last, ansatz='', cost=False, gradient=True)
                column_adam = [[None]]*df.shape[0]
                column_adam[-1] = final_result

                # Apply gradient on the best circuit if the best is not the last in the path
                if isinstance(column_cost, list):
                    index = column_cost.index(min(column_cost))
                else:
                    column_cost = np.array(column_cost)
                    # index = column_cost.idxmin()
                    index = np.argmin(column_cost)
                if index != len(qc_path[i])-1:
                    quantum_circuit_best = qc_path[i][index]
                    best_result = evaluation_function(quantum_circuit_best, ansatz='', cost=False, gradient=True)
                    column_adam[index] = best_result
                df["Adam"] = column_adam

        df.to_pickle(os.path.join(directory+filename + '.pkl'))
        if verbose:
            print('Columns added to: ', directory+filename)



def get_paths(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, ucb, n_iter=10):
    """ It opens the .pkl files and returns quantum circuits along the best path for all the independent run
    :return: four list of lists
    """
    qc_along_path = []
    children, visits, value = [], [], []
    for i in range(n_iter):
        directory, filename = get_filename(evaluation_function, criteria, budget, branches, iteration=i, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, roll_out_steps=roll_out_steps, image=False, ucb=ucb)
        if os.path.getsize(directory+filename+".pkl") == 0:
            raise EOFError(f"The file {directory+filename} is empty.")
        if os.path.isfile(directory+filename+'.pkl'):
            df = pd.read_pickle(directory+filename+'.pkl')
            qc_along_path.append([circuit for circuit in df['qc']])
            children.append(df['children'].tolist())
            value.append(df['value'].tolist())
            visits. append(df['visits'].tolist())
        else:
            raise FileNotFoundError("The file named "+directory+filename+" does not exists")
    return qc_along_path, children, visits, value


def best_in_path(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, ucb):
    """ It returns the list of the costs of best solution for all the independent runs right after mcts"""
    cost_overall, best_index = [], []
    for i in range(n_iter):
        directory, filename = get_filename(evaluation_function, criteria, budget, branches, iteration=i, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, roll_out_steps=roll_out_steps, image=False, ucb=ucb)
        df = pd.read_pickle(directory+filename + '.pkl')
        cost = df['cost'].tolist()
        if isinstance(cost[0], list):
            best = min(cost)[0]
        else:
            best = min(cost)
        cost_overall.append(best)
        best_index.append(cost.index(best))
    return cost_overall, best_index


def get_best_overall(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, ucb):
    """Given an experiment with fixed hyperparameters, it returns the index of the best run and its convergence via classical optimizer"""
    best = []
    for i in range(n_iter):
        directory, filename = get_filename(evaluation_function=evaluation_function, criteria=criteria, budget=budget, iteration=i, branches=branches, epsilon=epsilon, stop_deterministic=stop_deterministic, rollout_type=rollout_type, roll_out_steps=roll_out_steps,
                                image=False, ucb=ucb)
        df = pd.read_pickle(directory+filename + '.pkl')
        column = df['Adam']
        final = [column[j][-1] for j in range(df.shape[0]) if column[j][0] is not None]
        best.append(min(k for k in final if not math.isnan(k)))
    best_run = best.index(min(best))
    return best, best_run


# PLOTS
def plot_cost(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, ucb, verbose, benchmark=None):
    """It saves the convergence plot of the cost vs tree depth"""

    plt.xlabel('Tree Depth')
    plt.ylabel('Cost')
    max_tree_depth = 0
    for i in range(n_iter):
        directory, filename = get_filename(evaluation_function, criteria, budget, branches, i, epsilon, stop_deterministic, rollout_type, roll_out_steps=roll_out_steps, image=False, ucb=ucb)
        df = pd.read_pickle(directory+filename+'.pkl')
        cost = df['cost']
        tree_depth = len(cost)
        if tree_depth > max_tree_depth:
            max_tree_depth = tree_depth
        plt.plot(list(range(len(cost))), cost, marker='o', linestyle='-', label=str(i+1))

    # Set x-ticks
    indices = list(range(max_tree_depth + 2))
    if max_tree_depth > 20:
        indices = indices[::2]
    plt.xticks(indices)
    # Get benchmark value for the problem
    if evaluation_function == h2 or evaluation_function == h2o or evaluation_function == lih:
        benchmark = get_benchmark(evaluation_function)
        plt.ylabel('Energy (Ha)')
    if evaluation_function == h2:
        plt.yticks(np.arange(-1.2, 0.1, 0.1))

    if benchmark is not None:
        if isinstance(benchmark, list) or isinstance(benchmark, tuple):
            plt.axhline(y=benchmark[0], color='r', linestyle='--', label=f'bench_SCF({round(benchmark[0], 3)})')
            plt.axhline(y=benchmark[1], color='g', linestyle='--', label=f'bench_FCI({round(benchmark[1], 3)})')

        else:
            plt.axhline(y=benchmark, color='r', linestyle='--', label=f'ADAPT-VQE({round(benchmark, 3)})')
    directory, filename = get_filename(evaluation_function=evaluation_function, criteria=criteria, branches=branches, image=True, roll_out_steps=roll_out_steps, rollout_type=rollout_type, iteration=0, budget=budget, epsilon=epsilon, stop_deterministic=stop_deterministic, ucb=ucb)

    plt.legend(loc='best')
    plt.title(evaluation_function.__name__ + ' - Budget  '+str(budget))
    filename = filename + '_budget_'+str(budget)
    plt.savefig(directory+filename + '_cost_along_path.png')
    if verbose:
        print('Plot of the cost along the path saved in image', directory+filename)
    plt.clf()


def boxplot(evaluation_function, budget, criteria, branches, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, gradient, ucb, verbose=False):
    """ Save a boxplot image, with the stats on the n_iter independent runs vs the budget of mcts"""
    solutions = []

    ucb_ = '/' + 'ucb'+str(ucb)
    if not os.path.exists('experiments/'+criteria+ucb_+'/'+evaluation_function.__name__+'/continuous/rollout_'+rollout_type+'/images'):
        os.makedirs('experiments/'+criteria+ucb_+'/'+evaluation_function.__name__+'/continuous/rollout_'+rollout_type+'/images')
        if verbose:
            print("Directory created successfully!")
    for b in budget:
        if not check_file_exist(evaluation_function, criteria, branches, b, roll_out_steps, rollout_type, epsilon, stop_deterministic, ucb, n_iter):
            index = budget.index(b)
            budget = budget[:index]
            break

        if gradient:
            sol = get_best_overall(evaluation_function, criteria, branches, b, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, ucb)
            if isinstance(sol, tuple):
                sol = sol[0]
            solutions.append(sol)
        else:
            sol = best_in_path(evaluation_function, criteria, branches, b, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, ucb)[0]
            solutions.append(sol)

    # Plotting
    budget_effective = [str(b) for b in budget]

    plt.boxplot(solutions, patch_artist=True, labels=budget_effective, meanline=True, showmeans=True, showfliers=True)
    if verbose:
        print([min(a) for a in solutions])
    benchmark = get_benchmark(evaluation_function)
    if benchmark is not None:
        if evaluation_function == h2 or evaluation_function == h2o or evaluation_function ==h2o_full:
            plt.ylabel('Energy (Ha)')
            benchmark = round(benchmark[1], 3)
            label = 'bench_FCI'
        elif evaluation_function == lih:
            plt.ylabel('Energy (Ha)')
            benchmark = round(benchmark, 3)
            label = 'ADAPT-VQE'
        elif evaluation_function == sudoku2x2 or evaluation_function == sudoku:
            plt.ylabel('Cost')
            label = 'exact_oracle'
        else:
            plt.ylabel('Cost')
            label = 'benchmark'
        plt.axhline(y=benchmark, color='r', linestyle='--', label=label)

    directory, filename = get_filename(evaluation_function=evaluation_function, criteria=criteria, branches=branches, image=True, roll_out_steps=roll_out_steps, rollout_type=rollout_type, iteration=0, epsilon=epsilon, stop_deterministic=stop_deterministic, ucb=ucb, gradient=gradient, budget=0)
    plt.title(evaluation_function.__name__)
    plt.xlabel('MCTS Simulations')
    plt.legend()
    plt.savefig(directory+filename + '_boxplot.png')

    plt.clf()
    if verbose:
        print('boxplot image saved in ', directory+filename)
    return solutions


def plot_gradient_descent(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, n_iter, ucb, verbose):

    plt.xlabel('Steps')
    plt.ylabel('Cost')
    for b in budget:
        index = get_best_overall(evaluation_function, criteria, branches, b, roll_out_steps, rollout_type, epsilon,
                                 stop_deterministic, n_iter, ucb)[1]
        directory, filename = get_filename(evaluation_function=evaluation_function, criteria=criteria, budget=b, iteration=index, branches=branches,
                                epsilon=epsilon, stop_deterministic=stop_deterministic, rollout_type=rollout_type,
                                roll_out_steps=roll_out_steps, image=False, ucb=ucb)
        df = pd.read_pickle(directory+filename + '.pkl')
        filtered_column = df[df['Adam'].apply(lambda x: x != [None])]['Adam']

        gd_values = filtered_column.tolist()[0]
        plt.plot(range(len(gd_values)), gd_values, marker='o', linestyle='-', label=str(b))
    plt.title(evaluation_function.__name__ + ' - Adam Optimizer')

    benchmark_value = get_benchmark(evaluation_function)
    if evaluation_function == h2 or evaluation_function == h2o or evaluation_function == lih:
        plt.ylabel('Energy (Ha)')

        if isinstance(benchmark_value, list) or isinstance(benchmark_value, tuple):
            plt.axhline(y=benchmark_value[0], color='r', linestyle='--',
                        label=f'bench_SCF({round(benchmark_value[0], 3)})')
            plt.axhline(y=benchmark_value[1], color='g', linestyle='--',
                        label=f'bench_FCI({round(benchmark_value[1], 3)})')

        else:
            plt.axhline(y=benchmark_value, color='r', linestyle='--', label=f'ADAPT-VQE({round(benchmark_value, 3)})')
    else:
        plt.axhline(y=benchmark_value, color='r', linestyle='--', label=f'bench({benchmark_value})')
    plt.legend()


    directory, filename = get_filename(evaluation_function=evaluation_function, criteria=criteria, budget=budget, iteration=0, branches=branches,
                                        epsilon=epsilon, stop_deterministic=stop_deterministic, rollout_type=rollout_type,
                                        roll_out_steps=roll_out_steps, image=True, ucb=ucb)
    plt.savefig(directory+filename+'_gd.png')
    plt.clf()
    if verbose:
        print('Gradient descent image saved in ', directory+filename)
    return gd_values


# Utils
def check_file_exist(evaluation_function, criteria, branches, budget, roll_out_steps, rollout_type, epsilon, stop_deterministic, ucb,  n_iter=10, gate_set='continuous'):
    """ :return: bool. True if all the files are stored, false otherwise"""
    check = True
    for i in range(n_iter):
        directory, filename = get_filename(evaluation_function, criteria, budget, branches, iteration=i, gate_set=gate_set, rollout_type=rollout_type, epsilon=epsilon, stop_deterministic=stop_deterministic, roll_out_steps=roll_out_steps, image=False, ucb=ucb)
        if not os.path.isfile(directory+filename+'.pkl'):
            check = False
    return check


def get_benchmark(evaluation_function):
    """ It returns the classical benchmark value of the problems in input"""
    if evaluation_function == h2:
        # List
        # sol_scf = H2().benchmark()
        sol_scf = -1.115
        sol_fci = -1.136189454088     # Full configuration Interaction
        return sol_scf, sol_fci
    elif evaluation_function == sudoku2x2 or evaluation_function ==sudoku:
        counts_exact = grover_algo(oracle='exact', iterations=2, ancilla=1)

        if '1001' not in counts_exact:
            counts_exact['1001'] = 0
        elif '0110' not in counts_exact:
            counts_exact['0110'] = 0
        else:
            pass
        right_counts = counts_exact['1001'] + counts_exact['0110']
        return 1-(right_counts/1000)
    elif evaluation_function == lih:
        return -7.972
    elif evaluation_function == h2o or evaluation_function == h2o_full:
        return -75.16, -75.49
    else:
        return .0


def colorplot_oracle(criteria, qubits, gates, accuracy, simulations, steps, ucb):
    counts = np.zeros((len(gates), 2), dtype=int)  # Initialize count array
    magic = ('_easy', '_hard')
    ucb_ = '/' + 'ucb' + str(ucb)
    for i in range(len(gates)):
        for j in range(len(magic)):
            count = 0
            for run in range(10):

                df = pd.read_pickle('experiments/'+criteria+ucb_+'/fidelity_'+str(qubits)+'_'+str(gates[i])+magic[j]+'/continuous/rollout_classic/pw_budget_'+str(simulations)+'_rsteps_'+str(steps)+'_run_'+str(run)+'.pkl')
                cost = df['cost'].tolist()
                if isinstance(cost[0], list):
                    best = min(cost)[0]
                else:
                    best = min(cost)
                if best < accuracy:
                    count += 1
            counts[i, j] = count

    # Create the table plot
    plt.figure(figsize=(12, 5))
    plt.imshow(counts.T, cmap='viridis', origin='lower', aspect='auto', extent=[-0.5, len(gates)-0.5, -0.5, 1.5], vmin=0, vmax=10)
    plt.colorbar(label='Number of successful approximations')  # Add colorbar to show mapping of colors to counts
    plt.xlabel('Number of Quantum Gates')
    plt.ylabel('Magic')
    plt.title(f'Random Quantum Circuits with {str(qubits)} qubits')
    plt.xticks(range(len(gates)), gates)
    plt.yticks([0, 1], ['easy', 'hard'])
    plt.grid(False)
    plt.savefig('experiments/'+criteria+ucb_+'/colormap_'+str(qubits)+'_'+str(steps)+'_acc'+str(accuracy)+'.png')
    print('Image saved')


def oracle_interpolation(qubit, epsilon, steps):

    # Sample data
    gates = [5, 10, 15, 20, 30]
    if qubit == 8:
        gates = [5, 10, 15, 20]
    simulations = [1000, 2000, 5000, 10000, 50000, 100000, 200000]
    if qubit ==4:
        simulations = [1000, 2000, 5000, 10000, 50000, 100000]


    magic = ('_easy', '_hard')
    data = []
    for m in range(len(magic)):
        data_g = []
        for g in range(len(gates)):
            prova = []
            for s in simulations:
                count = 0

                for run in range(10):
                    df = pd.read_pickle('experiments/fidelity_' + str(qubit) + '_' + str(gates[g]) + magic[m] + '/continuous/rollout_classic/pw_budget_' + str(s) + '_rsteps_'+str(steps)+'_run_' + str(
                        run) + '.pkl')

                    cost = df['cost'].tolist()
                    if isinstance(cost[0], list):
                        best = min(cost)[0]
                    else:
                        best = min(cost)
                    if best < epsilon:
                        count += 1
                prova.append(count)
            data_g.append(prova)
        data.append(data_g)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for i in range(len(gates)):
        # Plotting the first scatter plot
        plt.scatter(simulations, data[0][i], label=str(gates[i])+' gates')
        plt.plot(simulations, data[0][i], linestyle='-', linewidth=1, alpha=0.7)
        print(i)
        print(data[0][i])

    plt.title('Easy')
    plt.ylabel('Successful Approximations')
    plt.xlabel('MCTS Simulations')
    plt.xscale('log')
    plt.xticks(simulations)
    plt.legend()
    plt.subplot(1, 2, 2)
    for i in range(len(gates)):
        plt.scatter(simulations, data[1][i], label=str(gates[i])+' gates')
        plt.plot(simulations, data[1][i], linestyle='-', linewidth=1, alpha=0.7)
        plt.title('Hard')

        print(i)
        print(data[1][i])
    plt.xlabel('MCTS Simulations')
    plt.xticks(simulations)
    plt.xscale('log')
    plt.ylabel('Successful Approximations')
    plt.tight_layout()
    plt.legend()
    plt.savefig('experiments/simulation_plot_'+str(qubit)+'_'+str(steps)+'.png')



def grid_search(hyperparameters):
    print('Grid Search Completed')
