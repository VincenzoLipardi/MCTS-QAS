import numpy as np
from structure import Circuit, GateSet
from qiskit import QuantumCircuit


class Node:
    def __init__(self, state: Circuit, max_depth, parent=None):
        """
        A node of the tree store a lot of information in order to guide the search
        :param state: Circuit object. This is the quantum circuit stored in the node.
        :param parent: Node object. Parent Node
        """
        # Quantum circuit
        self.state = state
        # Is the circuit respecting the constraint of the hardware? boolean
        self.isTerminal = False
        # Parent node. Node object. The root node is the only one not having that.
        self.parent = parent
        # List of children of the node. list
        self.children = []
        # Number of times the node have been visited. integer
        self.visits = 0
        # Value is the total reward. float
        self.value = 0
        # Maximum quantum circuit depth
        self.max_depth = max_depth
        # Position of the node in terms of tree depth. integer
        self.tree_depth = 0 if parent is None else parent.tree_depth + 1
        # Gate set
        self.gate_set = 'continuous'
        # Weights of the possible actions when expansion is executed
        self.stop_is_done = False

    def is_fully_expanded(self, branches):
        """
        :param branches: int or boolean. If true, progressive widening. if int the maximum number of branches is fixed.
        :return: Boolean. True if the node is a leaf. False otherwise.
        """
        if isinstance(branches, bool):
            if branches:
                t = self.visits
                # alpha in [0,1], set it close to 1 in domain is strongly stochastic, close to 0 otherwise
                if t == 0:
                    t = 1
                alpha = 0.3
                C = 1
                k = np.ceil(C*(t**alpha))
                return len(self.children) >= k
            else:
                raise NotImplementedError
        elif isinstance(branches, int):
            # max_branches
            return len(self.children) >= branches
        else:
            raise TypeError

    def define_children(self, prob_choice, roll_out=False):
        """

        :param prob_choice: dict.
        :param roll_out: boolean. True if it is used for the rollout (new nodes are temporary, not included in the tree)
        :return: Node
        """
        # Expand the node by adding a new gate to the circuit
        parent = self
        qc = parent.state.circuit.copy()
        stop = self.stop_is_done
        if roll_out:
            stop = True
        new_qc = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, prob_choice, stop)(qc)
        if new_qc == 'stop':
            # It means that get_legal_actions returned the STOP action, then we define this node as Terminal
            self.isTerminal = True
            self.stop_is_done = True
            return self
        else:

            while new_qc is None:
                # It chooses to change parameters, but there are no parametrized gates. Or delete in a very shallow circuit
                new_qc = parent.state.get_legal_action(GateSet(self.gate_set), self.max_depth, prob_choice, stop)(qc)

            if isinstance(new_qc, QuantumCircuit):
                new_state = Circuit(4, 1).building_state(new_qc)
                new_child = Node(new_state, max_depth=self.max_depth, parent=self)
                if not roll_out:
                    self.children.append(new_child)
                return new_child
            else:
                raise TypeError

    def best_child(self):
        children_with_values = [(child, child.value)
                                for child in self.children]
        return max(children_with_values, key=lambda x: x[1])[0]


def select(node, exploration=1.0):
    l = np.log(node.visits)
    children_with_values = [(child, child.value / child.visits +
                             exploration * np.sqrt(l / child.visits))
                            for child in node.children]

    selected_child = max(children_with_values, key=lambda x: x[1])[0]
    return selected_child


def expand(node, prob_choice):
    new_node = node.define_children(prob_choice=prob_choice)
    return new_node


def rollout(node, steps):
    new_node = node
    for i in range(steps):
        new_node = new_node.define_children(prob_choice={'a': 10, 'd': 10, 's': 40, 'c': 40, 'p': 0}, roll_out=True)
    return new_node


def evaluate(node, evaluation_function):
    return node.state.evaluation(evaluation_function)


def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent


def modify_prob_choice(dictionary, len_qc, stop_happened=True):

    keys = list(dictionary.keys())
    values = list(dictionary.values())

    modifications = [-40, 10, 10, 10, 10]
    modified_values = [max(0, v + m) for v, m in zip(values, modifications)]
    if stop_happened:
        modified_values[-1] = 0
    if len_qc < 6:
        modified_values[1] = 0
    # Normalize to ensure the sum is still 100
    modified_values = [v / sum(modified_values) * 100 for v in modified_values]
    # Normalize to ensure the sum is still 100
    modified_dict = dict(zip(keys, modified_values))
    return modified_dict


def mcts(root, budget, evaluation_function, rollout_type, roll_out_steps, branches, verbose=False):
    prob_choiche = {'a': 100, 'd': 0, 's': 0, 'c': 0, 'p': 0}
    if verbose:
        print('Root Node: \n', root.state.circuit)
    epoch_counter = 0
    best_found = None
    for _ in range(budget):
        current_node = root
        if verbose:
            print('Epoch Counter: ', epoch_counter)

        # Selection

        while not current_node.isTerminal and current_node.is_fully_expanded(branches=branches):
            current_node = select(current_node)
            if verbose:
                print('Selection done. The selected child is: ', current_node, 'Node tree depth: ', current_node.tree_depth)

        # Expansion
        if not current_node.isTerminal:
            current_node = expand(current_node, prob_choice=prob_choiche)
            if verbose:
                print("Tree expanded. Node's depth in the tree: ", current_node.tree_depth)

        # Simulation
        if not current_node.isTerminal:
            if isinstance(roll_out_steps, int):
                leaf_node = rollout(current_node, steps=roll_out_steps)
                result = evaluate(leaf_node, evaluation_function)
                if roll_out_steps > 1 and rollout_type == 'Rollout_max':
                    result_list = [result]
                    node_to_evaluate = leaf_node
                    for _ in range(roll_out_steps):
                        result_list.append(evaluate(node_to_evaluate.parent, evaluation_function))
                    result = max(result_list)

            else:
                if verbose:
                    print('No rollout')
                result = evaluate(current_node, evaluation_function)
        else:
            if verbose:
                print('It is a terminal node, evaluation done')
            result = evaluate(current_node, evaluation_function)
        if verbose:
            print('Simulation result: ', result)
        # Backpropagation
        backpropagate(current_node, result)
        epoch_counter += 1
        n_qubits = len(current_node.state.circuit.qubits)
        if current_node.tree_depth == 2*n_qubits:
            prob_choiche = {'a': 50, 'd': 0, 's': 25, 'c': 25, 'p': 0}

    print('Last epoch:', epoch_counter)
    # Return the best
    best_node = root
    path = []
    while len(best_node.children) > 1:
        path.append(best_node)
        best_node = best_node.best_child()
    if best_found is not None:
        best_node = best_found

    return best_node, path
