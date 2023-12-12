import numpy as np
from structure import Circuit, GateSet
from qiskit import QuantumCircuit
global max_depth


class Node:
    def __init__(self, state: Circuit, parent=None):
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
        # Position of the node in terms of tree depth. integer
        self.tree_depth = 0 if parent is None else parent.tree_depth + 1
        # Gate set
        self.gate_set = 'continuous'

    """def __repr__(self):
        return "State: {}\nParent: {}\nChildren: {} \nNum Visits: {}\nTotal Reward: {}\nisTerminal: {}\nTree Depth: {}".format(
            self.state, self.parent, self.children, self.visits, self.value, self.isTerminal, self.tree_depth)"""

    def is_fully_expanded(self, max_branches):
        """
        :param max_branches: Maximum number of branches per node
        :return: Boolean. True if the node is a leaf. False otherwise.
        """
        return len(self.children) >= max_branches

    def define_children(self, max_branches, prob_choice, roll_out=False):
        # Expand the node by adding a new gate to the circuit

        if len(self.children) > max_branches:
            return None
        else:
            parent = self

            qc = parent.state.circuit.copy()
            new_qc = parent.state.get_legal_action(GateSet(self.gate_set), max_depth, prob_choice)(qc)
            while new_qc is None:
                new_qc = parent.state.get_legal_action(GateSet(self.gate_set), max_depth, prob_choice)(qc)

            if isinstance(new_qc, QuantumCircuit):
                new_state = Circuit(4, 1).building_state(new_qc)
                new_child = Node(new_state, parent=self)
                # print('new node added', new_child.state.circuit, '\n whose circuit depth is: ', new_state.circuit.depth())
                if not roll_out:
                    self.children.append(new_child)
                return new_child

                # It means that he got a non possible action


            else:
                # It means that get_legal_actions returned the STOP action, then we define this node as Terminal
                self.isTerminal = True

    def best_child(self):
        children_with_values = [(child, child.value)
                                for child in self.children]

        return max(children_with_values, key=lambda x: x[1])[0]


def select(node, exploration=1.0):
    if not node.children:
        return None
    l = np.log(node.visits)
    children_with_values = [(child, child.value / child.visits +
                             exploration * np.sqrt(l / child.visits))
                            for child in node.children]

    selected_child = max(children_with_values, key=lambda x: x[1])[0]
    # print("Children UCB values: ", children_with_values)
    return selected_child


def expand(node, max_branches, prob_choice):
    new_node = node.define_children(max_branches=max_branches, prob_choice=prob_choice)
    return new_node

def rollout(node):
    new_node = node
    for i in range(2):
        new_node = new_node.define_children(max_branches=9999, prob_choice={'a': 70, 'd': 10, 's': 0, 'c': 20, 'p': 0}, roll_out=True)
    return new_node

def simulate(node, evaluation_function):
    return node.state.evaluation(evaluation_function)


def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent


def modify_prob_choice(dictionary, stop_happened=True):

    keys = list(dictionary.keys())
    values = list(dictionary.values())

    modifications = [-40, 10, 10, 10, 10]
    modified_values = [max(0, v + m) for v, m in zip(values, modifications)]
    if stop_happened:
        modified_values[-1] = 0
    # Normalize to ensure the sum is still 100
    modified_values = [v / sum(modified_values) * 100 for v in modified_values]
    # Normalize to ensure the sum is still 100
    modified_dict = dict(zip(keys, modified_values))
    return modified_dict


def mcts(root, budget, max_branches, evaluation_function, verbose=False):
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
        while not current_node.isTerminal and current_node.is_fully_expanded(max_branches=max_branches):
            current_node = select(current_node)
            if verbose:
                print('Selection done. The selected child is: ', current_node, 'Node tree depth: ', current_node.tree_depth)
        # Expansion
        if not current_node.isTerminal:
            current_node = expand(current_node, max_branches, prob_choice=prob_choiche)
        if verbose:
            print("Tree expanded. Node's depth in the tree: ", current_node.tree_depth)
        # Simulation
        leaf_node = rollout(current_node)
        result = simulate(leaf_node, evaluation_function)

        if verbose:
            print('Simulation result: ', result)
        # Backpropagation
        backpropagate(current_node, result)
        # print('Current node value:', current_node.value)

        epoch_counter += 1
        if current_node.tree_depth == 5:
            prob_choiche = modify_prob_choice(prob_choiche)
        if current_node.tree_depth == 15:
            prob_choiche = modify_prob_choice(prob_choiche)
        if result > 0.97:
            best_found = current_node
            break
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


def nested(iterations, root, budget, max_branches, evaluation_function, verbose):
    for _ in range(iterations):
        mcts(root, budget=budget, max_branches=max_branches, evaluation_function=evaluation_function, verbose=verbose)
        root = root.best_child()
    best_child = root.best_child()
    return best_child


max_depth = 20
