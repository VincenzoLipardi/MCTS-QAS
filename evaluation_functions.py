from problems.vqe import h2_class, lih_class, h2o_class, h2o_full_class, h2o_class_noise_01, h2o_class_noise_1, h2o_class_noise_2, h2_class_noise_01, h2_class_noise_1, h2_class_noise_2, h2_class_noise_depolarizing, h2_class_noise_mixed
from problems.vqls import vqls_demo, vqls_paper
from problems.oracles.oracle_approximation import Fidelity


# FIDELITY
def fidelity_8_20_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_20_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_20_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_20_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_20_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_20_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)
    
def fidelity_4_20_hard_bfnoise_1(quantum_circuit, ansatz='', cost=False, gradient=False, noise=0.1):
    problem = Fidelity(qubits=4, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit, noise=noise)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit, noise=noise)
    else:
        return problem.reward(quantum_circuit=quantum_circuit, noise=noise)
    
def fidelity_4_20_hard_bfnoise_2(quantum_circuit, ansatz='', cost=False, gradient=False, noise=0.2):
    problem = Fidelity(qubits=4, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit, noise=noise)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit, noise=noise)
    else:
        return problem.reward(quantum_circuit=quantum_circuit, noise=noise)
    
def fidelity_6_20_hard_bfnoise_1(quantum_circuit, ansatz='', cost=False, gradient=False, noise=0.1):
    problem = Fidelity(qubits=6, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit, noise=noise)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit, noise=noise)
    else:
        return problem.reward(quantum_circuit=quantum_circuit, noise=noise)
    
def fidelity_6_20_hard_bfnoise_2(quantum_circuit, ansatz='', cost=False, gradient=False, noise=0.2):
    problem = Fidelity(qubits=6, gates=20, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit, noise=noise)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit, noise=noise)
    else:
        return problem.reward(quantum_circuit=quantum_circuit, noise=noise)
    
def fidelity_4_20_easy_bfnoise_1(quantum_circuit, ansatz='', cost=False, gradient=False, noise=0.1):
    problem = Fidelity(qubits=4, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit, noise=noise)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit, noise=noise)
    else:
        return problem.reward(quantum_circuit=quantum_circuit, noise=noise)
    
def fidelity_4_20_easy_bfnoise_2(quantum_circuit, ansatz='', cost=False, gradient=False, noise=0.2):
    problem = Fidelity(qubits=4, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit, noise=noise)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit, noise=noise)
    else:
        return problem.reward(quantum_circuit=quantum_circuit, noise=noise)

def fidelity_6_20_easy_bfnoise_1(quantum_circuit, ansatz='', cost=False, gradient=False, noise=0.1):
    problem = Fidelity(qubits=6, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit, noise=noise)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit, noise=noise)
    else:
        return problem.reward(quantum_circuit=quantum_circuit, noise=noise)
    
def fidelity_6_20_easy_bfnoise_2(quantum_circuit, ansatz='', cost=False, gradient=False, noise=0.2):
    problem = Fidelity(qubits=6, gates=20, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit, noise=noise)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit, noise=noise)
    else:
        return problem.reward(quantum_circuit=quantum_circuit, noise=noise)

def fidelity_8_15_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=15, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_15_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=15, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_15_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=15, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_15_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=15, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_15_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=15, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_15_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=15, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_10_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=10, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_10_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=10, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_10_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=10, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_10_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=10, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_10_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=10, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_10_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=10, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_5_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=5, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_5_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=5, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_5_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=5, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_5_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=5, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_5_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=5, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_5_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=5, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_30_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=30, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_4_30_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=4, gates=30, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)

def fidelity_6_30_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=30, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_6_30_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=6, gates=30, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)

def fidelity_8_30_easy(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=30, magic='easy')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def fidelity_8_30_hard(quantum_circuit, ansatz='', cost=False, gradient=False):
    problem = Fidelity(qubits=8, gates=30, magic='hard')
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.cost(quantum_circuit=quantum_circuit)
    else:
        return problem.reward(quantum_circuit=quantum_circuit)


def h2(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)

def h2_noise_01(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2_class_noise_01
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)

def h2_noise_1(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2_class_noise_1  
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)

def h2_noise_2(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2_class_noise_2
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')  
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    
def h2_noise_depolarizing(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2_class_noise_depolarizing
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)    
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    
def h2_noise_mixed(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2_class_noise_mixed
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)    
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def lih(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = lih_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def h2o(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2o_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    
def h2o_noise_01(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2o_class_noise_01
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    
def h2o_noise_1(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2o_class_noise_1
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    
def h2o_noise_2(quantum_circuit, ansatz='all', cost=False, gradient=False): 
    problem = h2o_class_noise_2
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)

def h2o_full(quantum_circuit, ansatz='all', cost=False, gradient=False):
    problem = h2o_full_class
    if cost and gradient:
        raise ValueError('Cannot return both cost/reward and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def vqls_0(quantum_circuit, ansatz='all', cost=False):
    # Instance shown in pennylane demo: https://pennylane.ai/qml/demos/tutorial_vqls/
    problem = vqls_demo
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)


def vqls_1(quantum_circuit, ansatz='all', cost=False, gradient=False):
    # Define the problem A = c_0 I + c_1 X_1 + c_2 X_2 + c_3 Z_3 Z_4
    problem = vqls_paper

    if cost and gradient:
        raise ValueError('Cannot return both cost and gradient descent result')
    if gradient:
        return problem.gradient_descent(quantum_circuit=quantum_circuit)
    if cost:
        return problem.costFunc(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
    else:
        return problem.getReward(params=[0.1], quantum_circuit=quantum_circuit, ansatz=ansatz)
