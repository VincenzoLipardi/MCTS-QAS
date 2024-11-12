from qiskit import QuantumCircuit
from collections import defaultdict, Counter

# Sample dataset: List of quantum circuits (for illustration)
# Replace these with actual quantum circuits
n_qubits = 3
dataset = [QuantumCircuit(n_qubits) for _ in range(10)]

# Example circuits in the dataset
for j in range(len(dataset)):
    for i in range(n_qubits):
        dataset[j].h(i)
        dataset[j].rx(0, i)
    dataset[j].cx(0, 1)

"""dataset[0].x(2)
dataset[1].h(1)
dataset[2].cx(1, 2)
dataset[3].z(2)
dataset[4].h(2)
dataset[5].z(0)
dataset[6].cx(2, 0)
dataset[7].h(1)
dataset[8].cx(1, 2)
dataset[9].h(2)"""


# Function to extract gate sequences for each qubit in a circuit
def extract_gate_sequence_by_qubit(circuit):
    qubit_gates = defaultdict(list)

    for instruction in circuit.data:
        gate = instruction[0].name  # Extract gate name
        qubits = instruction[1]  # Extract the qubits this gate is applied to
        for qubit in qubits:
            qubit_gates[circuit.find_bit(qubit).index].append(gate)  # Record the gate applied on those qubits

    # Ensure all qubits are represented (even if no gates applied)
    for q in range(len(circuit.qubits)):
        if q not in qubit_gates:
            qubit_gates[q] = []
    return qubit_gates


# Function to build n-grams for a list of gates
def build_ngrams(gate_sequence, n):
    return [tuple(gate_sequence[i:i + n]) for i in range(len(gate_sequence) - n + 1)]


# Function to update n-grams for each qubit as circuits are processed
def update_ngrams(circuit, n, qubit_ngrams_counter):
    qubit_gates = extract_gate_sequence_by_qubit(circuit)
    n_qubits = len(circuit.qubits)

    for qubit in range(n_qubits):
        gate_sequence = qubit_gates[qubit]
        ngrams = build_ngrams(gate_sequence, n)
        qubit_ngrams_counter[qubit].update(ngrams)

    return qubit_ngrams_counter



# Function to update conditional results based on quality measure
def update_conditional_results(qubit_ngrams_counter, quality_results_by_qubit, quality_score):
    for qubit, ngrams_counter in qubit_ngrams_counter.items():
        for ngram, count in ngrams_counter.items():
            if ngram not in quality_results_by_qubit[qubit]:
                quality_results_by_qubit[qubit][ngram] = (quality_score, 1)
            else:
                counter = quality_results_by_qubit[qubit][ngram][1]
                average_quality = (quality_results_by_qubit[qubit][ngram][0]+quality_score)/ counter
                quality_results_by_qubit[qubit][ngram] = (average_quality, counter+1)

    return quality_results_by_qubit


# Initialize the n-gram counter and quality results storage
n = 2  # n-gram size (bigrams)
n_qubits = len(dataset[0].qubits)
qubit_ngrams_counter = {q: Counter() for q in range(n_qubits)}
quality_results_by_qubit = defaultdict(lambda: defaultdict(list))
quality_score = 1
# Process each circuit and update n-grams and quality results
for circuit in dataset:
    qubit_ngrams_counter = update_ngrams(circuit, n, qubit_ngrams_counter)
    quality_results_by_qubit = update_conditional_results(qubit_ngrams_counter, quality_results_by_qubit, quality_score)
for qubit, ngrams_quality in quality_results_by_qubit.items():
    print(f"\nQubit {qubit} - N-grams and their quality results:")
    print(ngrams_quality)
    for ngram, quality_scores in ngrams_quality.items():
        print(f"N-gram {ngram}: Quality Scores = {quality_scores}")

