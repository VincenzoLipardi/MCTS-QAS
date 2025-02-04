This is the Github repository hosting code and experiments of the article: "Quantum Circuit Design using Progressive Widening Enhanced Monte Carlo Tree Search".

# Environment setup
All the scripts are written in python. In order to run these scripts follow the steps below:
- clone the repository
- setup a conda environment (or virtual environment)
- install the requirements with the following command: pip install -r requirements.txt


# Tutorial
In the Jupyter Notebook file "tutorial.ipynb" you can find a guided tutorial into the main functionality of the repository.

# Monte Carlo Tree Search for Quantum Circuit Design
The performance of Variational Quantum Algorithms (VQAs) strongly depends on the choice of the parameterized quantum circuit to optimize. One of the biggest challenges in VQAs is designing quantum circuits tailored to the particular problem and to the quantum hardware. 
This article proposes a gradient-free Monte Carlo Tree Search (MCTS) technique to automate the process of quantum circuit design. It introduces a novel formulation of the action space based on a sampling scheme and a progressive widening technique to explore the space dynamically. When testing our MCTS approach on the domain of random quantum circuits, MCTS approximates unstructured circuits under different values of stabilizer R\'enyi entropy. It turns out that MCTS manages to approximate the benchmark quantum states independently from their degree of nonstabilizerness. Next, our technique exhibits robustness across various application domains, including quantum chemistry and systems of linear equations. Compared to previous MCTS research, our technique reduces the number of quantum circuit evaluations by a factor of 10 to 100 while achieving equal or better results. In addition, the resulting quantum circuits have up to three times fewer CNOT gates.

# Guide into the repository

In the directory 'problems' there is the implementation of all the applications described in the article and in file "evaluation_functions.py" a list of all the objective functions used. 
In the directory 'experiments' there are the numerical results obtained and shown in the paper.

The MCTS algorithm is implemented in mcts.py, and some utils are in structure.py.

