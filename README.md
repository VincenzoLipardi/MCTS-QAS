This is the Github repository hosting code and experiments of the article: ...

# Environment setup
All the file are written in python and the libraries needed to successfully run the code are listed in the requirements.txt.

# Tutorial
A tutorial into the library is provided in the Jupyter Notebook file "tutorial.ipynb" .

# Monte Carlo Tree Search for Quantum Circuit Design
In this project we have further investigated the role of Monte Carlo Tree Search within the domain of quantum architecture search.
We have carried out experiments for finding the ground state energy in quantum chemistry problems (VQE),
for solving systems of linear equations (VQLS) and for the general problem of oracle approximation.

## All-in-one MCTS
Despite the previous work we used MCTS to find both the topology the parametrized quantum circuits 
and their parameters. Then we provided MCTS of a progressive widening technique in order to deal with the continuous action space.


# Guide into the repository

In the directory 'problems' there is the implementation of all the applications described in the article and in file "evaluation_functions.py" a list of all the objective functions used. 
In the directory 'experiments' there are the numerical results obtained and shown in the paper.

The MCTS algorithm is implemented in mcts.py, and some utils are in structure.py.

