This is the Github repository hosting code and experiments of the article: 

All the file are written in python and the library needed to successfully run the code are available in requirements.txt.

# Monte Carlo Tree Search for Quantum Circuit Design
In this project we have further investigated the role of Monte Carlo Tree Search within the domain of quantum architecture search.
We have carried out experiments for finding the ground state energy in quantum chemistry problems (VQE),
for solving systems of linear equations (VQLS) and for the general oracle approximation problem.

## All-in-one MCTS
Despite the previous work we used MCTS to find both the topology the parametrized quantum circuits 
and their parameters. Then we provided MCTS of a progressive widening technique in order to deal with the continuous action space.


## Guide into the repository

In the directory 'problems' there is the implementation of all the applications described in the article. 
While in the 'experiments' directory you can find the numerical results of all the experiments run.
The truct