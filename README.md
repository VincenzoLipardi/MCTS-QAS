# Quantum Circuit Design using Progressive Widening Enhanced Monte Carlo Tree Search  

This repository contains the code and experiments for the article:  
**"Quantum Circuit Design using Progressive Widening Enhanced Monte Carlo Tree Search"**  

## 📖 Paper  
[🔗 Link to the paper](https://arxiv.org/abs/2502.03962)  

## 🚀 Environment Setup  

All scripts are written in Python. Follow these steps to set up the environment:  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```  

2. **Set up a virtual environment (Conda or venv)**  
   - **Using Conda:**  
     ```bash
     conda create --name quantum-mcts python=3.8  
     conda activate quantum-mcts
     ```
   - **Using venv:**  
     ```bash
     python -m venv venv  
     source venv/bin/activate  # On Windows, use: venv\Scripts\activate
     ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 📚 Tutorial  

A guided tutorial showcasing the main functionality of the repository is available in:  
📄 [tutorial.ipynb](tutorial.ipynb)  

---

## 🔍 Monte Carlo Tree Search for Quantum Circuit Design  

Variational Quantum Algorithms (VQAs) rely heavily on the structure of parameterized quantum circuits. However, designing optimal circuits for specific problems and hardware is a major challenge.  

This work introduces a gradient-free **Monte Carlo Tree Search (MCTS)** method for quantum circuit design. Key innovations:  
- **Progressive widening**: Dynamically explores the action space.  
- **Sampling-based action formulation**: Efficiently navigates the search space.  
- **Application domains**:  
  - Random quantum circuits: Approximates unstructured circuits based on stabilizer Rényi entropy.  
  - Quantum chemistry & linear systems: Robustness across different applications.  
- **Performance improvement**:  
  - Reduces quantum circuit evaluations **by a factor of 10-100** compared to prior MCTS approaches.  
  - Generates circuits with **up to 3× fewer CNOT gates**.  

---

## 📂 Repository Structure  

- 📁 [`problems/`](problems/) – Implementations of the applications discussed in the paper.  
- 📄 [`evaluation_functions.py`](problems/evaluation_functions.py) – List of all objective functions used.  
- 📁 [`experiments/`](experiments/) – Contains the numerical results presented in the paper.  
- 📄 [`mcts.py`](mcts.py) – Implementation of the Monte Carlo Tree Search algorithm.  
- 📄 [`structure.py`](structure.py) – Utility functions used throughout the repository.  

---

## 🛠 Usage  

To run experiments, execute [`main.py`](main.py) file  where you can custom problems and hyperparameters.

---

## 📬 Contact  

For questions or contributions, feel free to open an issue or submit a pull request.  

