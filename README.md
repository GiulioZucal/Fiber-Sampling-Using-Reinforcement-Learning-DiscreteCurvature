## Fiber-Sampling-Using-Reinforcement-Learning for sampling Joint Degree matrices with fixed Forman-Ricci curvature and degree sequence

The code in this repo is for the paper: "Markov and lattice bases for Forman-Ricci curvature of graphs" (https://arxiv.org/pdf/2608.01929)
and it is based on the code for the paper "Learning to sample fibers for goodness-of-fit testing". (https://arxiv.org/abs/2405.13950)

---

## Project Structure

```
GaussianAC/
├── DeepFiberSamplingGaussianENV.py# RL environment
├── Gaussian A2C Fiber Sampling.ipynb # Main notebook from where to run the code.
├── GaussianA2C.py # Model code, model initialization and training.
├── helper_functions.py # Helper functions for the data preparation.
├── reward_functions.py # Different reward functions.
├── stats_functions.py # Functions for processing the design matrix and moves.
├── stats_problems.py # Different statistical problems and fibers, in particular there are different problems here for sampling Joint Degree Matrices (JDMs) of graphs with a given curvature and degree sequence.
├── matrix_utils.py #Several functions to deal with the matrices (in particular JDMs) needed for the problem of sampling JDMs with a given curvature and degree sequence.
├── functions_algorithm_stanton_pinar_JDM.py # Functions for greedy algorithm to obtain a graph with a specific JDM (Algorithm from Stanton-Pinar, 2011 (Algorithm 1) 
└── README.md
```

---

## How to use the code:

- Clone the whole repository.
```
git clone <repo http>
```

-  Run the notebook:
```
  Gaussian A2C Fiber Sampling notebook.ipynb
```
- In this notebook you can define sampling Joint degree matrices for different discrete curvature and degree sequences related to different graphs. For custome problems, you need to define the design matrix $A$, initial solution $x_0$ and margin $Ax = b$. Then extract the lattice basis.
- After computing the lattice basis, simply run the trainig cell and the RL will start training.
- In the end of the notebook, you can load the trained policy, rerun it on the same fiber and compute the random sample from the fiber.
