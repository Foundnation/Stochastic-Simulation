# Stochastic Simulation

# 1. Monte Carlo Area Estimation of Mandelbrot Set

This repository contains code for assignments related to the Monte Carlo area estimation of the Mandelbrot set. The Mandelbrot set is a famous fractal in mathematics that is explored using iterative algorithms. In this assignment, Monte Carlo methods are used to estimate the area of the Mandelbrot set.

## Files

### `mandelbrot.py`

Contains core functions for the Monte Carlo area estimation of the Mandelbrot set. This file includes functions essential for iterative calculations, as well as some plotting functions related to the Mandelbrot set.

### `ortho_sampling.py`

Includes code for orthogonal sampling, which is later utilized in `mandelbrot.py`. Orthogonal sampling is a technique used to sample points in a specific pattern that aids in the calculation process for the Mandelbrot set.

### `mandelbrot_notebook.py`

This file contains the results obtained from the execution of the code. It serves as a notebook showcasing the computed results and visualizations to the area estimation of the Mandelbrot set and methods convergence.

## Usage

To utilize the code in this repository, follow these steps:

1. Clone the repository to your local machine:

2. Open the files using your preferred Python environment or IDE.

3. Run the `mandelbrot_notebook.py` file or explore the code within individual files to understand the implementations and results obtained.

# Queueing Theory

This repository contains notebooks and resources related to Queueing Theory. Queueing Theory involves the study of queues or waiting lines and explores how entities move through these queues, their behaviors, and the overall system's performance.

## Notebooks

### 1. simulations_and_analysis_part1.ipynb

This notebook covers practical simulations and analyses related to Queueing Theory. It dives into fundamental concepts, simulates various queueing models, and performs analytical analyses of these models. 

#### Key topics covered:
- Simulation of different queueing models (M/M/1, M/M/n, etc.)
- Performance metrics and analysis (waiting times)

### 2. simulations_and_analysis_part2.ipynb

This notebook extends the exploration of Queueing Theory by delving deeper into advanced simulations and analyses. It builds upon the concepts introduced in part 1 and explores more complex scenarios, possibly involving multiple queues, different arrival patterns, or service strategies.

#### Key topics covered:
- Parameter analysis for number of simulations
- Analysis of models with SJF scheduling
- Comparative study of queueing models for FIFO and SJF scheduling

### 3. theory.ipynb

This notebook provides the theoretical expected waiting time and delay probabilitiy and some visualizations.

# 3. Traveling Salesman Problem Solver

This repository contains implementations and experimental setups for solving the Traveling Salesman Problem (TSP) using Simulated Annealing algorithm.

## Contents

- `tsp_annealing.py`: Main source file containing most of the implemented functions for the Annealing algorithm, data management, and concurrency.
- `tsp_notebook.ipynb`: Jupyter notebook containing experimental setups for calculations and some plots related to the TSP.
- `analyse_data.ipynb`: Jupyter notebook that loads generated data from `tsp_notebook.ipynb` and performs data analysis and additional plotting.
- `generated_data/`: Folder containing generated data for the TSP.

## Chapter 3: Simulated Annealing for Solving the Traveling Salesman Problem

Simulated Annealing is a metaheuristic algorithm used to find an approximate solution to optimization problems. It's particularly effective for solving combinatorial optimization problems such as the Traveling Salesman Problem (TSP).

### Implementation Overview

The `tsp_annealing.py` file serves as the core implementation for the Simulated Annealing algorithm tailored to solve the TSP. This file contains the necessary functions to:

- Define the TSP problem.
- Implement the Annealing algorithm with appropriate cooling schedules and permutation operators.
- Manage data related to cities, distances, and permutations.
- Handle concurrency for improved performance.


### Usage

To utilize the implementation and perform experiments:

1. Access `tsp_notebook.ipynb` for setting up and executing various experiments related to solving the TSP using Simulated Annealing.
2. For in-depth analysis and additional plotting, refer to `analyse_data.ipynb` after generating data in the notebook.



## Dependencies

This project is built using Python and may require additional dependencies. The primary libraries used include:

- `numpy`
- `matplotlib`
- `simpy`

Use requirements.txt to find other dependecies




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
