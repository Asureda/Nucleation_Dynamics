# Nucleation Dynamics

## Overview
Nucleation Dynamics isdesigned to simulate and analyze nucleation processes. It leverages Julia and Python programming languages to offer a powerful and flexible environment for numerical simulations.

## I. CLUSTER DYNAMICS

### Numerical Implementation
The cluster size changes via single molecule attachment and detachment, as described by the master equations.

### Master Equations
The rate of change of the number density of *i*-mers is given by:

$$
\frac{dN_i}{dt} = D_{i+1}N_{i+1} + A_{i-1}N_{i-1} - (D_i + A_i)N_i
$$

Where:
- $ N_i $ is the number density of *i*-mers,
- $ i \geq u $, and *u* is the minimum cluster size treated numerically,
- $ A_i $ and $ D_i  4are the rates of single molecule attachment and detachment, respectively.

### Steady State Balance
At steady state ($ \theta = 0 $), the balance is given by:

$$
0 = A_{i-1}N_{eq,i-1} - D_iN_{eq,i+1}
$$

### Equilibrium Number Densities
The equilibrium number densities follow the Boltzmann distribution:

$$
N_{eq,i} = N_1 \exp\left(-\frac{W_i}{kT}\right)
$$

Where:
- $ W_i $ is the free energy of formation of the *i*-mer,
- $ k $ is the Boltzmann's constant,
- $ T $ is the temperature.

### Formation Rate of Clusters
The formation rate of clusters is concerned with the net formation rate of clusters of given size and the number density of such clusters. It is expressed as:

$$
J_i = A_iN_i - D_{i+1}N_{i+1}
$$

### Simulation Framework
- Built on [Julia](https://julialang.org/) and Python, offering flexibility and performance for numerical simulations.
- **Example Cases**: Contains an example simulation.
- **Extensible**: Easy to extend and integrate with other computational tools and libraries.

## Getting Started

### Prerequisites
- Julia (version x.x or later)
- Python (version 3.x or later)

This project uses `pyproject.toml` for Python dependency management and adopts a similar approach for Julia package management.

### Installation

#### Python Environment Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Asureda/Nucleation_Dynamics.git
   cd Nucleation_Dynamics
   pip install .

