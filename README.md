# Continuous Kernel Message Passing Neural Network (MPNN)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-3C2179?style=for-the-badge&logo=python&logoColor=white)](https://pytorch-geometric.readthedocs.io/)

> **A physics-aware Geometric Deep Learning model that predicts quantum properties by explicitly modeling continuous interatomic distances.**

## Overview
This repository hosts the implementation of a **Continuous Kernel-based Message Passing Neural Network (MPNN)** designed to predict thermodynamic stability (Formation Energy) and electronic structure (Band Gap) of crystal materials.

Unlike standard Graph Neural Networks that treat chemical bonds as static edges, this architecture utilizes **Dynamic Filter Generation** to learn a continuous mapping from geometric distance to interaction strength. Trained on the **Open Quantum Materials Database (OQMD)**, it serves as a high-speed surrogate for computationally expensive Density Functional Theory (DFT) simulations.

## Key Results
The model achieves state-of-the-art performance, particularly in modeling complex electronic properties where standard GNNs often struggle.

| Target Property | R² Score | MAE (eV) | Inference Speed |
| :--- | :--- | :--- | :--- |
| **Formation Energy** | **0.922** | **0.137** | **0.21 ms** |
| **Band Gap** | **0.813** | **0.204** | **0.21 ms** |

> **Impact:** Reduces inference latency by **six orders of magnitude (10^6x)** compared to traditional DFT calculations, enabling the screening of millions of candidates in minutes.

## Methodology & Architecture

### 1. Graph Representation
Crystal structures are converted into undirected multigraphs where:
* **Nodes:** Represent atoms, encoded with 5 fundamental features (Atomic Number, Group, Period, Mass, Radius).
* **Edges:** Represent chemical bonds, treated not as binary links but as continuous variables expanded using a **Gaussian Radial Basis Function (RBF)** basis.

### 2. Continuous Kernel Convolution (NNConv)
The core of the architecture is the **NNConv** operator. Instead of learning a fixed set of weights for all edges, the model generates a unique weight matrix $\mathbf{\Theta}$ for *each* specific bond based on its length.

$$\mathbf{x}'_i = \mathbf{\Theta} \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot h_{\mathbf{\Theta}}(\mathbf{e}_{i,j})$$

### 3. Edge Neural Network (EdgeNN)
A secondary neural network (EdgeNN) acts as the filter generator:
1.  **Input:** High-dimensional RBF distance vector of the bond.
2.  **Process:** Linear Layer $\to$ SiLU Activation $\to$ Linear Layer.
3.  **Output:** Dynamic convolution filter specific to that bond's geometry.

This allows the model to respect the physical reality that atomic interaction strength decays continuously with distance.

