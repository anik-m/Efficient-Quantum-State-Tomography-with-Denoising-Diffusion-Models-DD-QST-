# Efficient Quantum State Tomography with Denoising Diffusion Models (DD-QST)

This repository contains the official implementation for the research project "Efficient Quantum State Tomography with Denoising Diffusion Models." The project aims to develop a scalable, noise-robust protocol for QST by framing it as a conditional generation problem solved by a classical Denoising Diffusion Model (DDM).

## Project Overview

Quantum State Tomography (QST) is a critical tool for benchmarking quantum devices, but it suffers from the "curse of dimensionality," requiring a number of measurements that scales exponentially with the number of qubits. This project leverages a state-of-the-art Denoising Diffusion Model, implemented in PyTorch, to reconstruct a high-fidelity density matrix from a limited and noisy set of classical measurement outcomes.

The core components of this project are:
- **Data Simulation:** A robust pipeline using `qiskit-experiments` to generate realistic, noisy measurement data for various quantum states.
- **Classical Baselines:** Implementation and benchmarking of standard QST methods like Linear Inversion and Maximum Likelihood Estimation (MLE) for performance comparison.
- **DD-QST Model:** A U-Net based Denoising Diffusion Model conditioned on measurement data to iteratively reconstruct the quantum state. The model incorporates structure-preserving mechanisms to ensure the physical validity of the outputted density matrix.
