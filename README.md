# Scalable Quantum State Tomography via Conditional Denoising Diffusion Models

## 1. The Verification Crisis

As quantum hardware scales into the **Noisy Intermediate-Scale Quantum (NISQ)** era (50-100 qubits), the protocols required to verify these devices have hit a bottleneck. The standard method, **Quantum State Tomography (QST)**, suffers from the *curse of dimensionality*.

The number of parameters required to describe a quantum state scales as ( d^2 - 1 = 4^N - 1 ), meaning full tomography becomes intractable for systems larger than ( N = 10 ) qubits.

This project proposes a **sample-efficient** alternative: framing tomography as a generative modeling problem using **Conditional Denoising Diffusion Models (cDDMs)**.

## 2. The Solution: Generative Tomography

Instead of explicitly calculating exponentially many density matrix coefficients, we train a neural network to learn the underlying probability distribution of measurement outcomes. If the model can generate synthetic bitstrings indistinguishable from experimental data, it has effectively captured the quantum state.

### Why Diffusion?

While **GANs** and **VAEs** have been explored for this task, they suffer from specific failures in the quantum context:

* **GANs** suffer from *"mode collapse"*, often predicting pure states even when the actual system is mixed due to decoherence.
* **VAEs** tend to produce *"blurred"* distributions, failing to capture high-frequency phase correlations necessary for entanglement verification.
* **Autoregressive Models (e.g., ShadowGPT)** introduce an *"ordering bias"*, breaking the permutation symmetry of entangled states like GHZ states.

Diffusion models optimize a variational lower bound, forcing the model to cover the full support of the data distribution. This makes them uniquely suited for **mixed-state tomography**.

<!-- ## 3. Methodology & Architecture -->
<!---->
<!-- ### 3.1 Conditional Architecture (FiLM) -->
<!---->
<!-- A quantum state looks different depending on the measurement basis (e.g., ( X, Y, Z )). Therefore, the generative model must be **conditioned** on the measurement settings. We use **Feature-wise Linear Modulation (FiLM)** to inject this physical context into the neural network. -->
<!---->
<!-- Mathematically, the basis vector ( \mathbf{c} ) modulates the internal features ( \mathbf{h}_l ) of the network via learned scale (( \gamma )) and shift (( \beta )) parameters: -->
<!---->
<!-- [ -->
<!-- \mathbf{h}_{l+1} = \text{Activation} \left( \gamma_l(\mathbf{c}) \odot \mathbf{h}_l + \beta_l(\mathbf{c}) \right) -->
<!-- ] -->
<!---->
<!-- This affine transformation effectively "rotates" the latent representation to match the measurement basis, analogous to a unitary rotation in Hilbert space. -->
<!---->
<!-- ### 3.2 Discrete Diffusion (D3PM) -->
<!---->
<!-- Standard diffusion models use Gaussian noise, but quantum measurement data consists of discrete bitstrings (( 0 ) or ( 1 )). To align the machine learning model with the physics of the device, we implement **Discrete Denoising Diffusion Probabilistic Models (D3PM)**. -->
<!---->
<!-- The forward diffusion process is modeled as a **Bit Flip Channel** (or Depolarizing Channel), which is mathematically identical to the Readout Error observed in quantum hardware. The transition matrix ( Q_t ) for a single qubit is given by: -->
<!---->
<!-- [ -->
<!-- Q_t = \begin{bmatrix} -->
<!-- 1 - \beta_t & \beta_t \ -->
<!-- \beta_t & 1 - \beta_t -->
<!-- \end{bmatrix} -->
<!-- ] -->
<!---->
<!-- Where ( \beta_t ) represents the probability of a bit flip at timestep ( t ). The model is trained using **Cross-Entropy loss** to reverse this process and reconstruct the clean bitstring: -->
<!---->
<!-- [ -->
<!-- L = \mathbb{E}*{q} \left[ -\log p*\theta(x_0 | x_t, \mathbf{c}) \right] -->
<!-- ] -->
<!---->
## 4. Roadmap

| **Phase**          | **Innovation**              | **Target System**    | **Status**   |
| ------------------ | --------------------------- | -------------------- | ------------ |
| **1. Foundation**  | Basis Conditioning (MLP)    | 1-Qubit / Bell State | **Complete** |
| **2. Scalability** | Transformer Backbone + FiLM | 5-8 Qubit GHZ        | **Active**   |
| **3. Refinement**  | Physics-Aware Noise Models  | Mixed Werner States  | Planned      |

---

## 5. Further Reading

* **Quantum State Tomography with Conditional Generative Adversarial Networks**
  *Ahmed, Shahnawaz et al.* (2021). *Physical Review Letters*.
  A pioneering approach using GANs for tomography, though noting the stability issues our Diffusion approach aims to solve.

* **Mixed-state quantum denoising diffusion probabilistic model**
  *Kwun, Gino et al.* (2025). *Physical Review A*.
  Recent work extending diffusion to mixed states, validating the theoretical path of this project.

* **Generative quantum machine learning via denoising diffusion probabilistic models**
  *Zhang, Bingzhi et al.* (2024). *Physical Review Letters*.
  Foundational work on applying DDPMs to quantum data generation.

* **ShadowGPT: Learning to Solve Quantum Many-Body Problems from Randomized Measurements**
  *Yao, Jian and You, Yi-Zhuang* (2024). *arXiv preprint*.
  Investigates autoregressive transformers (ShadowGPT) for tomography; this project benchmarks against ShadowGPT to test for ordering bias.

* **Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning**
  *Chen, Ting et al.* (2023). *ICLR*.
  Key reference for handling discrete data in diffusion models (Bit Diffusion).

* **Structured Denoising Diffusion Models in Discrete State-Spaces**
  *Austin, Jacob et al.* (2021). *NeurIPS*.
  The primary reference for the D3PM (Discrete Diffusion) mathematical framework used in Phase 2.

* **Learning hard quantum distributions with variational autoencoders**
  *Rocchetto, Andrea et al.* (2018). *npj Quantum Information*.
  Early work using VAEs for quantum states; provides a baseline for sample complexity.
<!---->
<!-- * **Predicting many properties of a quantum system from very few measurements** -->
<!--   *Huang, Hsin-Yuan et al.* (2020). *Nature Physics*. -->
<!--   The seminal paper on "Classical Shadows," a randomized measurement technique relevant to our benchmarking. -->
<!---->
* **Attention-based quantum tomography**
  *Cha, Peter et al.* (2021). *Machine Learning: Science and Technology*.
  Explores attention mechanisms for QST, supporting our Phase 2 shift to Transformers.
<!---->
<!-- * **Neural-network quantum state tomography** -->
<!--   *Torlai, Giacomo et al.* (2018). *Nature Physics*. -->
<!--   One of the first demonstrations of using neural networks (RBMs) to represent quantum states. -->
<!---->
* **Reconstructing quantum states with generative models**
  *Carrasquilla, Juan et al.* (2019). *Nature Machine Intelligence*.
  A broad overview of generative models in QST, establishing the comparison between RBMs, VAEs, and GANs.
