# Generative-Adversarial-Network-GAN-Vanilla---MNIST
This project implements a Vanilla GAN using TensorFlow to generate handwritten digits from the MNIST dataset.
It includes:

* Generator & Discriminator models
* Training pipeline
* Image generation
* Evaluation using FID (Fréchet Inception Distance)

---
# What is a GAN?

A Generative Adversarial Network (GAN) consists of two neural networks:
* Generator (G):Generates fake images from random noise
* Discriminator (D):Distinguishes real vs fake images
They compete in a minimax game.
---
# How GAN Works
1. Sample random noise → Generator creates fake image
2. Discriminator evaluates:
   * Real image → label = 1
   * Fake image → label = 0
3. Loss is computed
4. Both networks update:
   * Generator improves realism
   * Discriminator improves detection
---

# Mathematical Formulation

# GAN Objective Function

$$
\min_G \max_D V(D, G) = \mathbb{E}*{x \sim p*{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

---

# Generator Loss
$$
L_G = -\mathbb{E}[\log D(G(z))]
$$

---

# Discriminator Loss

$$
L_D = -\left(\log D(x) + \log(1 - D(G(z)))\right)
$$

---

# Evaluation Metric

# FID Score (Fréchet Inception Distance)

$$
FID = ||\mu_r - \mu_f||^2 + Tr(\Sigma_r + \Sigma_f - 2(\Sigma_r \Sigma_f)^{1/2})
$$
Lower FID = better quality

---

# Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# How to Run

# 1. Train GAN

```bash
python train.py
```

# 2. Generate Images

```bash
python generate.py
```

# 3. Compute FID

```bash
python fid.py
```

---

# Libraries Used

* TensorFlow
* NumPy
* Matplotlib
* tqdm
* Pillow
* SciPy

---

# Sample Output

Generated digits improve over epochs.

---

# Key Concepts Covered

* Deep Learning
* Adversarial Training
* Latent Space Representation
* Binary Cross Entropy Loss
* CNN/Dense Architectures
* Image Generation
* Evaluation Metrics (FID)

---

# License

This project is licensed under the MIT License.

---

# Author

Poorva Verma
