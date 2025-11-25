# CS726 — Advanced Machine Learning (Programming Assignments)

This repository contains my implementations for the three programming assignments of **CS726: Advanced Machine Learning**, covering graphical models, diffusion-based generative modeling, and large-language-model decoding.

---

## Assignment 1 — Probabilistic Graphical Models  
Building core inference machinery for undirected graphical models: triangulating graphs, constructing junction trees, and running message-passing to compute marginals and MAP estimates. The assignment includes producing the top-k most likely variable assignments and verifying correctness on structured probabilistic models.

---

## Assignment 2 — Diffusion Models and Classifier-Free Guidance  
Building a full Denoising Diffusion Probabilistic Model (DDPM) from scratch, experiment with noise schedules and sampling quality, and generate samples from a simple dataset. The assignment then extends to **conditional diffusion**, including classifier-free guidance and training/testing a classifier to steer generation.

---

## Assignment 3 — LLM Decoding & Medusa  
Exploring decoding strategies for LLaMA-2, starting with standard methods (greedy, temperature sampling, top-k, nucleus). Implementing **constrained decoding** using word-level constraints, and finally building **Medusa** decoding heads to accelerate inference. The focus is on comparing translation quality (BLEU/ROUGE) and measuring speedups from the Medusa approach.

---
