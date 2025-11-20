# COMET: Conditionally-Overlapping Mixture of Experts

Paper: [More Experts Than Galaxies: Conditionally-Overlapping Experts With Biologically-Inspired Fixed Routing](https://openreview.net/pdf?id=1qq1QJKM5q)

## Overview

<details>
<summary>About this repository</summary>

<br>

This repository contains the official code for reproducing results from the paper:  

**"More Experts Than Galaxies: Conditionally-Overlapping Experts With Biologically-Inspired Fixed Routing"**  
by Sagi Shaier, Francisco Pereira, Katharina von der Wense, Lawrence E. Hunter, and Matt Jones.  

The code enables training and evaluating sparse modular neural networks based on our proposed COMET architecture on benchmark datasets.

</details>

---

## Prerequisites
- Python 3.11
- Conda (for environment management)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Shaier/COMET.git
cd COMET
```

### 2. Set Up the Conda Environment
Create and activate a Conda environment with Python 3.11:
```bash
conda create -n COMET python=3.11
conda activate COMET
```

### 3. Install Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

---

## Usage

### Run Experiments
To reproduce standard model experiments (4-layer MLPs on CIFAR-10, illustrating the effect of capacity and sparsity — Figures 4 and 9 in the paper):
```bash
python run_models.py
```

### Plot Results
To visualize experimental results:
```bash
python plot_results.py
```

---

## Citation

If you use this codebase or refer to our paper, please cite:

```bibtex
@misc{shaier2025expertsgalaxiesconditionallyoverlappingexperts,
      title={More Experts Than Galaxies: Conditionally-overlapping Experts With Biologically-Inspired Fixed Routing}, 
      author={Sagi Shaier and Francisco Pereira and Katharina von der Wense and Lawrence E Hunter and Matt Jones},
      year={2025},
      eprint={2410.08003},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.08003}, 
}
```

---
