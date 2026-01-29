# A Persistent Homology Pipeline for Neural Spike Train Data

This repository implements our persistent homology based pipeline for neural spike train analysis, accompanying our paper [arXiv:2512.08637](https://arxiv.org/abs/2512.08637), and includes code to reproduce all results and figures.

## Quickstart

1. Clone the repository:
```bash
git clone https://github.com/ayhncgty/Persistent-Homology-Pipeline-for-Neural-Spike-Train-Data.git
cd Persistent-Homology-Pipeline-for-Neural-Spike-Train-Data
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate spike-train-tda
```

3. Launch JupyterLab / Notebook and open the notebooks:
```bash
conda install jupyterlab   # or: conda install notebook. This should be run inside the environment.
jupyter lab
```

## Environment & Dependencies

You can reproduce our conda environment using the provided `environment.yml`:

```yaml
name: spike-train-tda
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - pip
  - pip:
      - persim
      - ripser
      - elephant
      - neo
      - quantities
      - gudhi
```


## Implementation Details

All analyses were performed on standard laptop CPUs. The pipeline relies on:

- **Victor–Purpura distance:**  
  - We compute VP distances using the [`Elephant`](https://elephant.readthedocs.io/en/latest/) library  
    (see [Victor & Purpura, 1996](https://journals.physiology.org/doi/pdf/10.1152/jn.1996.76.2.1310)).  
  - For the special case \(q > 2\), we use a simplified implementation (`VP_trivial`).
- **Persistent homology:** computed with [`ripser.py`](https://github.com/scikit-tda/ripser.py).
- **Bottleneck distance:**
  - For degree-0 diagrams, we use a custom optimized bottleneck routine (`bottleneck_zero`), which is more computationally efficient than general-purpose implementations such as `persim.bottleneck` but applies only to degree-0 diagrams. It is based on the formula in Corollary 4.15 of [Ayhan & Needham (2025)](https://arxiv.org/abs/2506.21488).
  - For degree-1, we use the `persim` implementation from the `scikit-tda` library.
  - For the biological datasets, we additionally use `Gudhi` for bottleneck computations.


## Pipeline overview

**Note:** To run your own computations, open a Jupyter notebook and import the utility functions:

```python
from utils import *
```

Begin with a **train ensemble** $\mathcal{R}$, stored as NxT NumPy array of 0s and 1s (one row per neuron):
```python
train_ensemble_R = np.array([...])
```
Compute pairwise **Victor-Purpura distances**:
```python
vp_dm = VP_trivial(train_ensemble_R)
```
Compute **persistent homology** with `ripser` (choose a homology dimension):
```python
ph_R = ripser(vp_dm,distance_matrix = True)['dgms'][0] # say 0-dim homology
```
Visualize the barcode:
```python
plot_barcode(ph_R,r =200) # choose r large enough to show all bars
```

Suppose now you have a dataset of train ensembles, each labeled with its corresponding stimulus. 
  * Compute persistent homology for each ensemble.
  * Compute pairwise bottleneck distance matrix and store it in a numpy array `BDM`.

To assess classification performance:
- Train 1-NN classifier and,
- validate it using Leave One Out cross-validation.
Both steps are handled by the `LeaveOneOut` function applied to `BDM` and class labels:
```python
LeaveOneOut(BDM, labels)["accuracy_score"]  # returns the classification score
```
You can also visualize the MDS plot of the BDM for a qualitative understanding of how the barcodes are distributed.


## Data
- **Data From Behaving Mice:** The biological datasets used in the paper are held by the Vincis lab: [https://github.com/vincisLab/neuron-response-classification]
- **Synthetic Data:** Code for generating the synthetic datasets used in our experiments are included in this repository.

## Notebooks

We include Jupyter notebooks for reproducing the main experiments and figures from the paper.  
Each notebook is named according to the corresponding figure (e.g., `Figure-4-Synthetic-Data-Experiment-1.ipynb` reproduces Figure 4).

**Note:** The file `Data for Figure Generation.zip` contains pre-computed classification scores for varying $q$ values in the Victor–Purpura distance.  
These data are used for generating Figure 6 and accompany the notebook dedicated to that figure.


## Contact
For questions about the code or the paper, please contact the corresponding author, Tom Needham, as listed in the manuscript: https://arxiv.org/abs/2512.08637


