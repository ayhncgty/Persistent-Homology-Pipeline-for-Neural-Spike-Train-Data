EVERYTHING BELOW IS TO BE COMPLETED/REFINED.
# A Persistent Homology Pipeline for Neural Spike Train Data

This repository contains the full implementation of our persistent homology–based pipeline for analyzing neural spike-train data, accompanying our paper **[arXiv Link soon]**

## Implementation Details
* To be written

## Pipeline overview

Begin with a train ensemble $\mathcal{R}$. This can be stored as a $NxT$ numpy array of $0s$ and $1s$ where each row represents a neuron's spike train (N neurons, recorded for Tms.)

To compute persistence barcode of a train ensemble:
  * Use `VP` function to compute the Victor-Purpura pairwise distance matrix.
  * Feed that distance matrix into `ripser`. persistent homology in a chosen homology dimension.
  * Use `plot_barcode` function to study its persistence barcode qualitatively.

Suppose now you have a dataset of train ensembles, each labeled with its corresponding stimulus. 
  * Compute persistent homology of each train ensemble in the dataset.
  * Compute pairwise bottleneck distances using `bottleneck_zero` if in dimension 0, otherwise use `persim.bottleneck`.
  * To assess the network's quality and the model's ability to classify stimuli:
    *    Train 1-NN classifer on the entire dataset using the labeled barcodes.
    *    Perform LOOCv to obtain a classification score.
  * You can also visualize the MDS plot of the BDM for a qualitative understanding of how the barcodes are distributed.


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
jupyter lab
# or
jupyter notebook
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
```


## Data
The datasets used in the paper are held by the Vincis lab [CITE]

## Notebooks & demos

We include Jupyter notebooks to reproduce the main experiments and to help you run the pipeline.

- Synthetic-Data-Experiment-1.ipynb
  In this notebook we show:
    * Construction of the synthetic dataset for experiment 1
    * Performing the TDA pipelin.
    * Comparison of the network approach with the individual neurons
    * Statistical significance test in classification scores.
  
- Synthetic-Data-Experiment-2.ipynb  
  In this notebook
    * We construct a synthetic train ensemble $\mathcal{R}$ with non-trivial 1-dimensional homology feature.
    * We go through how we compute the persistent homology of $\mathcal{R}$ equipped with the Victor-Purpura distance.
    * We construct another synthetic train ensemble, denoted $\mathcal{R}'$ such that its connectivity is similar to that of $\mathcal{R}$ but exhibits no 1-dimensional homology.
    * We make the point that TDA Detects Higher-Order Structure Beyond Connectivity by comparing the persistence barcodes of $\mathcal{R}$ and $\mathcal{R}'$ using bottleneck distance.

- Analysis-of-Data-From-Behaving-Mice.ipynb  
    * To be written.



## License
To be written.

## Contact
For questions about the code or dataset access, contact the corresponding author listed in the paper [citation].
