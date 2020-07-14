![Python Version](https://img.shields.io/badge/Python%20Version-3.7.1-blue.svg)
![R Version](https://img.shields.io/badge/R%20Version-3.6.0-green.svg)

# Deep learning enables therapeutic antibody optimization in mammalian cells by deciphering high-dimensional protein sequence space

## Overview

This repository contains the scripts to perform therapeutic antibody optimization with deep learning, as described in Mason et al., 2019 [[1](https://www.biorxiv.org/content/10.1101/617860v2.abstract)]. The Python scripts are found in folder [scripts/](scripts/). The resulting *in silico* predicted binders are then subjected to multiple developability filters in the R script [developability_filters.R](developability_filters.R).

### Table of contents

1. [Prepare working environment](#prepare-working-environment)
2. [Usage](#usage)
3. [License](#license)
4. [Citation](#citation)

## Prepare working environment

Before running any of the scripts, the necessary packages need to be installed. This is done with Conda, the open source package management system [[2](https://docs.conda.io/)], and the environment can be loaded via the provided [config.yaml](config.yaml) file, using following commands:

```
conda env create -f config.yaml -n $Project_Name
source activate $Project_Name
```

## Usage

### Deep learning: Python scripts

The full deep learning analysis, written in Python, can be summarized into three consecutive steps:

 1. Compare classification performance of different machine learning models on rationally designed site-directed mutagenesis libraries.
 2. The best-performing model, a convolutional neural network (CNN), is tuned with a randomized search on hyper parameters.
 3. The optimized CNN model is used to identify antigen-binding sequences from an *in-silico* generated library.

Step 1 and 3 are performed simultaneously, by running the following command:

`python scripts/main.py`

This will produce the folder [figures](figures/), where the performance of different models is visualized, and the folder [classification](classification/) with the CNN model summary and the predicted values for the *in-silico* generated library.

Inside the main script, hyper parameters were already selected according to the optimized model. However, the randomized search can be run again:

`python scripts/model_tuning.py`

A folder [model_tuning](model_tuning/) will be created, containing the best hyper paramter settings and the corresponding mean cross-validated score. Those parameters can then be included inside the main script (`params`).

### Applying developability filters

With the results from the previous analysis, developability filters can be applied to the *in silico* generated library:

`R --vanilla < developability-filters.R`

The developability filter based on CamSol solubility scores [[3](http://dx.doi.org/10.1016/j.jmb.2014.09.026)]-[[4](https://www.nature.com/articles/s41598-017-07800-w)] needs to be run on their [web server](http://www-mvsoftware.ch.cam.ac.uk/index.php/camsolintrinsic). Additionally, netMHCIIpan [[5](https://www.ncbi.nlm.nih.gov/pubmed/29315598)], version 3.2, needs to be downloaded and installed under following [link](https://services.healthtech.dtu.dk/service.php?NetMHCIIpan-3.2), otherwise the R script will terminate.

## Datasets used

DMS23_Library

DMS23_HEL1

DMS23_HEL2

DMS23_HEL3_High

DMS23_HEL3_Low

## [License](https://raw.githubusercontent.com/dahjan/DMS_opt/master/LICENSE.md)

## Citation

If you use the the code in this repository for your research, please cite our paper.

```
@article {Mason617860,
	author = {Mason, Derek M and Friedensohn, Simon and Weber, C{\'e}dric R and Jordi, Christian and Wagner, Bastian and Meng, Simon and Gainza, Pablo and Correia, Bruno E and Reddy, Sai T},
	title = {Deep learning enables therapeutic antibody optimization in mammalian cells by deciphering high-dimensional protein sequence space},
	year = {2019},
	doi = {10.1101/617860},
	URL = {https://www.biorxiv.org/content/early/2019/05/30/617860},
	journal = {bioRxiv}
}
```

## Authors

Derek Mason, Jan Dahinden, Simon Friedensohn, Bastian Wagner, Cédric Weber, Sai Reddy
