![Python Version](https://img.shields.io/badge/Python%20Version-3.6.5-blue.svg)
![R Version](https://img.shields.io/badge/R%20Version-3.6.0-blue.svg)

## Overview
---

This repository contains the scripts to perform therapeutic antibody optimization with deep learning, as described in Mason et al., 2019 [1].

This repository summarizes the pipeline that was used to op

### Table of contents

1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Prepare working environment](#prepare-working-environment)
4. [Models to test](#models-to-test)
5. [Task list](#task-list)
6. [Datasets used](#datasets-used)
---

This github repository should be used to store and share useful scripts and analysis steps for the DMS+ML project.

## Repository Structure

The scripts folder should contain useful helper scripts that we want to reuse  several times. It also includes the data.py file which contains a fixed train/test split of the dataset. The data can be loaded via:
```python
from scripts.data import dms23

# The train fold can be accessed like this (and is stored as a pandas dataframe)
dms23.train

# Test fold
dms23.test

# Complete dataset
dms23.complete

```
The notebook directory contains all the results for each individual model. Eventually, we could create a summary file and put this in the main directory. The data folder contains the original .csv file Derek sent around. References can be used to store papers or documentation files of individual packages. The figures folder can be used to store figures produced during any of the analysis steps.

## Prepare working environment

In order to ensure that we are using the same packages and package versions, I suggest that we use conda as our environment + package manager. Conda makes it easy to load an environment from .yaml files via the following command:

```
conda env create -f config.yaml -n $Project_Name 
```

You can then load your new environment using:

```
source activate $Project_Name
```

If you want to install a new package, you can install it via conda like this:

```
conda install $PackageName
```

If you updated you environment, update the .yaml file via:

```
conda env export > config.yaml
```

And push your new .yaml file to the repository. Just notify the rest of us on slack so that we can update our environments as well. I already included many packages in the original .yaml file, but feel free to add more packages if you need them.

```
source activate $Project_Name
conda env update -f=config.yaml
```

## Models to test

- Logistic Regression (Bastian/Simon)
- K-Nearest Neighbors (Derek/Cedric)
- Support Vector Machine (Cedric)
- Naive Bayes (Derek/Bastian)
- Decision Tree/Random Forest/Gradient boosted trees (Simon)
- RNN (Derek/Cedric)
- CNN (Bastian/Simon)
- Multi-layer Perceptron (Simon)

## Task list

- [ ] Finish [overview](https://github.com/SimFri/DMS_Analysis/blob/master/notebooks/Overview.ipynb) notebook
- [ ] Finish logistic regression notebook
- [ ] Finish K-NN notebook
- [ ] Finish SVM notebook
- [ ] Finish Naive Bayes notebook
- [ ] Finish Random Forest notebook
- [ ] Finish RNN notebook
- [ ] Finish CNN notebook
- [ ] Finish MLP notebook
- [ ] Simulate in-silico sequences

## Datasets used

DMS23_Library

DMS23_HEL1

DMS23_HEL2

DMS23_HEL3_High

DMS23_HEL3_Low

## Cite
---

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1] Mason, D., Friedensohn, S., Weber, C., Jordi, C., Wagner, B., Meng, S., Gainza, P., Correia, B., and Reddy, S. "Deep learning enables therapeutic antibody optimization in mammalian cells by deciphering high-dimensional protein sequence space." *bioRxiv* 617860 (**2019**)

## Authors

Jan Dahinden, Derek Mason, Bastian Wagner, Cédric Weber, Simon Friedensohn, Sai Reddy
