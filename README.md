# PreMI: Logic-Minimized Tree Models for Uniform Latency Inference

This repository implements a framework for converting **Binary Decision Trees (BDTs)** and **Random Forests** into **Boolean decision representations** that enable **constant-time inference**.

The inference time of a binary decision tree (BDT) in its conventional form is determined by the tree’s depth, which limits efficiency when making predictions on new data. To overcome this limitation, we reformulate a BDT into a **Boolean expression**, allowing predictions to be computed in **constant time**, independent of the tree structure.

In this approach, every decision node of the BDT is mapped to a Boolean variable, producing a **Boolean Decision Structure (BDS)**. We further apply the **ESPRESSO logic minimization algorithm** to optimize the Boolean functions within the BDS, resulting in an enhanced representation called **Enhanced Boolean Decision Structure (EBDS)**.

Experimental results demonstrate that **EBDS achieves prediction accuracy statistically equivalent to the original BDT and the Random Forest models derived from it**, while ensuring **constant-time inference**. This constancy holds regardless of tree depth or the number of trees in the forest, eliminating the variable latency typically associated with BDT-based models.

---

## Repository Structure

```
PreMI
│
├── data/                     # Dataset files
├── output/                   # Generated experiment outputs
│
├── main.py                   # Main execution script
├── new_train.py              # Training pipeline
├── predict_func.py           # Prediction and evaluation
│
├── config.py                 # Global configuration parameters
│
├── new_RF.py                 # Random Forest prediction
├── new_IC_func.py            # Tree construction and information gain
├── new_MintermCal.py         # Minterm extraction from trees
├── new_gmm.py                # Gaussian mixture model utilities
│
├── basic_functions.py        # Utility functions
├── new_basic_functions.py    # Extended utility helpers
│
├── Evaluate_boolean.py       # Boolean expression evaluation
│
├── bds_fun.py                # Boolean Decision Structure generation
├── Eobds_fun.py              # Enhanced Boolean Decision Structure (EBDS)
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/msrinivaskgp/PreMI.git
cd PreMI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Main Python libraries used:

* numpy
* scipy
* pandas
* scikit-learn
* pyeda
* matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Configuration

Model parameters are defined in:

```
config.py
```

Example configuration:

```python
Terms = 8      # Number of Boolean terms per tree
trees = 100     # Number of trees in the forest
n_class = 2    # Number of output classes
```

---

## Training

To train the decision trees and extract Boolean rules:

```bash
python new_train.py
```

This step performs:

* Decision tree training
* Minterm extraction from tree paths
* Boolean rule generation
* Storage of intermediate results in the `output/` directory

---

## Prediction and Evaluation

To run prediction using EBDS:

```bash
python predict_func.py
```

This evaluation computes:

* Prediction accuracy
* Boolean circuit size
* Boolean circuit depth
* Cardinality of Boolean expressions
* AND / OR gate counts
* Binary Decision Diagram (BDD) statistics

---

## Output

Results and intermediate experiment files are stored in:

```
output/
```

Typical outputs include:

* Decision tree structures
* Boolean rule sets
* Pickled model outputs
* CSV result summaries

---

## Method Overview

The proposed EBDS framework follows these steps:

1. Train Binary Decision Trees or Random Forests.
2. Extract root-to-leaf paths from each tree.
3. Convert each path into Boolean expressions.
4. Construct a Boolean Decision Structure (BDS).
5. Apply **ESPRESSO logic minimization** to generate EBDS.
6. Perform predictions using minimized Boolean expressions.

This process enables **constant-time inference independent of tree depth**.

---

## Research Applications

The EBDS framework can support research in:

* Interpretable Machine Learning
* Boolean rule extraction
* Efficient inference for ensemble models
* Hardware-efficient machine learning implementations
* Model compression and optimization

---

## Author

Maddimsetti Srinivas,
PhD Researcher,
Indian Institute of Technology Kharagpur.

Debdoot Sheet,
Associate Professor,
Indian Institute of Technology Kharagpur.

---
