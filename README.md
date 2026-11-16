# Hazard Detection for Human Intervention in Autonomous Robot Navigation

This repository contains the code for my CSCI 5622 final project using the Hazards&Robots dataset.

The goal of the project is to predict whether a robot camera frame contains a hazard that should cause a human operator to intervene. The main metric is **precision while maintaining at least 95% hazard recall**.

## Data

The project uses the Hazards&Robots dataset, including the provided 512-dimensional CLIP embeddings and metadata labels.

Dataset:  
https://zenodo.org/records/7859211

The analysis uses:

- `training_mixed_embs.npy`
- `test_embs.npy`
- `training_mixed_frames_labels.csv`
- `test_frames_labels.csv`
- environment metadata files

Place the downloaded embeddings and metadata in:

```text
embeddings/
metadata/
```

## Running the analysis

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Then start Jupyter:

```bash
jupyter notebook
```

Open:

```text
project_final.ipynb
```

and run all cells from top to bottom.

The notebook reproduces the headline result:

> Logistic Regression achieved approximately **89.77% precision at at least 95% hazard recall** on the final test set.

## Reproducibility

The random seed is fixed at:

```python
SEED = 42
```

The official test set is kept separate from model tuning. Model hyperparameters and thresholds are selected using the development/validation data before final test evaluation.

## Models

The project compares:

- Always-hazard baseline
- CLIP-distance baseline
- Logistic Regression
- Histogram Gradient Boosting

It also includes PCA and K-means for unsupervised analysis and failure analysis by hazard type and environment.