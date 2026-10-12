# Kaggle Donor-Availability Classification Experiment Report

## Boundary Limitation
This experiment predicts the Kaggle dataset's `availability` label only. This is a general availability classification and does **not** represent a validated emergency recruitment-response probability. This probability must not be fed directly into the UDI field during emergency simulations unless properly calibrated against authorized invitation-response data. The UDI simulation will continue to use synthetic labels for its comparative experiments.

## Evaluation
The models were trained on 80% of the dataset using stratified 5-fold CV to select hyperparameters. They were then evaluated on a held-out 20% test set.

**Best Model Selected:** XGBoost
Selection was based on the highest mean ROC-AUC during 5-fold cross-validation on the training set (CV ROC-AUC: 0.4959).

Metrics, cross-validation results, and figures are available in the `reports/` directory.
