# Predicting-Stroke-Risk-Using-Machine-Learning
# Project Overview

This project applies supervised machine learning techniques to predict stroke risk using a publicly available healthcare dataset. The goal is to walk through the complete predictive modeling workflow data exploration, visualization, preprocessing, model building, and performance evaluation within a health informatics context.

We implemented two classification models, Decision Tree and Naïve Bayes, to evaluate how well common supervised learning algorithms perform in predicting a binary clinical outcome. This project was completed for HI 2022 (Introduction to Python for Health Informatics).

# Dataset Description

* Source: Publicly available stroke dataset (Kaggle) 
* Format: CSV
* Rows: ~5,100
* Columns: Demographic, behavioral, and clinical features
* Outcome Variable:
 `stroke` (0 = No stroke, 1 = Stroke)

* Feature Types:

  * Numerical (age, glucose level, BMI)
  * Categorical (gender, work type, smoking status, hypertension, heart disease)

The dataset is imbalanced, with far fewer stroke cases than non-stroke cases, which impacts model performance.

# Project Steps 

# Step 1 — Dataset Selection

A publicly available, supervised learning ready dataset on stroke prediction was selected.

# Step 2 — Loading and Exploring the Dataset

The dataset was loaded into Python. Shape, data types, missing values, and feature distributions were examined.

# Step 3 — Data Visualization

Multiple visualizations (histograms, bar charts, scatterplots) were created to explore feature patterns.

# Step 4 — Preparing the Dataset for Machine Learning

* Encoded categorical variables
* Normalized numerical features
* Selected predictors (X) and outcome (y)
* Train/test split
* K-fold cross-validation applied during evaluation

# Step 5 — Building Predictive Models

Two supervised models were implemented:

* Decision Tree Classifier
* Naïve Bayes (GaussianNB)

Both were evaluated using accuracy, precision, and recall.

# Step 6 — Model Comparison

Model outputs were compared to identify which algorithm performed best overall and for the clinical target class.

# Step 7 — Conclusion 

A written summary of findings, limitations, and future improvement opportunities was developed (included in the .ipynb file).

# Model Performance Summary

# Naïve Bayes (GaussianNB)

| Metric                | Result   |
| --------------------- | -------- |
| Accuracy              | 0.86     |
| Precision (No Stroke) | 0.97     |
| Recall (No Stroke)    | 0.89     |
| Precision (Stroke)    | 0.16     |
| Recall (Stroke)       | 0.42     |


# Decision Tree

| Metric                | Result                          |
| --------------------- | ------------------------------- |
| Accuracy              | 0.91                            |
| Precision (No Stroke) | 0.96                            |
| Recall (No Stroke)    | 0.95                            |
| Precision (Stroke)    | 0.15                            |
| Recall (Stroke)       | 0.18                            |

---

# Performance Interpretation

The Decision Tree achieved the highest overall accuracy (0.91) and performed strongly on the majority “No Stroke” class. The Naïve Bayes model had slightly lower accuracy (0.86) but also performed well on majority class predictions.

Both models struggled to correctly identify stroke cases due to class imbalance, with low precision and recall for the stroke class. The Decision Tree was the stronger model overall, but additional techniques such as SMOTE oversampling, hyperparameter tuning, or more advanced models (e.g., Random Forest) could improve minority-class detection.

---

# Repository Structure

```
/project-root
│
├── Final_Project_for_Submission.ipynb     # Jupyter Notebook implementation
├── healthcare-dataset-stroke-data.csv     # Dataset used for modeling
└── README.md                              # Project documentation
```

---

# How to Run This Project

1. Install Python 3.x

2. Install required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook Final_Project_for_Submission.ipynb
   ```

   or upload it to **Google Colab**.

4. Run all cells in order to reproduce the results.
