# dualPredictor: An Open-Source Tool for Simultaneously Grade Prediction and At-Risk Student Classification

by Dong, Cheng, and Kan

## Introduction

The dualPredictor is an innovative educational analytics tool that combines regression analysis with binary classification to forecast student academic outcomes and identify at-risk students. This user guide provides a step-by-step walkthrough on how to install and use the dualPredictor package.

## Installation

You can install the dualPredictor package via PyPI or GitHub. Choose one of the following methods:

### PyPI Installation

```bash
pip install dualPredictor
```
### GitHub Installation
```bash
pip install git+https://github.com/098765d/dualPredictor.git
```

## Getting Started
**1. Import the Package:** Import the dualPredictor package in your Python environment.
```python
from dualPredictor import DualModel, model_plot
```
**2. Model Initialization:** Create a DualModel instance by specifying the regression model type ('lasso', 'ridge', or 'ols'), the metric for cutoff tuning ('f1_score', 'f2_score', or 'youden_index'), and a default cutoff value.
```python
model = DualModel(model_type='lasso', metric='f1_score', default_cut_off=2.5)
```
**3. Model Fitting:** Fit the model to your dataset using the fit method.
```python
model.fit(X, y)
```
- X: The input training data (pandas DataFrame).
- y: The target values (predicted grades).

**4. Predictions:** Use the prediction method to generate grade predictions and at-risk classifications.
  ```python
predictions = model.predict(X)
```

**5.Visualization:** Visualize the model's performance using the model_plot module.
```python
# Scatter plot for regression analysis
model_plot.plot_scatter(y_pred, y_true)

# Confusion matrix for binary classification
model_plot.plot_cm(y_label_true, y_label_pred)

# Feature importance plot
model_plot.plot_feature_coefficients(coef=model.coef_, feature_names=model.feature_names_in_)
```

## Notes for Practice
- Choose the regression model and metric based on your dataset's characteristics and the specific requirements of your educational context.
- Set a well-informed default cutoff for binary classification to define at-risk students accurately.
- Utilize dynamic cutoff tuning to refine classification thresholds, especially in imbalanced datasets.
The dualPredictor tool empowers educators with data-driven insights to enhance learning outcomes and support at-risk students. By following this guide, you can effectively harness the power of the dualPredictor to make informed decisions in your educational setting.
