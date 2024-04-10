# dualPredictor: An Open-Source Tool for Simultaneously Grade Prediction and At-Risk Student Classification

by Dong, Cheng, and Kan

## Introduction

The dualPredictor is an innovative educational analytics tool that combines regression analysis with binary classification to forecast student academic outcomes and identify at-risk students. This user guide provides a step-by-step walkthrough on how to install and use the dualPredictor package.

## Motivation
The motivation behind the dualPredictor package is to make the use of complicated models as simple as possible for all users with limited code experience. The model package is designed using the same syntax as the popular scikit-learn models, making it easier for users with experience in scikit-learn to quickly start using the dualPredictor with just a single line of code. The attributes, model fitting (model.fit(X, y)), and prediction methods (model.predict(X)) are intentionally designed to mimic the scikit-learn model object, providing a familiar and user-friendly experience for data scientists and educators alike.
```python
# intialize the model, specify the parameters
model = DualModel(model_type='lasso', metric='f1_score', default_cut_off=2.5)
```

| Model Methods | Description |
|--------------|-------------|
| `fit(X, y)`  | - **X**: The input training data, pandas data frame. <br> - **y**: The target values (predicted grade). <br> - **Returns**: Fitted DualModel instance |
| `predict(X)` | - **X**: The input training data, pandas' data frame. |

| Model Attributes   | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `alpha_`           | The value of penalization in Lasso and ridge, for OLS alpha = 0 |
| `coef_`            | The coefficients of the model                                  |
| `Intercept_`       | The intercept value of the model                               |
| `feature_names_in_`| Names of features during model training                        |
| `optimal_cut_off`  | The optimal cut-off value that maximizes the metric            |

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
model = DualModel(model_type='lasso', metric='youden_index', default_cut_off=2.5)
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

## References

- Fluss, R., Faraggi, D., & Reiser, B. (2005). Estimation of the Youden Index and its associated cutoff point. _Biometrical Journal: Journal of Mathematical Methods in Biosciences_, 47(4), 458-472.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. _Technometrics_, 12(1), 55-67.
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. _The Journal of Machine Learning Research_, 12, 2825-2830.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical Society Series B: Statistical Methodology_, 58(1), 267-288.
