# DualPredictor

`DualPredictor` is an innovative Python package designed for educational analytics, offering a novel approach by providing simultaneous regression and binary classification results. Utilizing well-established regression models such as Lasso, Ridge, and OLS (Ordinary Least Squares) from scikit-learn, `DualPredictor` goes a step further by introducing a cutoff-based binary classification. This dual approach allows users not only to predict student grades but also to identify at-risk students efficiently, bridging the gap between traditional regression and classification methods.

## Features

- **Simultaneous Predictions**: Seamlessly perform regression and binary classification in a single step.
- **Flexible Model Selection**: Choose from LassoCV, RidgeCV, or LinearRegression as the base regression model.
- **Dynamic Cutoff Tuning**: Automatically tunes the cutoff value to maximize the Youden index, F1 score, or F2 score, making it particularly suited for educational settings where identifying at-risk students is crucial.
- **Ease of Use**: Designed to follow scikit-learn's familiar API, making it accessible for both beginners and experts in machine learning.

## Installation

Install `DualPredictor` directly from PyPI using pip:

```bash
pip install DualPredictor
```
## Example Usage

Install `DualPredictor` directly from PyPI using pip:

```python
from sklearn.datasets import fetch_california_housing
from DualPredictor.dual_model import DualModel
from DualPredictor.model_plot import plot_scatter,plot_feature_coefficients,plot_cm

# Fetching a dataset from scikit-learn for demonstration purposes
housing = fetch_california_housing(as_frame=True)
y = housing.target  # Target variable (e.g., housing prices)
X = housing.data  # Feature matrix

# Initializing and fitting the DualModel
# 'ols' for Ordinary Least Squares, a default cut-off value is provided
# The metric parameter specifies the method to tune the optimal cut-off
dual_clf = DualModel(model_type='ols', default_cut_off=2.5)
dual_clf.fit(X, y, metric='youden_index')

# Accessing the true binary labels generated based on the default cut-off
y_label_true = dual_clf.y_label_true_
# Retrieving the optimal cut-off value tuned based on the Youden Index
optimal_cut_off = dual_clf.optimal_cut_off

# Predicting grades and binary classification (at-risk or not) based on the optimal cut-off
y_pred, y_label_pred = dual_clf.predict(X)

# Visualizations
# Plotting the actual vs. predicted values to assess regression performance
scatter_plot_fig = plot_scatter(y_pred, y)
# Plotting the confusion matrix to evaluate binary classification performance
cm_plot = plot_cm(y_label_true, y_label_pred)
# Plotting the non-zero coefficients of the regression model to interpret feature importance
feature_plot = plot_feature_coefficients(coef=dual_clf.coef_, feature_names=dual_clf.feature_names_in_)

```
