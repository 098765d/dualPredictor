# dualPredictor

by D

`dualPredictor` is can provide simultaneous **regression** and **binary classification** results for tabular dataset. 

## Features

- **Simultaneous Predictions**: Seamlessly perform regression and binary classification in a single step.
- **Regressor Selection (choose one)**: Choose from **Lasso**, **Ridge**, or **LinearRegression(OLS)** as the base regression model.
- **Dynamic Cutoff Tuning (choose one)**: Automatically tunes the cutoff value to maximize the Youden index, F1 score, or F2 score, user choose metric type.


<img src='https://github.com/098765d/dualPredictor/blob/890d66657e969334fb46d5b97cafbeb2d04fdff6/figs/regressor_metric.png' width="300" height="200">
  

**1. Youden Index (J)**

![](https://miro.medium.com/v2/resize:fit:842/1*LVilqC3cy4AgyC1wD4RH-A.png)

https://miro.medium.com/v2/resize:fit:842/1*LVilqC3cy4AgyC1wD4RH-A.png

 $$J= Recall + Specificity - 1$$
J is a measure of the overall performance of a binary classifier. It is calculated as the sum of the recall and specificity minus 1. A high J statistic indicates that the classifier is performing well on both positive and negative cases.

* **Recall** measures the ability of a classifier to correctly identify positive cases. A high recall means that the classifier is avoiding miss detects.
* **Specificity** measures the ability of a classifier to correctly identify negative cases. A high specificity means that the classifier is avoiding false alarms.

**2. F scores (Option F1, F2 in Package)**

$$F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$$

F1 score is another measure of the overall performance of a binary classifier. It is calculated as the harmonic mean of the precision and recall. A high F1 score indicates that the classifier is performing well on both positive and negative cases.

$$F_\beta = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\beta^2 \cdot precision + recall}$$
F-score with factor beta is a generalization of the F1 score that allows for different weights to be given to precision and recall. A beta value less than 1 indicates that the F-score is prone to precision, while a beta value greater than 1 indicates that the F-score is prone to recall.

**In educational settings**

In educational settings, it is important to avoid miss detects (i.e., failing to identify at-risk students). However, it is also important to avoid false alarms (i.e., identifying students as at-risk when they are not). Therefore, it is often desirable to use a measure that is prone to recall, such as the F1 score with beta > 1.
Youden's J statistic and the F1 score are both measures that balance the avoidance of miss detects and the avoidance of false alarms. However, the Youden's J statistic is less sensitive to false alarms (Specificity is less sensitive to false alarms compares to Precision) than the F1 score.


## Installation

Install `dualPredictor` directly from PyPI using pip:

```bash
pip install dualPredictor
```
## Example Usage


```python
from sklearn.datasets import fetch_california_housing
from dualPredictor.dual_model import DualModel
from dualPredictor.model_plot import plot_scatter,plot_feature_coefficients,plot_cm

# Fetching a dataset from scikit-learn for demonstration purposes
housing = fetch_california_housing(as_frame=True)
y = housing.target  # Target variable (e.g., housing prices)
X = housing.data  # Feature matrix
```
### Train Model based on X and y

choose model_type and metric

```python
# Initializing and fitting the DualModel, 'ols' for Ordinary Least Squares, a default cut-off value is provided
dual_clf = DualModel(model_type='ols', default_cut_off=2.5)
# The metric parameter specifies the method to tune the optimal cut-off
dual_clf.fit(X, y, metric='youden_index')
```

The `DualModel` object has the following attributes:

* **`alpha_`**: The alpha value of the model. This value is only available if the model is a Lasso or Ridge regression model. (OLS do not have alpha)
* **`coef_`**: The coefficients of the model.
* **`intercept_`**: The intercept of the model.
* **`feature_names_in_`**: The names of the features that were used to train the model.
* **`optimal_cut_off`**: The optimal cut-off value that was determined by the specified metric.
* **`y_label_true_`**: The true binary labels that were generated using the optimal cut-off value.


```python
print(dual_clf.coef_)
[ 4.36693293e-01  9.43577803e-03 -1.07322041e-01  6.45065694e-01
 -3.97638942e-06 -3.78654265e-03 -4.21314378e-01 -4.34513755e-01]
print(dual_clf.intercept_)
-36.94192020718441
print(dual_clf.feature_names_in_)
['MedInc' 'HouseAge' 'AveRooms' 'AveBedrms' 'Population' 'AveOccup'
 'Latitude' 'Longitude']
print(dual_clf.optimal_cut_off)
2.5679
print(dual_clf.y_label_true_)
0        0
1        0
2        0
3        0
4        0
        ..
20635    1
20636    1
20637    1
20638    1
20639    1
Name: MedHouseVal, Length: 20640, dtype: int64
```

```python
# Accessing the true binary labels generated based on the default cut-off
y_label_true = dual_clf.y_label_true_
# Retrieving the optimal cut-off value tuned based on the Youden Index
optimal_cut_off = dual_clf.optimal_cut_off
```
model predict on new data
```python
# Predicting grades and binary classification (at-risk or not) based on the optimal cut-off
y_pred, y_label_pred = dual_clf.predict(X)
```

**y_pred (regression result)**
```python
array([4.13164983, 3.97660644, 3.67657094, ..., 0.17125141, 0.31910524,
       0.51580363])
```
**y_label_pred (binary classification result)**
```python
array([0, 0, 0, ..., 1, 1, 1])
```

## Exmaples of Model Performances Plot
```python
# Visualizations
# Plotting the actual vs. predicted values to assess regression performance
scatter_plot_fig = plot_scatter(y_pred, y)
```
![](https://github.com/098765d/dualPredictor/blob/17cea04496fef61cfa8985852bd5de0d104ead8a/figs/scatter_plot.png)
```python
# Plotting the confusion matrix to evaluate binary classification performance
cm_plot = plot_cm(y_label_true, y_label_pred)
```
![](https://github.com/098765d/dualPredictor/blob/17cea04496fef61cfa8985852bd5de0d104ead8a/figs/cm_plot.png)
```python
# Plotting the non-zero coefficients of the regression model to interpret feature importance
feature_plot = plot_feature_coefficients(coef=dual_clf.coef_, feature_names=dual_clf.feature_names_in_)
```
![](https://github.com/098765d/dualPredictor/blob/17cea04496fef61cfa8985852bd5de0d104ead8a/figs/feature_coefficients.png)

## Another Demo use UCI's Student Performance Dataset
https://www.kaggle.com/code/ddatad/dual-predictor-demo

