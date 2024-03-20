# dualPredictor

by D

`dualPredictor` is a Python package that can provide simultaneous **regression** and **binary classification** results for tabular datasets. 


- **Simultaneous Predictions**: A model that perform **regression** and binary **classification** tasks simultaneously
- **Regressor Selection (choose one)**: Choose from **Lasso**, **Ridge**, or **LinearRegression(OLS)** as the base regression model.
- **Dynamic Cutoff Tuning metrics (choose one)**: Automatically tunes the cutoff value to maximize the **Youden index**, **F1**, or **F2** score. Users can choose a metrics type.

<img src='https://github.com/098765d/dualPredictor/blob/890d66657e969334fb46d5b97cafbeb2d04fdff6/figs/regressor_metric.png' width="300" height="200">
  

**1. Youden Index (J)**

![](https://miro.medium.com/v2/resize:fit:842/1*LVilqC3cy4AgyC1wD4RH-A.png)

https://miro.medium.com/v2/resize:fit:842/1*LVilqC3cy4AgyC1wD4RH-A.png

 $$J= Recall + Specificity - 1$$
J is a measure of the overall performance of a binary classifier. It is calculated as the sum of the recall and specificity minus 1. A high J statistic indicates that the classifier performs well on positive and negative cases.

* **Recall** measures a classifier's ability to identify positive cases correctly. A high recall means that the classifier is avoiding miss detects.
* **Specificity** measures the ability of a classifier to identify negative cases correctly. A high specificity means that the classifier is avoiding false alarms.

**2. F scores (Option F1, F2 in Package)**

$$F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$$

F1 score is another measure of the overall performance of a binary classifier. It is calculated as the harmonic mean of the precision and recall. A high F1 score indicates that the classifier is performing well on both positive and negative cases.

$$F_\beta = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\beta^2 \cdot precision + recall}$$
F-score with factor beta is a generalization of the F1 score that allows for different weights to be given to precision and recall. A beta value less than 1 indicates that the F-score is prone to precision, while a beta value greater than 1 indicates that the F-score is prone to recall.

**In educational settings**

In educational settings, avoiding miss detects (i.e., failing to identify at-risk students) is important. However, it is also important to avoid false alarms (i.e., identifying students as at-risk when they are not). Therefore, using a measure prone to recall is often desirable, such as the F1 score with beta > 1.
Youden's J statistic and the F1 score are both measures that balance the avoidance of miss detects and the avoidance of false alarms. However, Youden's J statistic is less sensitive to false alarms (Specificity is less sensitive to false alarms compared to Precision) than the F1 score.


## Installation

Install `dualPredictor` directly from PyPI using pip:

```bash
pip install dualPredictor
```
or Directly install from the Github Repo:

```bash
pip install git+https://github.com/098765d/dualPredictor.git
```

Dependencies
dualPredictor requires:
- numpy
- scikit-learn
- matplotlib
- seaborn

## DualModel

The `DualModel` class is a custom regressor that combines a base regression model (lasso, ridge, or OLS) with a dual classification approach. It allows for tuning an optimal cut-off value to classify samples into two classes based on the predicted regression values.

### Parameters

- `model_type` (str, default='lasso'): The base regression model to use. Supported options are 'lasso', 'ridge', and 'ols' (Ordinary Least Squares).

- `metric` (str, default='youden_index'): The metric used to tune the optimal cut-off value. Supported options are 'f1_score', 'f2_score', and 'youden_index'. 

- `default_cut_off` (float, default=0.5): The default cut-off value used to create binary labels. Samples with regression values below the cut-off are labeled as 0, and samples above or equal to the cut-off are labeled as 1.

### Methods

- `fit(X, y)`: Fit the DualModel to the training data.

    - Parameters:
        - `X` (array-like of shape (n_samples, n_features)): The input training data.
        - `y` (array-like of shape (n_samples,)): The target values.

    - Returns:
        - `self`: Fitted DualModel instance.

- `predict(X)`: Predict the input data's regression values and binary classifications.

    - Parameters:
        - `X` (array-like of shape (n_samples, n_features)): The input data for prediction.

    - Returns:
        - `grade_predictions` (array-like of shape (n_samples,)): The predicted regression values.
        - `class_predictions` (array-like of shape (n_samples,)): The predicted binary classifications based on the optimal cut-off.

### Attributes

- `alpha_`: The alpha value of the model. This value is only available if the model is a Lasso or Ridge regression model. (OLS do not have alpha)
- `coef_`: The coefficients of the model.
- `intercept_`: The intercept of the model.
- `feature_names_in_`**: The names of the features used to train the model.
- `optimal_cut_off`: The optimal cut-off value determined by the specified metric.
- `y_label_true_`: The true binary labels are generated using the default cut-off value.
  
### Example

```python
# Import the DualModel class
from dual_model import DualModel

# Initializing and fitting the DualModel
# 'ols' for Ordinary Least Squares, a default cut-off value is provided
# The metric parameter specifies the method to tune the optimal cut-off
dual_clf = DualModel(model_type='ols', metric='youden_index', default_cut_off=1)
dual_clf.fit(X, y)

# Accessing the true binary labels generated based on the default cut-off
y_label_true = dual_clf.y_label_true_

# Retrieving the optimal cut-off value tuned based on the Youden Index
optimal_cut_off = dual_clf.optimal_cut_off

# Predicting grades (y_pred) and binary classification (at-risk or not) based on the optimal cut-off (y_label_pred)
y_pred, y_label_pred = dual_clf.predict(X)
```


## Examples of Model Performances Plot
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

## Example 1: UCI student Performance Dataset
[Link to UCI student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

https://www.kaggle.com/code/ddatad/dual-predictor-demo?scriptVersionId=167940301

Train/Test Data Information:
- Number of data points in training set: 454 (70.0%)
- Number of data points in test set: 195 (30.0%)

If default cut_off = 10 (label = 1 will be fail students), select lasso + youden

**Train set performance**
- Number of data points: 454
- Number of total positive (label=1): 74
- Number of miss detects: 2
- Number of false alarms: 61
- Classification rate: 0.861
- R2 = 0.83, MSE = 1.68

**Test set performance**
- Number of data points: 195
- Number of total positive (label=1): 26
- Number of miss detects: 1
- Number of false alarms: 22
- Classification rate: 0.882
- R2 = 0.88, MSE = 1.3

## Example 2: a Local University Students Program GPA Prediction

Since Test Set Students does not have y-label, therefore only able to show the train set performance.
default cut-off = 2.5 , lasso + youden_index

**Train set performance**
- Number of data points: 154
- Number of true positive (label=1): 5
- Number of miss detects: 0
- Number of false alarms: 6
- Classification rate: 0.961
- R2 = 0.96
- Optimal_cut_off=2.70

**Test set performance**
- Number of data points: 71
- Number of label = 1 prediction: 3

## Example 3: Object Oriented Programming Class Student Grades from Mugla Sitki Kocman University | '19 OOP Class Student Grades

https://www.kaggle.com/datasets/onurduman/grades/data

Train/Test Data Information:
- Number of data points in training set: 33 (60.0%)
- Number of data points in test set: 22 (40.0%)

If default cut_off = 50 (label = 1 will be fail students), select ols + youden

**Train set performance**
- Number of data points: 33
- Number of true positive (label=1): 21
- Number of miss detects: 1
- Number of false alarms: 1
- Classification rate: 0.939
- R2 = 0.94
- Optimal_cut_off= 50

**Test set performance**
- Number of data points: 22
- Number of true positive (label=1): 13
- Number of miss detects: 2
- Number of false alarms: 0
- Classification rate: 0.909
- R2 = 0.65

### References:

- [Youden's J statistic - Wikipedia](https://en.wikipedia.org/wiki/Youden%27s_J_statistic)
- [F-score - Wikipedia](https://en.wikipedia.org/wiki/F-score)
- [Scikit-learn - Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
