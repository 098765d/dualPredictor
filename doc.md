# dualPredictor: An Open-Source Tool for Simultaneously Grade Prediction and At-Risk Student Classification

by D

PyPI Link: https://pypi.org/project/dualPredictor/

Github Repo: https://github.com/098765d/dualPredictor/

## 1. Introduction

The **dualPredictor** package combines regression analysis with binary classification to forecast student academic outcomes. Meanwhile, dualPredictor offers model explanations at both global and local levels.

The accompanying figure (Fig 1) illustrates how dualPredictor generates dual output—regression and classification—by combining a regressor and a metric.

![](https://github.com/098765d/dualPredictor/raw/eb30145140a93d355342340d2a7ab256ccbbbf6e/figs/how_dual_works.png)
**Fig 1**: How does dualPredictor provide dual prediction output?

### 1.1 Dual Prediction Output Mechanism
- **Step 1: Grade Prediction Using the Trained Regressor** (Fig 1, Step 1)
  fit the linear model f(x) using the training data, and grade prediction can be generated from the fitted model
  
  ```math
      y\_pred = f(x) = \sum_{j=1}^{M} w_j x_j + b 
  ```
  
- **Step 2: Determining the Optimal Cut-off** (Fig 1, Step 2)
  
  The goal is to find the **cut-off (c)** that maximizes the binary classification accuracy.
  here we offer 3 options of metrics that measure the classification accuracy: Youden index, f1_score, and f2_score.
  Firstly, the user specifies the metric type used for the model (e.g., Youden index) and denotes the **metric function as g(y_true_label, y_pred_label)**, where:
  ```math
  \text{optimal\_cut\_off} = \arg\max_c g(y_{\text{true\_label}}, y_{\text{pred\_label}}(c))
  ```
  This formula searches for the cut-off value that produces the highest value of the metric function g, where:
  
  * **c**: The tunned cut-off that determines the y_pred_label
  * y_true_label: True label of the data point based on the default cut-off (e.g., 1 for at-risk, 0 for normal)
  * y_pred_label: Predicted label of the data point based on the tunned cut-off value

    
- **Step 3: Binary Label Prediction**: (Fig 1, Step 3)
  
  - y_pred_label = 1 (at-risk): if y_pred < optimal_cut_off
  - y_pred_label = 0 (normal): if y_pred >= optimal_cut_off



### 1.2 Model Explanations
- **Global level Model Explanations**: The model's feature coefficients plot (See Fig 2c)
- **Local level Model Explanations**: The model's feature contribution for a specific data point (See Fig 2d)
    - How to get the feature contribution for a given data point?

      Given a linear model with a total number of M features, the model can be represented as:
      ```math
      f(x) = \sum_{j=1}^{M} w_j x_j + b 
      ```
      
      The j-th feature contribution for the i-th data point can be approximated from the formula:
      
      ```math
      \phi_i(f, x) = w_j (x_j - E[x_j])
      ```

      The formula can be seen as an approximation of the Shapley value for linear models from page 6 of the paper:
      [Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.](https://dl.acm.org/doi/10.5555/3295222.3295230)
       

## 2. The Model Methods and Attributes
The dualPredictor package aims to simplify complex models for users of all coding levels. It adheres to the syntax of the scikit-learn library. The core part of the package is the model object called DualModel, which can be imported from the dualPredictor library.

**Table 0:** Model Parameters

| Parameter | Description | Default Value |
|---|---|---|
| `model_type` | Type of regression model to use. Options include:  - `'lasso'` (Lasso regression)  - `'ridge'` (Ridge regression)  - `'ols'` (Ordinary Least Squares regression) | `None` |
| `metric` | Metric used for optimizing the cut-off value. Options include:  - `'f1_score'` (F1 score)  - `'f2_score'` (F2 score)  - `'youden_index'` (Youden's Index) | `None` |
| `default_cut_off` | Initial cut-off value used for binary classification. | `None` |

```python
from dualPredictor import DualModel
# intialize the model, specify the parameters
model = DualModel(model_type='lasso', metric='f1_score', default_cut_off=2.5)
```

The model object's **methods** and **attributes** (See Table 1 and 2) follow the sci-kit-learn style.

**Table 1**: Model methods (scikit-learn linear model object style)
| Model Methods | Description |
|--------------|-------------|
| `fit(X, y)`  | - **X**: The input training data, pandas data frame. <br> - **y**: The target values (predicted grade). <br> - **Returns**: Fitted DualModel instance |
| `predict(X)` | - **X**: The input training data, pandas' data frame. |

```python
model.fit(X_train, y_train)
model.predict(X_train)
```
**Table 2**: Model attributes (scikit-learn linear model attributes style)
| Model Attributes   | Description                                                   |
|--------------------|---------------------------------------------------------------|
| `alpha_`           | The value of penalization in Lasso and ridge (for OLS, alpha = 0) |
| `coef_`            | The coefficients of the model                                  |
| `Intercept_`       | The intercept value of the model                               |
| `feature_names_in_`| Names of features during model training                        |
| `optimal_cut_off`  | The optimal cut-off value that maximizes the metric            |


## 3. How to Install?

### 3.1 Dependencies Installation

dualPredictor requires the following libraries to be installed:

- NumPy: A fundamental package for scientific computing with Python.
- scikit-learn: A simple and efficient tools for predictive data analysis.
- Matplotlib: A comprehensive library for creating static, animated, and interactive visualizations in Python.
- Seaborn: A Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
You can install all the dependencies at once using the following command:

```bash
pip install numpy scikit-learn matplotlib seaborn
```


### 3.2 Package Installation 

You can install the dualPredictor package via PyPI or GitHub (Recommended). Choose one of the following methods:

```bash
pip install dualPredictor
```

```bash
pip install git+https://github.com/098765d/dualPredictor.git
```

## 4. User Guide with Example Code
After installation, start with:

**Step 1. Import the Package:** Import the dualPredictor package into your Python environment.
```python
from dualPredictor import DualModel, model_plot
```
**Step 2. Model Initialization:** 
Create a DualModel instance by specifying the regressor type ('lasso', 'ridge', or 'ols'), the metric for cutoff tuning ('f1_score', 'f2_score', or 'youden_index'), and a default cutoff value.
```python
# model_type options: 'lasso', 'ridge', or 'ols'
# metric options: 'f1_score', 'f2_score', or 'youden_index'
model = DualModel(model_type='lasso', metric='youden_index', default_cut_off=2.5)
```
**Step 3. Model Fitting:** Fit the model to your dataset using the fit method.
```python
model.fit(X_train, y_train)
```
- X: The input training data (type: pandas DataFrame).
- y: The target values (type: pandas data series).

**Step 4. Predictions:** Use the model's predict method to generate grade predictions and at-risk classifications.
  ```python
# example for demo only, model prediction dual output
y_train_pred,y_train_label_pred=model.predict(X_train)

# example of 1st model output = predicted scores (regression result)
y_train_pred
array([3.11893389, 3.06013236, 3.05418893, 3.09776197, 3.14898782,
       2.37679417, 2.99367804, 2.77202421, 2.9603209 , 3.01052573,
       2.99974477, 3.11286716, 3.14708887, 2.78737598, 2.88134869,
       3.07517748, 3.17370297, 3.26615469, 3.2328493 , 2.98423656,
       3.02108518, 2.87746064, 3.03491596, 2.89875586, 3.11079315,
       3.23177653, 3.34291929, 2.57402463, 3.27019917, 3.20073168,
       2.94514418, 3.25307175, 3.19145494, 3.15909904, 3.01481681,
       3.07551728, 2.70973767, 3.07226583, 3.04692613, 2.8883649 ,
       2.63833457, 3.03978663, 3.20974038, 3.13091091, 3.42223703,
       3.07012029, 3.01981077, 3.22368756, 2.69376153, 2.93594929,
       2.91493381, 3.22273808, 2.59310411, 3.00767959, 3.21869359,
       2.86065334, 3.16865551, 3.11258742, 2.87948289, 2.64564212,
       2.88646595, 3.48716006, 3.14482003, 3.15513751, 3.05299286,
       3.20858237, 2.63172024, 2.42824269, 2.88352738, 3.0479989 ,
       2.82405611, 3.16516577, 2.94324523, 3.4453079 , 2.48497569,
       3.00081754, 3.04180887, 3.32979373, 3.12686642, 2.90359338,
       2.95509896, 2.96429385, 3.44471154, 3.20251564, 3.08765075,
       2.5607482 , 3.23986551, 3.19644891, 3.16032825, 2.68092384,
       3.04907167, 2.8159268 , 3.05030088, 3.178372  ])

# example of 2nd model output = predicted at-risk status (binary label)
y_train_label_pred
array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 0, 0])
```
- y_train_pred: Predicted grades (regression result).
- y_train_label_pred: Predicted at-risk status (binary label).

**Step 5.Visualization:** Visualize the model's performance using the model_plot module (Optional)
```python
# Scatter plot for regression analysis - a
model_plot.plot_scatter(y_pred, y_true)

# Confusion matrix for binary classification - b
model_plot.plot_cm(y_label_true, y_label_pred)

# Model's global explanation: Feature importance plot - c
model_plot.plot_feature_coefficients(coef=model.coef_, feature_names=model.feature_names_in_)

# Model's local explanation: Feature contributions for each data point - d
# 'idx' is the index value used to locate a specific row in the dataframe
plot_local_shap(X=X_test, model=model, idx='E115CCCD')
```

![Fig2](https://github.com/098765d/dualPredictor/raw/75e331cae5017839b4ce6022a27d70d2e33f1605/figs/model_plot.png)
**Fig 2**: Sample plots generated by the model_plot modules

## References

[1] Fluss, R., Faraggi, D., & Reiser, B. (2005). Estimation of the Youden Index and its associated cutoff point. _Biometrical Journal: Journal of Mathematical Methods in Biosciences_, 47(4), 458-472.

[2] Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. _Technometrics_, 12(1), 55-67.

[3] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

[4] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. _The Journal of Machine Learning Research_, 12, 2825-2830.

[5] Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. _Journal of the Royal Statistical Society Series B: Statistical Methodology_, 58(1), 267-288.
