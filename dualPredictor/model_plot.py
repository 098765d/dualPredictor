# dualPredictor/model_plot.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def plot_scatter(y_pred, y_true):

    # Calculate the r2 and mse
    r2 = r2_score(y_true, y_pred)
    r2 = abs(r2)
    mse = mean_squared_error(y_true, y_pred)

    print('R2 = ',r2)
    print('MSE = ',mse)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)

    # Plot the data
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)

    # Set the x and y limits
    ax.set_xlim(min(y_true)*0.8, max(y_true)*1.1)
    ax.set_ylim(min(y_true)*0.8, max(y_true)*1.1)

    # Add the r2 and mse to the figure
    ax.text(0.05, 0.95, f"R2 =  {r2:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.90, f"MSE = {mse:.2f}", transform=ax.transAxes)

    # Add a diagonal line
    ax.plot([min(y_true)*0.8, max(y_true)*1.1], [min(y_true)*0.8, max(y_true)*1.1], color='red', linestyle='--')

    # Set the x and y axis descriptions
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')

    # Add a title with total data points and purpose
    num_data_points = len(y_true)
    title = f"Actual vs Predicted Plot\nTotal Data Points: {num_data_points}"
    ax.set_title(title)

    return fig


def plot_cm(y_label_true, y_label_pred):

    # Calculate the confusion matrix
    cm = confusion_matrix(y_label_true, y_label_pred)
    tn, fp, fn, tp=cm.ravel()

    # Calculate the number of data points, number of miss detects, number of false alarms, and classification rate
    num_data_points = len(y_label_true)
    num_false_alarms,num_miss_detects = fp,fn
    classification_rate = round(np.trace(cm) / num_data_points,3)
    num_tp=tp+fn

    # Print the number of data points, number of miss detects, number of false alarms, and classification rate
    print("Number of data points:", num_data_points)
    print("Number of label=1 points:", num_tp)
    print("Number of miss detects:", num_miss_detects)
    print("Number of false alarms:", num_false_alarms)
    print("Classification rate:", classification_rate)

    # Display the confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=500)
    disp.plot(ax=ax)

    # Add figure title with total number of data points and classification rate
    title = f"Confusion Matrix\nTotal Data Points: {num_data_points}\nClassification Rate: {classification_rate:.2f}"
    ax.set_title(title)

    return fig

def plot_feature_coefficients(coef, feature_names):

    # Round the coefficients to two decimal places
    rounded_coef = np.round(coef, decimals=2)

    # Get non-zero feature coefficients and corresponding names
    nonzero_coef = rounded_coef[rounded_coef != 0]
    nonzero_names = np.array(feature_names)[rounded_coef != 0]
    num_of_features=len(nonzero_names)

    # Sort non-zero coefficients and names in descending order
    sorted_indices = np.argsort(np.abs(nonzero_coef))[::-1]
    sorted_coef = nonzero_coef[sorted_indices]
    sorted_names = nonzero_names[sorted_indices]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 6), dpi=500)

    # Plot horizontal bars of feature coefficients
    ax.barh(sorted_names, sorted_coef)

    # Set limits for x-axis
    ax.set_xlim([np.min(sorted_coef) * 1.1, np.max(sorted_coef) * 1.1])

    # Add coefficient values at the end of each bar
    for i, coef in enumerate(sorted_coef):
        ax.text(coef, i, str(coef), va="center")

    # Add a title to the plot
    ax.set_title(f"Feature Coefficient Plot - {num_of_features} features")

    return fig
