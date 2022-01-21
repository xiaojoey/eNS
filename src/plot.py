import os
import csv
import io
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy import interpolate

def calculate_pr_recall(predictions, actual, threshold=.1):
    """
    Calculates the precision and information for each threshold
    Args:
        predictions - (np.ndarray) Predicted probabilities for being positive
        actual - (np.ndarray) Actual labels
        threshold - (float) Steps between threshold values
    Returns:
        results - (np.ndarray) array containing precision, recall and f1 scores for each threshold
    """
    threshold_list = np.round(np.linspace(0, 1, int(1 + (1/threshold))), 2)
    results = []
    for limit in threshold_list:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(predictions)):
            if actual[i] == 1 and predictions[i] >= limit:
                TP += 1
            if actual[i] != 1 and predictions[i] >= limit:
                FP += 1
            if actual[i] == 0 and predictions[i] < limit:
                TN += 1
            if actual[i] != 0 and predictions[i] < limit:
                FN += 1

        recall = TP / (TP + FN)

        if recall == 0:
            precision = 1
        elif TP == FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if precision == recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        results.append([limit, precision, recall, f1, TP, FP, TN, FN])
    return np.array(results)

def plot_pr_recall(data_path, file_name):
    """
    Generate a plot for the precision recall curve
    Args:
        data_path - (string) Path to the prediction and actual labels
        file_name - (string) Name of the file to be saved
    """
    precision, recall, threshold, auc = get_pr(data_path)
    title = "Precision-Recall curve: " + "AUC = " + str(np.round(auc, 4))
    multicolored_lines(recall, precision, threshold, title, file_name)

def save_chart(data, file_name):
    """
    Saves the precision recall chart
    Args:
        data - (np.ndarray) Analysis results from calculate_pr_recall
        file_name - (string) Name of the file to be saved
    """
    with open(file_name, 'w', newline='\n', encoding="ISO-8859-1") as csvfile:
        record_writer = csv.writer(csvfile, delimiter=',')
        attribute_names = ['threshold', 'precision', 'recall', 'f1', 'TP', 'FP', 'TN', 'FN']
        record_writer.writerow(attribute_names)
        for i  in data:
            record_writer.writerow(i)

def multicolored_lines(x, y, z, title, file_name):
    """
    Plots line colored based on z
    Args:
        x - (np.ndarray) Recall values
        y - (np.ndarray) Precision values
        z - (np.ndarray) Threshold values
        title - (string) Title for the plot
        file_name - (string) Path to save the output figure
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, axs = plt.subplots(figsize=(10,8))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_capstyle('round')
    # Set the values used for colormapping
    lc.set_array(z)
    lc.set_linewidth(4)
    line = axs.add_collection(lc)
    axcb = fig.colorbar(line, ax=axs)

    axs.set_xlim(-0.05, 1.05)
    axs.set_ylim(-0.05, 1.05)
    plt.title(title, size=16, pad=20)
    plt.xlabel('Recall', size=13, labelpad=15)
    plt.ylabel('Precision', size=13, labelpad=15)
    axcb.set_label('Classification Threshold', size=13, labelpad=20)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close(fig)

def multicolored_lines2(x, y, z, title, file_name, upper, lower):
    """
    Plots line colored based on z
    Args:
        x - (np.ndarray) Recall values
        y - (np.ndarray) Precision values
        z - (np.ndarray) Threshold values
        title - (string) Title for the plot
        file_name - (string) Path to save the output figure
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, axs = plt.subplots(figsize=(10,8))
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_capstyle('round')
    # Set the values used for colormapping
    lc.set_array(z)
    lc.set_linewidth(4)
    line = axs.add_collection(lc)
    # axcb = fig.colorbar(line, ax=axs)
    axs.fill_between(x, lower, upper, color='grey', alpha=.2,
        label=r'$\pm$ 1 std. dev.')

    axs.set_xlim(-0.05, 1.05)
    axs.set_ylim(-0.05, 1.05)
    plt.title(title, size=16, pad=20)
    plt.xlabel('Recall', size=13, labelpad=15)
    plt.ylabel('Precision', size=13, labelpad=15)
    # axcb.set_label('Classification Threshold', size=13, labelpad=20)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close(fig)

def get_pr(path):
    """
    Calculates precision and recall for validation output
    Args:
        path - (string) Paths for the validation data
    """
    data = pd.read_csv(path)
    precision, recall, threshold = precision_recall_curve(data['actual'], data['prediction'])
    prauc = auc(recall, precision)
    return precision, recall, threshold, prauc


def combine_val(paths):
    """
    Combines all the prediction and validation data in list of files and calculates precision and recall
    Args:
        paths - (list) List of paths for the validation data
    Returns:
        precision - (np.ndarray)
        recall - (np.ndarray)
        threshold - (np.ndarray)
        prauc - (int)
    """
    precision_list = []
    pr_auc_list = []
    new_recall = np.linspace(0, 1, 100)
    threshold = np.linspace(1, 1, 100)
    for path in paths:
        data = pd.read_csv(path)
        actual = data['actual']
        prediction = data['prediction']
        precision, recall, threshold = precision_recall_curve(actual, prediction)
        prauc = auc(recall, precision)
        pr_auc_list.append(prauc)

        f = interpolate.interp1d(recall, precision)
        new_precision = f(new_recall)
        precision_list.append(new_precision)

    mean_pr_auc = np.array(pr_auc_list).mean()
    print(mean_pr_auc)
    mean_precision = np.array(precision_list).mean(axis=0)
    new_pr_auc = auc(new_recall, mean_precision)
    print(new_pr_auc)

    std_precision = np.std(np.array(precision_list))
    precision_upper = np.minimum(mean_precision + std_precision, 1)
    precision_lower = np.maximum(mean_precision - std_precision, 0)

    return mean_precision, new_recall, threshold, mean_pr_auc, precision_upper, precision_lower


def plot_combined(folder, output):
    """
    Plot the combined precision and recall for a folder of validation output
    Args:
        folder - (string) Name of the folder containing the validation data
        output - (string) Path to save the output figure
    """

    validation_folder = folder
    validation_paths = [join(validation_folder, f) for f in listdir(validation_folder) if (isfile(join(validation_folder, f)) and ('val.csv' in f))]
    precision, recall, threshold, auc, upper, lower = combine_val(validation_paths)
    title = "Precision-Recall curve: " + "AUC = " + str(np.round(auc, 4))
    multicolored_lines2(recall, precision, threshold, title, output, upper, lower)
