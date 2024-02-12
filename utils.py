import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score

np.random.seed(42)

PROJECT_ROOT_DIR = "."

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


def image_path(file_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", file_id)


def resource_path(file_id):
    return os.path.join(PROJECT_ROOT_DIR, "resources", file_id)


def save_fig(file_id, tight_layout=True):
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(file_id) + ".png", format='png', dpi=300)


def print_eval(y_test, y_pred):
    precision = 100 * precision_score(y_test, y_pred)
    recall = 100 * recall_score(y_test, y_pred)
    f1 = 100 * f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print("Precision: {:.2f}%".format(precision))
    print("Recall: {:.2f}%".format(recall))
    print("F1: {:.2f}%".format(f1))
    print("AUC: {:.2f}".format(auc))


def print_feature_importances(headers, values):
    idx = np.argsort(values)[::-1]
    sorted_values = np.array(values)[idx]
    sorted_headers = np.array(headers)[idx]
    for i, value in enumerate(sorted_values):
        if value > 0:
            print(sorted_headers[i], ": ", value)
