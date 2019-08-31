import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def calculate_ratios(tp, fp, fn, tn):
    true_pos_rate = tp / (tp + fn)  # TPR / Recall / Sensitivity
    false_pos_rate = fp / (fp + tn)  # FPR = 1 - Specificity
    return true_pos_rate, false_pos_rate


def get_roc_curve_tpr_fpr(tpr, fpr):
    plt.title('ROC curve')
    x_axis = np.array(fpr)
    y_axis = np.array(tpr)
    plt.plot(x_axis, y_axis, color='g')  # x, y
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.show()


def get_roc_curve_frr_far(frr, far):
    plt.title('ROC curve')
    x_axis = np.array(far)  # False Positive (Accept / Match) Rate
    y_axis = np.array(frr)  # False Reject Rate = 1 - TPR
    plt.plot(x_axis, y_axis)  # x, y
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('False Reject Rate (FRR)')
    plt.xlabel('False Accept Rate (FAR)')
    plt.show()


def get_cmc_curve(x_values, y_values):
    plt.title('CMC curve')
    x_axis = np.array(x_values)
    y_axis = np.array(y_values)
    plt.plot(x_axis, y_axis)  # x, y
    plt.ylim([0, 1])
    plt.xlabel('Rank')
    plt.ylabel('Probability of Recognition')
    plt.show()


def plot_genuine_impostor_distribution(genuine_scores, impostor_scores):
    fig, ax1 = plt.subplots()
    bins = 100
    ax1.set_xlabel('Matching scores (distance)')
    ax1.set_title('Genuine / Impostor scores plot')
    ax1.set_ylabel('Frequency (Genuines)')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Frequency (Impostors)')

    ax1.hist(genuine_scores, bins=bins, color='g', label='Genuine scores %d' % len(genuine_scores))
    ax2.hist(impostor_scores, bins=bins, alpha=0.5, color='b', label='Impostor scores %d' % len(impostor_scores))
    fig.legend(prop=FontProperties(size=10), bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.show()


def plot_accuracy_f1_scores(accuracy_scores, f1_scores, thresholds):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Distance Thresholds')
    ax1.set_title('Accuracy / F1 scores plot')
    ax1.set_ylabel('Accuracy')
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 score')

    ax1.plot(thresholds, accuracy_scores, alpha=0.5, color='g', label='Accuracy scores')
    ax2.plot(thresholds, f1_scores, alpha=0.5, color='b', label='F1 scores')
    fig.legend(prop=FontProperties(size=10), bbox_to_anchor=(0, 0, 1, 1), loc=2, bbox_transform=ax1.transAxes)
    plt.show()


def plot_faces(faces):
    plt.subplot(1, 3, 1)
    plt.title(faces.target[7])
    plt.imshow(faces.images[7], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title(faces.target[17])
    plt.imshow(faces.images[17], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title(faces.target[27])
    plt.imshow(faces.images[27], cmap='gray')
    plt.show()

