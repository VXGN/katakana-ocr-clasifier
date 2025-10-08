import math
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_results(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(2, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_svc_confusion_grid(results):
    cols = 2
    rows = math.ceil((len(results) + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.ravel()
    for ax, (title, cm) in zip(axes, results):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    for ax in axes[len(results):]:
        ax.axis('off')
    fig.tight_layout()
    plt.show()