import itertools

import matplotlib.pyplot as plt
import numpy as np


def lab2color(lab):
    h, w = lab.shape
    col = np.zeros((h, w, 3), dtype=np.uint8)

    col[lab == 1, 0] = 255  # red = urb
    col[lab == 0, 1] = 255  # green = non-urb
    col[lab == 2, 2] = 255  # blue = cloud

    return col


# Source: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def plot_confusion_matrix(cm, target_names, ax, title='Confusion matrix',
                          cmap=None, normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(target_names, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(target_names, rotation=45)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="black" if cm[i, j] > thresh else "white")
        else:
            ax.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="black" if cm[i, j] > thresh else "white")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
