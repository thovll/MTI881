import numpy as np

from pytorch_toolbelt.utils import plot_confusion_matrix


def test_plot_confusion_matrix():
    cm = np.random.randint(0, 7, (7, 7))

    plot_confusion_matrix(
        cm, class_names=["A", "B", "C", "D", "E", "F", "G"], normalize=True, fname="test_plot_confusion_matrix.png"
    )
