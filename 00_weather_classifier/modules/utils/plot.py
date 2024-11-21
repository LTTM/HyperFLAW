import numpy as np
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

from ..dataset.condition_map import MAPPING_WEATHER


# Define the export
__all__ = ["plot_loss", "plot_accuracies", "plot_confusion_matrix"]


def plot_losses(losses: [np.ndarray, np.ndarray], save_path: Path = None) -> None:
    # Set the theme
    sns.set_theme()

    if isinstance(losses, np.ndarray):
        losses = [losses]

    assert len(losses) <= 2, "The number of losses must be less or equal to 2"

    labels = ["Train", "Test"] if isinstance(losses, list) else None

    plt.figure()
    for i in range(len(losses)):
        plt.plot(losses[i], label=labels[i])
    plt.xlabel("Epoch")
    plt.xticks(
        ticks=np.arange(losses[0].shape[0]), labels=np.arange(1, losses[0].shape[0] + 1)
    )
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_accuracies(
    accuracies: [np.ndarray, np.ndarray], save_path: Path = None
) -> None:
    # Set the theme
    sns.set_theme()

    if isinstance(accuracies, np.ndarray):
        accuracies = [accuracies]

    assert len(accuracies) <= 2, "The number of accuracies must be less or equal to 2"

    linestyles = ["-", "--"] if isinstance(accuracies, list) else None

    plt.figure()
    for k in range(len(accuracies)):
        for i in range(accuracies[k].shape[0]):
            plt.plot(
                accuracies[k][i, :],
                label=list(MAPPING_WEATHER.keys())[i],
                linestyle=linestyles[k],
            )
    plt.xlabel("Epoch")
    plt.xticks(
        ticks=np.arange(accuracies[0].shape[1]),
        labels=np.arange(1, accuracies[0].shape[1] + 1),
    )
    plt.ylabel("Accuracy")
    plt.title("Accuracy per class (-: train, --: test)") if isinstance(
        accuracies, list
    ) else plt.title("Accuracy per class")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_confusion_matrix(confusion_matrix: np.ndarray, save_path: Path = None) -> None:
    sns.set_theme()

    plt.figure()
    sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.xticks(
        ticks=np.arange(confusion_matrix.shape[1]) + 0.5,
        labels=list(MAPPING_WEATHER.keys()),
    )
    plt.ylabel("True")
    plt.yticks(
        ticks=np.arange(confusion_matrix.shape[1]) + 0.5,
        labels=list(MAPPING_WEATHER.keys()),
    )
    plt.title("Confusion matrix")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
