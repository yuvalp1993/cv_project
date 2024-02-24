"""Plot ROC and DET curves."""
import os
import argparse

import torch
import scipy.stats as sp
import matplotlib.pyplot as plt

from sklearn import metrics
from torch.utils.data import DataLoader

from common import FIGURES_DIR
from utils import load_dataset, load_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Arguments
def parse_args():
    """Parse script arguments.

    Returns:
        Namespace with model name, checkpoint path and dataset name.
    """
    parser = argparse.ArgumentParser(description='Analyze network performance.')
    parser.add_argument('--model', '-m',
                        default='XceptionBased', type=str,
                        help='Model name: SimpleNet or XceptionBased.')
    parser.add_argument('--checkpoint_path', '-cpp',
                        default='checkpoints/XceptionBased.pt', type=str,
                        help='Path to model checkpoint.')
    parser.add_argument('--dataset', '-d',
                        default='fakes_dataset', type=str,
                        help='Dataset: fakes_dataset or synthetic_dataset.')

    return parser.parse_args()


def get_soft_scores_and_true_labels(dataset, model):
    model.to(device)  # Ensure model is on the correct device
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_first_soft_scores = []
    all_second_soft_scores = []
    gt_labels = []

    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # Forward pass

            # Softmax to convert outputs to probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # Split probabilities into "real" and "fake/synthetic" scores
            all_first_soft_scores.extend(probabilities[:, 0].cpu().numpy())
            all_second_soft_scores.extend(probabilities[:, 1].cpu().numpy())
            gt_labels.extend(labels.cpu().numpy())

    return all_first_soft_scores, all_second_soft_scores, gt_labels


def plot_roc_curve(roc_curve_figure,
                   all_first_soft_scores,
                   all_second_soft_scores,
                   gt_labels):
    """Plot a ROC curve for the two scores on the given figure.

    Args:
        roc_curve_figure: the figure to plot on.
        all_first_soft_scores: iterable of soft scores.
        all_second_soft_scores: iterable of soft scores.
        gt_labels: ground truth labels.

    Returns:
        roc_curve_first_score_figure: the figure with plots on it.
    """
    fpr, tpr, _ = metrics.roc_curve(gt_labels, all_first_soft_scores)
    plt.plot(fpr, tpr)
    fpr, tpr, _ = metrics.roc_curve(gt_labels, all_second_soft_scores)
    plt.plot(fpr, tpr)
    plt.grid(True)
    plt.xlabel('False Positive Rate (Positive label: 1)')
    plt.ylabel('True Positive Rate (Positive label: 1)')
    plt.title(f'ROC curves AuC Score for the first score: '
              f'{metrics.roc_auc_score(gt_labels, all_first_soft_scores):.3f}, '
              f'AuC second score: '
              f'{metrics.roc_auc_score(gt_labels, all_second_soft_scores):.3f}')
    plt.legend(['first score', 'second score'])
    roc_curve_figure.set_size_inches((8, 8))
    return roc_curve_figure


def plot_det_curve(det_curve_figure,
                   all_first_soft_scores,
                   all_second_soft_scores,
                   gt_labels):
    """Plot a DET curve for the two scores on the given figure.

    Args:
        det_curve_figure: the figure to plot on.
        all_first_soft_scores: iterable of soft scores.
        all_second_soft_scores: iterable of soft scores.
        gt_labels: ground truth labels.

    Returns:
        roc_curve_first_score_figure: the figure with plots on it.
    """
    fpr, fnr, _ = metrics.det_curve(gt_labels, all_first_soft_scores)
    plt.plot(sp.norm.ppf(fpr), sp.norm.ppf(fnr))
    fpr, fnr, _ = metrics.det_curve(gt_labels, all_second_soft_scores)
    plt.plot(sp.norm.ppf(fpr), sp.norm.ppf(fnr))

    plt.grid(True)
    plt.xlabel('False Positive Rate (Positive label: 1)')
    plt.ylabel('False Negative Rate (Positive label: 1)')
    plt.title('DET curve for the first score')
    axes = det_curve_figure.gca()
    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_labels = [
        '{:.0%}'.format(s) if (100 * s).is_integer() else '{:.1%}'.format(s)
        for s in ticks
    ]
    tick_locations = sp.norm.ppf(ticks)
    axes.set_xticks(tick_locations)
    axes.set_xticklabels(tick_labels)
    axes.set_yticks(tick_locations)
    axes.set_yticklabels(tick_labels)
    axes.set_ylim(-3, 3)
    plt.legend(['first score', 'second score'])
    det_curve_figure.set_size_inches((8, 8))
    return det_curve_figure


def main():
    """Parse script arguments, log all the model's soft scores on the dataset
    images and the true labels. Use the soft scores and true labels to
    generate ROC and DET graphs."""
    args = parse_args()

    # load model
    model_name = args.model
    model = load_model(model_name)
    model.load_state_dict(torch.load(args.checkpoint_path)['model'])
    model.eval()

    # load dataset
    test_dataset = load_dataset(dataset_name=args.dataset, dataset_part='test')
    all_first_soft_scores, all_second_soft_scores, gt_labels = \
        get_soft_scores_and_true_labels(test_dataset, model)

    # plot the roc curves
    roc_curve_figure = plt.figure()
    roc_curve_figure = plot_roc_curve(roc_curve_figure,
                                      all_first_soft_scores,
                                      all_second_soft_scores,
                                      gt_labels)
    roc_curve_figure.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_roc_curve.png'))

    # plot the det curve for the scores of the first output of the network
    det_curve_figure = plt.figure()
    det_curve_figure = plot_det_curve(det_curve_figure,
                                      all_first_soft_scores,
                                      all_second_soft_scores,
                                      gt_labels)
    det_curve_figure.savefig(
        os.path.join(FIGURES_DIR,
                     f'{args.dataset}_{args.model}_det_curve.png'))


if __name__ == '__main__':
    main()
