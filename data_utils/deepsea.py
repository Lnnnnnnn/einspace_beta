
from typing import Callable, Optional, Union

import torch
import numpy as np
import pickle
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics


def build_nasbench360_deepsea_dataset(cfg):
    path = cfg["deepsea_root"]
    data = np.load(os.path.join(path, "deepsea_filtered.npz"))
    train_data, train_labels = torch.from_numpy(data["x_train"]).type(torch.FloatTensor), torch.from_numpy(
        data["y_train"]
    ).type(torch.FloatTensor)
    train_data = train_data[:, :, :]
    trainset = TensorDataset(train_data, train_labels)

    val_data, val_labels = torch.from_numpy(data["x_val"]).type(torch.FloatTensor), torch.from_numpy(
        data["y_val"]
    ).type(torch.FloatTensor)
    val_data = val_data[:, :, :]
    valset = TensorDataset(val_data, val_labels)

    test_data, test_labels = torch.from_numpy(data["x_test"]).type(torch.FloatTensor), torch.from_numpy(
        data["y_test"]
    ).type(torch.FloatTensor)
    test_data = test_data[:, :, :]
    testset = TensorDataset(test_data, test_labels)

    return trainset, valset, testset


def calculate_stats(output, target, class_indices=None):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)
      class_indices: list
        explicit indices of classes to calculate statistics for

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    # Class-wise statistics
    for k in class_indices:
        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)


        dict = {'AP': avg_precision,
                'auc': auc}
        stats.append(dict)

    return stats


def calculate_auroc(target, output, class_indices=None):
    output = torch.tensor(output)
    target = torch.tensor(target)
    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    # Class-wise statistics
    for k in class_indices:
        # # Average precision
        # avg_precision = metrics.average_precision_score(
        #     target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)
        
        # dict = {'AP': avg_precision,
        #         'auc': auc}
        stats.append(auc)

    return np.mean(stats)


if __name__ == "__main__":
    cfg = {"deepsea_root": "/localdisk/home/lericsso/code/einspace/data/deepsea"}
    trainset, valset, testset = build_nasbench360_deepsea_dataset(cfg)

    print("Traning set size:", len(trainset))
    print("Validation set size:", len(valset))
    print("Test set size:", len(testset))

    print(trainset[0][0].shape, trainset[0][1].shape)
    print(valset[0][0].shape, valset[0][1].shape)
    print(testset[0][0].shape, testset[0][1].shape)
