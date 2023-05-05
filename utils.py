from __future__ import annotations

import math
import os
import time

from pandas import DataFrame
import pandas as pd
import random
from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


class Samples:
    def __init__(self, data, label):
        self.data: np.ndarray = data
        self.label: np.ndarray = label


def get_data(normalize, binary_label) -> tuple[pd.DataFrame, Samples, Samples, Samples]:
    # check if the data is already downloaded
    if not os.path.exists("OnlineNewsPopularity/OnlineNewsPopularity.csv"):
        # download the data
        # !wget https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip
        # !unzip OnlineNewsPopularity.zip
        pass
    df = pd.read_csv("OnlineNewsPopularity/OnlineNewsPopularity.csv")
    df = df.rename(columns=lambda x: x.strip())
    df = df.iloc[:, 2:]

    data = np.array(df)
    x = data[:, :-1]

    if normalize:
        # normalize the data
        # data = (data - data.mean(axis=0)) / data.std(axis=0)
        x = (x - x.mean(axis=0)) / x.std(axis=0)
    # x = data[:, :-1]
    # converting the last column to boolean
    if binary_label:
        # assert not normalize
        y = np.array([elem >= 1400 for elem in data[:, -1]])
    else:
        y = np.array(data[:, -1])
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, random_state=1
    )
    # val_x, test_x, val_y, test_y = train_test_split(
    #     test_x, test_y, test_size=0.5, random_state=1
    # )
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=1
    )
    return (
        df,
        Samples(train_x, train_y),
        Samples(test_x, test_y),
        Samples(val_x, val_y),
    )


def train_model(model, train: Samples):
    time_start = time.time()
    model.fit(train.data, train.label)
    print("Time taken to train the model: ", time.time() - time_start)
    return model


def test_model(model, train: Samples, test: Samples, classification):
    model = train_model(model, train)
    pred = model.predict(test.data)
    if classification:
        return analyze_pred_bin(pred, test.label)
    else:
        return analyze_pred(pred, test.label)


def analyze_pred(pred, truth):
    # Print the mean squared error and R-squared score
    mse = mean_squared_error(truth, pred)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", math.sqrt(mse))
    print("R-squared Score:", r2_score(truth, pred))
    # print(np.mean(pred))
    # print(np.mean(truth))
    # bin_pred = pred >= 0
    # bin_truth = truth >= 0
    # print("Accuracy:", accuracy_score(bin_truth, bin_pred))
    return mse


def analyze_pred_bin(pred, truth):
    print("Binary cross entropy:", log_loss(truth, pred))
    print("Predicted mean:", np.mean(pred), "Actual mean:", np.mean(truth))
    accuracy = accuracy_score(truth, pred)
    print("Accuracy:", accuracy)
    return accuracy


def plot_2d(x, y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def optimize_hyperparameters(
    get_model: callable,
    param_generator,
    train: Samples,
    test: Samples,
    val: Samples,
    classification,
):
    best_hyperparameters = None
    best_accuracy = -1 if classification else float("inf")
    hyperparameters = []
    accuracies = []
    for p in param_generator:
        print("Trying hyperparameters:", p)
        model = get_model(p)
        accuracy = test_model(model, train, val, classification)
        print("Value:", accuracy)
        hyperparameters.append(p)
        accuracies.append(accuracy)
        if classification:
            if accuracy > best_accuracy:
                best_hyperparameters = p
                best_accuracy = accuracy
        else:
            if accuracy < best_accuracy:
                best_hyperparameters = p
                best_accuracy = accuracy

    print("Best hyperparameters:", best_hyperparameters)
    print("Best value:", best_accuracy)
    return best_hyperparameters, hyperparameters, accuracies
