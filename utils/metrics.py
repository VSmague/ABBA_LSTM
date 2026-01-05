import numpy as np


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0

    return 200 * np.mean(
        np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]
    )
