import torch


def create_lagged_series_continuous(series, lag):
    """
    series : tensor (N,)
    retourne :
    X : (N-lag, lag, 1)
    y : (N-lag, 1)
    """
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series[i:i+lag])
        y.append(series[i+lag])

    X = torch.tensor(X).unsqueeze(-1)
    y = torch.tensor(y).unsqueeze(-1)
    return X.float(), y.float()


def create_lagged_series_symbolic(series, lag):
    """
    series : torch.LongTensor (N,)  # symboles ABBA
    retourne :
    X : (N-lag, lag)
    y : (N-lag,)
    """
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series[i:i+lag])
        y.append(series[i+lag])

    X = torch.stack(X)   # (N-lag, lag)
    y = torch.tensor(y)  # (N-lag,)

    return X.long(), y.long()
