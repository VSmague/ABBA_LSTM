import torch


def create_lagged_series_continuous(series, lag):
    """
    series : tensor (N,)
    retourne :
    X : (N-lag, lag, 1)
    y : (N-lag, 1)
    """
    X, y = [], []
    if len(series) <= lag:
        # Si la série est trop courte, on ne peut créer qu'une seule séquence partielle
        # On complète avec des zéros à gauche pour atteindre la longueur lag
        padding = torch.zeros(lag - len(series))
        padded_series = torch.cat([padding, series])
        X.append(padded_series)
        y.append(series[-1])
    else:
        for i in range(len(series) - lag):
            X.append(series[i:i+lag])
            y.append(series[i+lag])

    X = torch.stack(X).unsqueeze(-1)
    y = torch.stack(y).unsqueeze(-1)
    return X.float(), y.float()


def create_lagged_series_symbolic(series, lag):
    """
    series : torch.LongTensor (N,)  # symboles ABBA
    retourne :
    X : (N-lag, lag)
    y : (N-lag,)
    """
    X, y = [], []
    if len(series) <= lag:
        # Si la série est trop courte, on ne peut créer qu'une seule séquence partielle
        # On complète avec des zéros à gauche pour atteindre la longueur lag
        padding = torch.zeros(lag - len(series), dtype=torch.long)
        padded_series = torch.cat([padding, series])
        X.append(padded_series)
        y.append(series[-1])
    else:
        for i in range(len(series) - lag):
            X.append(series[i:i+lag])
            y.append(series[i+lag])

    X = torch.stack(X)   # (N-lag, lag)
    y = torch.tensor(y, dtype=torch.long)  # (N-lag,)

    return X.long(), y.long()
