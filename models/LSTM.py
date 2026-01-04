import torch
import torch.nn as nn
import numpy as np


class TimeSeriesLSTM(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_sizes=[50, 50],
        output_size=1,
        lag=10,
        stateful=False
    ):
        """
        input_size   : nombre de variables (1 pour une série univariée)
        hidden_sizes : liste, ex [50, 50] = 2 layers de 50 cellules
        output_size  : dimension de sortie
        lag          : nombre de retards (l)
        stateful     : True = stateful LSTM, False = stateless
        """
        super(TimeSeriesLSTM, self).__init__()

        self.lag = lag
        self.stateful = stateful
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        # Construction dynamique des layers LSTM
        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_size = input_size if i == 0 else hidden_sizes[i - 1]
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_sizes[i],
                    batch_first=True
                )
            )

        # Couche de sortie
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

        # États internes (pour stateful)
        self.hidden_states = None

    def reset_states(self):
        """À appeler entre deux séries si stateful=True"""
        self.hidden_states = None

    def forward(self, x):
        """
        x : (batch_size, lag, input_size)
        """
        batch_size = x.size(0)
        device = x.device

        new_hidden_states = []

        for i, lstm in enumerate(self.lstm_layers):

            # Initialisation des états
            if self.stateful and self.hidden_states is not None:
                h0, c0 = self.hidden_states[i]
            else:
                h0 = torch.zeros(1, batch_size, self.hidden_sizes[i]).to(device)
                c0 = torch.zeros(1, batch_size, self.hidden_sizes[i]).to(device)

            x, (hn, cn) = lstm(x, (h0, c0))
            new_hidden_states.append((hn.detach(), cn.detach()))

        if self.stateful:
            self.hidden_states = new_hidden_states

        # Dernier pas de temps
        x = x[:, -1, :]
        output = self.fc(x)

        return output

    def forecast(self, initial_series, horizon):
        """
        initial_series : tensor (lag,)
        horizon        : k
        """

        self.eval()
        device = next(self.parameters()).device

        if self.stateful:
            self.reset_states()
        
        if isinstance(initial_series, np.ndarray):
            initial_series = torch.tensor(
                initial_series, dtype=torch.float32
            )
        history = initial_series.clone().to(device)
        predictions = []

        for _ in range(horizon):
            x = history[-self.lag:].view(1, self.lag, 1)
            with torch.no_grad():
                y_hat = self(x)
            predictions.append(y_hat.item())
            history = torch.cat([history, y_hat.view(1)])
        return torch.tensor(predictions)