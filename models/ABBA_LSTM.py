import torch
import torch.nn as nn


class ABBALSTM(nn.Module):
    def __init__(
        self,
        n_symbols,
        embedding_dim=16,
        hidden_sizes=[50],
        lag=10,
        stateful=False
    ):
        super(ABBALSTM, self).__init__()

        self.lag = lag
        self.stateful = stateful

        self.embedding = nn.Embedding(n_symbols, embedding_dim)
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        self.abbalstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_size = embedding_dim if i == 0 else hidden_sizes[i - 1]
            self.abbalstm_layers.append(
                    nn.LSTM(
                        input_size=in_size,
                        hidden_size=hidden_sizes[i],
                        batch_first=True
                    )
                )
        self.fc = nn.Linear(hidden_sizes[-1], n_symbols)

        self.hidden_states = None

    def reset_states(self):
        self.hidden_states = None

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)
        device = x.device
        new_hidden_states = []
        for i, lstm in enumerate(self.abbalstm_layers):
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
        
        logits = self.fc(x[:, -1, :])
        return logits

    def forecast(self, initial_symbols, horizon):
        self.eval()
        if self.stateful:
            self.reset_states()
        
        history = initial_symbols.clone().to(next(self.parameters()).device)
        predictions = []

        for _ in range(horizon):
            x = history[-self.lag:].unsqueeze(0)
            with torch.no_grad():
                logits = self(x)
                s = torch.argmax(logits, dim=1)
            predictions.append(s.item())
            history = torch.cat([history, s])

        return predictions
