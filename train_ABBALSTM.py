import copy
import torch
import os
import torch.nn as nn

from utils.data_handler import create_lagged_series_symbolic


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    patience_counter
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        checkpoint["epoch"] + 1,
        checkpoint["best_val_loss"],
        checkpoint["patience_counter"]
    )


def temporal_train_val_split(series, val_ratio=0.2):
    n = len(series)
    split = int(n * (1 - val_ratio))
    return series[:split], series[split:]


def train_abba(
    model,
    series,
    epochs=200,
    lr=1e-3,
    batch_size=32,
    val_ratio=0.2,
    patience=15,
    device="cpu",
    checkpoint_path=None,
    resume=False
):
    if model.stateful and batch_size != 1:
        raise ValueError("Stateful LSTM requires batch_size=1")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not torch.is_tensor(series):
        series = torch.tensor(series, dtype=torch.long)

    lag = model.lag

    # Split temporel
    train_series, val_series = temporal_train_val_split(series, val_ratio)
    X_train, y_train = create_lagged_series_symbolic(train_series, lag)
    X_val, y_val = create_lagged_series_symbolic(val_series, lag)

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    # === Initialisation reprise ===
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    if resume:
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise ValueError("Checkpoint introuvable pour reprise")
        start_epoch, best_val_loss, patience_counter = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        print(f"Reprise du training à l'epoch {start_epoch}")

    # === Loop d'entraînement ===
    for epoch in range(start_epoch, epochs):

        # -------- TRAIN --------
        model.train()
        if model.stateful:
            model.reset_states()

        train_loss = 0.0

        for i in range(0, X_train.size(0), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]

            if not model.stateful:
                model.reset_states()

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= (X_train.size(0) // batch_size + 1)

        # -------- VALID --------
        model.eval()
        if not model.stateful:
            model.reset_states()

        with torch.no_grad():
            val_loss = 0.0
            for i in range(0, X_val.size(0), batch_size):
                xb = X_val[i:i+batch_size]
                yb = y_val[i:i+batch_size]

                if not model.stateful:
                    model.reset_states()

                y_pred = model(xb)
                loss = criterion(y_pred, yb)
                val_loss += loss.item()

            val_loss /= (X_val.size(0) // batch_size + 1)

        # -------- EARLY STOPPING --------
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # -------- CHECKPOINT --------
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                patience_counter
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train: {train_loss:.6f} "
                f"Val: {val_loss:.6f}"
            )

        if patience_counter >= patience:
            print(
                f"Early stopping à l'epoch {epoch+1} "
                f"(best val loss = {best_val_loss:.6f})"
            )
            break

    # Restaurer le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
