import torch


class Trainer:
    """Generic trainer for MAP learning models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        early_stopping=False,
        patience=20,
    ):
        self._model = model.to(device)
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._device = device

        self._early_stopping = early_stopping
        self._patience = patience

    def fit(self, train_loader, val_loader, epochs, model_path=None):
        best_val_loss = float('inf')

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # checkpoint
            if model_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self._model.state_dict(), model_path)

                print(
                    f'Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f} | BEST MODEL SAVED'
                )

            elif epoch % 10 == 0:
                print(
                    f'Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}'
                )

        print('Best validation loss:', best_val_loss)

        return train_losses, val_losses

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        self._model.train()
        total_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            self._optimizer.zero_grad()
            outputs = self._model(inputs)
            loss = self._loss_fn(outputs, targets)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> float:
        """
        Fix len(test_loader) == 0:
        Sliding-window forecasting does not split by points.
        It splits by valid windows.
        - Detect empty validation set
        - Skip validation automatically
        """
        if len(dataloader) == 0:
            return float('nan')

        self._model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                outputs = self._model(inputs)
                loss = self._loss_fn(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    # save trained model
    def save_model(self, path: str):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path: str):
        self._model.load_state_dict(torch.load(path, map_location=self._device))
