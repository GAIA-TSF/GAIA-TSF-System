
from torch.utils.data import DataLoader, Subset

def create_dataloaders(dataset, train_idx, test_idx, batch_size):
    train_loader = DataLoader(
        Subset(dataset, list(train_idx)),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        Subset(dataset, list(test_idx)),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def create_model(learning_module, model_cfg, horizon):
    return learning_module.create_forecasting_model(
        input_size=model_cfg['input_size'],
        hidden_size=model_cfg['hidden_size'],
        num_layers=model_cfg['num_layers'],
        horizon=horizon,
        dropout=model_cfg['dropout'],
        bidirectional=model_cfg['bidirectional'],
    )

