
import matplotlib.pyplot as plt
import torch

from ..dataset.insar import create_synthetic_insar_dataset
from ..learning import LearningModule
from . import InferenceModule

"""
Run prediction on synthetic dataset. 
- train model quickly
- run prediction
- plot observed vs predicted vs anomaly score.
"""


def main():
    
    print('config')
    look_back = 12
    horizon = 5
    device = torch.device('cpu')

    # -------------------------
    # Dataset
    # -------------------------
    print('dataset')
    dataset = create_synthetic_insar_dataset(
        length=80,
        noise_std=0.5,
        trend_amplitude=20.0,
        anomaly_magnitude=40.0,
        look_back=look_back,
        horizon=horizon,
    )

    # -------------------------
    # Train quick model
    # -------------------------
    print('learning')
    learning = LearningModule()

    model = learning.create_forecasting_model(
        input_size=1,
        hidden_size=32,
        num_layers=1,
        horizon=horizon,
    )

    print('training')
    trainer = learning.create_trainer(
        model=model,
        learning_rate=1e-3,
        device=device,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    for _ in range(60):
        trainer.train_epoch(loader)

    # -------------------------
    # Inference
    # -------------------------
    print('inference')
    inference = InferenceModule()

    predictor = inference.create_predictor(
        model=model,
        device=device,
        look_back=look_back,
        horizon=horizon,
    )

    displacement = dataset.displacement

    prediction = predictor.predict_series(displacement)

    residuals = predictor.compute_residuals(displacement, prediction)
    score = predictor.anomaly_score(residuals)

    # -------------------------
    # Plot
    # -------------------------
    print('plot')
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(displacement, label='Observed', color='black')
    plt.plot(prediction, label='Predicted', color='blue')
    plt.legend()
    plt.title('Prediction')

    plt.subplot(2, 1, 2)
    plt.plot(score, label='Anomaly score', color='red')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
