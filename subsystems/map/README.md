# Machine-Learning-Predictive-Analytics

Trend detection, anomaly analysis, and dynamic risk scoring.

The Machine Learning  Predictive Analytics sub-system functions as the core intelligence engine of the GAIA-TSF architecture, responsible for transforming historical and current monitoring data into actionable predictive insights (trend, anomaly, and risk scoring). The sub-system is designed to execute comprehensive modeling workflows that include data ingestion, feature engineering, model training, inference, and output results upload.

![ML Predictive Analytics](../../images/map_subsystem.png)

## Deployment

Prepare and plot Mirmazloumi et al. (2023) and synthetic data of slope stability:

```
python3 -m subsystems.map.scripts.visualize_insar_dataset \
    --dataset mirmazloumi_2023


python3 -m subsystems.map.scripts.visualize_insar_dataset \
    --dataset synthetic \
    --anomaly-magnitude 5.0
```

Learn LSTM recurrent model on the Mirmazloumi et al. (2023) and synthetic dataset of slope stability: 

```
python3 -m subsystems.map.learning.lstm_learning \
    --dataset synthetic

python3 -m subsystems.map.learning.lstm_learning \
    --dataset mirmazloumi_2023
```


Run inference on synthetic dataset. 
```
python3 -m subsystems.map.inference.lstm_inference --dataset synthetic

python3 -m subsystems.map.inference.lstm_inference --dataset mirmazloumi_2023
```
