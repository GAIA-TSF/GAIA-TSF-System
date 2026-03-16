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

Workflow: 
1. Train baseline model
```
python3 -m subsystems.map.learning.lstm_learning --dataset synthetic
```
2. Validate model
```
python3 -m subsystems.map.learning.validate_lstm --dataset synthetic
```
3. Run hyperparameters tuning
```
python3 -m subsystems.map.learning.tune_lstm \
    --dataset synthetic \
    --config subsystems/map/learning/config.yaml
```

4. Train best model and validate 
```
python3 -m subsystems.map.learning.lstm_learning --dataset synthetic

python3 -m subsystems.map.learning.validate_lstm --dataset synthetic
```

5. Registering model 
```
Done automatically by lstm_learning.py 
```

6. Run monitoring inference 
``` 
python3 -m subsystems.map.inference.lstm_inference --dataset synthetic
```

TESTS:  
Requirement ML_R_02: 
```
pytest subsystems/map/tests/test_training.py -v 

pytest subsystems/map/tests/test_validation.py -v 

pytest subsystems/map/tests/test_tuning.py -v
```

Requirement ML_R_03: 
pytest subsystems/map/tests/test_inference.py -v 
```

TODO: Requirement ML_R_06: model registry
```

```