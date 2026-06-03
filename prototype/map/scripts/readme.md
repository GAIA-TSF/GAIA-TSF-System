
### This is short note on prototyping scripts

**DAG/MAP Scripts**                   Description 

1. amd_eda.py                         Exploratory data analysis (before any modelling) 
2. amd_spectral_indicators.py         Calculate spectral EO products 
3. amd_temporal_features.py           Calculate temporal features (dynamics) 
4. amd_anomalies.py                   Detect anomaly scoring  
5. amd_risk_maps.py                   Spatial aggregation  
6. xai_analysis.py                    Calculate explainability       


**EDA** 

Louis' profile reproduction (B2 thr cloud masking) 
```
amd_eda_temporal_plot.py
``` 

EDA from geodata: saves histograms, temporal plots, and overall JSON summary
``` 
amd_eda_geodata_amd_clean_one_season
```

AMD EDA (one year so far)
```
amd_eda.py
```

**Modelling** 
This sript is initial prototype to test how to monitor AMD anomaly, deviation from clean water.
```
amd_modelling_gee_csv.py
``` 

This script is old and shall not be used. 
```
amd_modelling_geodata.py
```

The next script explores only the idea of dashboard. 
```
amd_modelling_dashboard.py
``` 

