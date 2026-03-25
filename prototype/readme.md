Instructions to run the AMD prototype: 

Modify path to your data dir in config.yaml FIRST!!! 

```
Run input setup
python3 setup_inputs.py

# In-situ downloader to get references
python3 download_insitu.py

# EO downloader to get S2 images 
python3 download_eo.py

# EO pre-processing: set THR and run 
python3 preprocessing.py 

# Data overlay
python3 data_overlay.py 

# Visualise TODO: fix 
python3 visualisation.py 

# Run MAP module (training and inference)

python3 run_map.py

# Calculate AMD index 
python3 amd_index.py 

# Detect alarm 
python3 detect_alarm.py 


```