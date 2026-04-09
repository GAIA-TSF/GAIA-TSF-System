### DAG coding context
DAG role: integration of EO and in-situ data into unified, analysis-ready datasets.

### 1. Architecture overview
See ./images/dag_architecture.png 

#### 2. Requirements: 
- DA_R_01: The sub-system shall transform raw time-series data and multi-temporal satellite image stacks stored in the SDI into meaningful, model-ready features.
- DA_R_02: The sub-system shall perform temporal harmonization to align timestamps between remote sensing observations and in-situ reference samples.
- DA_R_03: The sub-system shall execute spatial overlay operations to sample pixels around in-situ locations for data co-location.
- DA_R_04: The sub-system shall combine spectral data and derived indices, such as NDVI, NDWI, and NDSI, and add contextual features like slope and DEM to the dataset.
MAP: DA_R_05: The sub-system shall normalize input features using techniques such as Min-Max normalization (scaling values between 0 and 1) or Z-score standardization (mean of zero, standard deviation of one) to ensure features with broader ranges do not dominate the model.
- DA_R_08: The sub-system shall transform categorical data (e.g., site identifiers, status flags) into numerical formats using One-Hot Encoding or Label Encoding to ensure compatibility with machine learning algorithms.
- DA_R_09: The sub-system shall identify and handle missing values by employing imputation strategies (mean, median, or mode replacement) or by removing incomplete entries to maintain dataset integrity.
- DA_R_10: The sub-system shall apply logarithmic transformations to skewed data distributions or outliers to prevent them from distorting model predictions.


### 3. Module responsibilities

#### Ingestion Module Responsibility:
- Load EO data from SDI
- Stack into DataCube

#### Harmonization Module
- Implements DA_R_02 & spatial consistency
- Temporal alignment = optional (no in-situ)

#### Feature Engineering Module
- Slope Stability displacement feature extraction 
- AMD feature (index) extraction 
- Extendable: NDVI, NDWI, custom indices (DA_R_04)

#### Preprocessing Module Implements:
- DA_R_05 (normalization)
- DA_R_09 (missing values)
- DA_R_10 (log transforms)

#### Masking & Auxiliary Data: 
... 

#### Tensorization Module: 
- Output: T x H x W x C 
- Optional: N_pixels x T x C

#### Data Validation Module: 
- Data validatior

#### Pipeline Design: 
- Base pipeline (config) _> FeatureCube
- Slope stability pipeline -> FeatureCube
- AMD pipeline -> FeatureCube

#### Execution: 
- Pipeline executor 


After refactoring, respecting D5.1 design
```
subsystems/
└── dag/
    ├── __init__.py
    ├── config.yaml
    ├── core/                      # engine layer
    │   ├── data_model.py
    │   ├── registry.py
    │   └── executor.py
    │
    ├── data_import/               # (Layer 1)
    │   ├── __init__.py           
    │   ├── eo_loader.py           # EO Time Series 
    │   └── insitu/                # In-situ datasets
    │        ├── __init__.py
    │        ├── loader.py
    │        └── aligner.py
    │  
    ├── data_processing/           # (Layer 2)
    │   ├── __init__.py
    │   ├── alignment.py           # temporal alignment
    │   ├── harmonization.py       # spatial harmonization
    │   ├── masking.py
    │   ├── validation.py
    │   ├── preprocessing.py
    │   └── sampling.py            # spatio-temporal sampling
    │   
    ├── feature_engineering/       # (Layer 3)
    │   ├── __init__.py
    │   ├── aggregation.py         # multi-modal aggregation
    │   ├── eo_features.py         # spectral indices etc.
    │   └── tensorization.py       # ML tensor creation
    │   
    ├── pipelines/
    │   ├── base_pipeline.py
    │   ├── slope_pipeline.py
    │   └── amd_pipeline.py
    │ 
    └── tests/
        ├── test_amd_pipeline.py
        └── test_slope_pipeline.py
```


```
subsystems/
└── dag/                       # Data Aggregation subsystem
    ├── __init__.py            
    ├── config.yaml            # Config definitions 
    │   └── schema.py          
    │
    ├── modules/
    │   ├── ingestion
    │   ├── harmonization
    │   ├── feature_engineering
    │   ├── preprocessing
    │   ├── tensorization
    │   ├── masking 
    │   └── validation
    │
    ├── pipelines/
    │   ├── base_pipeline.py
    │   ├── slope_pipeline.py
    │   └── amd_pipeline.py
    │
    ├── core/
    │   ├── data_model.py      # core data structures
    │   ├── registry.py        # module registry
    │   └── executor.py        # pipeline execution engine
    │
    ├── utils/
    │   ├── raster.py
    │   ├── time.py
    │   └── geometry.py
    │
    └── tests/
        ├── test_subsystem.py
        ├── test_modules.py
        └── test_interfaces.py
```

### 4. Naming conventions
* No in-situ -> unsupervised-like design
* No dependency on labels
* Focus on temporal features
* KV-specific pipelines
* Avoid “one giant pipeline”
* Separate:
    - physics (slope)
    - chemistry (AMD)
* Lazy / chunked processing (IMPORTANT with hardware)
* Use of GDAL or rasterio windowed reading?
* Avoid loading full cube if unnecessary!!! 


### 5. Data flow 
* Raw EO data (S1/S2) [Ingestion]
* DataCube (T, H, W, C) 
    - [Spatial Harmonization] 
    - [Masking: AOI, water]
    - [Feature Engineering]
* FeatureCube
    - [Preprocessing: normalize, impute, transform]
    - [Validation]
    - [Tensorization]
- ML-ready tensor (T, H, W, C) => MAP subsystem


### 6. Key assumptions
* Future Extensions (KV, sensors, in-situ)
* Plug-in feature calcualtion
* GPU acceleration (PyTorch transforms)
* Streaming pipelines? 
* Multi-resolution fusion

## => A spatiotemporal feature abstraction layer between EO data and probabilistic models. 
