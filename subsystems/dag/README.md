# Data Aggregation (DAG)

The **Data Aggregation** (DAG) sub-system serves as the critical
processing bridge that transforms harmonised data stored within the
Spatial Data Infrastructure (SDI) into structured inputs suitable for
machine learning analysis. Its primary function is to prepare data in
structures that are harmonised and ready for downstream consumption,
adhering to the principle that "garbage in, garbage out" dictates
model performance. The sub-system ingests multi-temporal satellite
image stacks (e.g., Sentinel-2) and co-located in-situ measurements
(e.g., pH, pore pressure) to produce model-ready feature tensors.

-![Data Agregation Architecture](../../images/dag_subsystem.png)

## Usage
