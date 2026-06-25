# In-Situ Data Uploader (ISU) Sub-system

The **In-Situ Data Uploader** is the core data ingestion hub for the GAIA-TSF platform. It is responsible for securely and efficiently collecting geotechnical and environmental data from physical sensors, external FTP directories, and high-frequency real-time data streams.

ISU employs a highly concurrent, decoupled modern architecture. All ingestion endpoints operate in parallel and converge at the central ETL Engine for standardized parsing and preliminary cleaning.

## Architecture Design

ISU strictly adheres to the D5.1 Architecture Specification, utilizing the Facade and Factory design patterns to ensure maximum decoupling.

![In-Situ Data Uploader Architecture](../../images/isu_subsystem.png)

The core module consists of a main orchestrator (Facade) and four sub-engines:

1. **InSituDataUploader (Facade):** The master controller. Provides a unified interface to start and stop the subsystem, managing component lifecycles and dependency injection.

2. **ETLEngine (Central Processing Hub):** Contains an intelligent routing ParsingEngine. It identifies CSV/Excel files, extracts timestamps, evaluates confidence scores, and acts as the centralized callback endpoint for all streaming data. CSV files are read with automatic encoding detection (UTF-8, GBK, GB18030, Latin-1 fallback) so real-world datasets exported from regional tools parse correctly, and multi-bit QC flag strings (e.g. `000000000000000000`) are normalised into the standard pass/fail QC column.

3. **BulkUploadScheduler (Background Scanner):** A daemon-backed scheduler. Depending on `bulk_source_type`, it either scans the `data/input` directory (archiving processed files to `data/processed`) or fetches new files remotely over HTTPS, S3, FTP, or SFTP and routes them to the ETL engine.

4. **ManualFileLoader (Manual API):** Designed to support frontend Web UIs or RESTful APIs. It accepts local file paths or raw byte streams and feeds them directly into the ETL engine.

5. **StreamingDataHandler (Real-time Stream Processor):** A built-in Factory that supports seamless subscription to Kafka, AWS Kinesis, and OGC SensorThings APIs. It features integrated QA/QC validation and graceful degradation.

## Directory Structure

```
subsystems/isu/
├── __init__.py                # ISU Facade class (InSituDataUploader)
├── etl_engine/                # Central ETL Hub
│   ├── __init__.py            # ETL Engine initialization
│   ├── pipeline.py            # Main ETL engine class
│   └── parsers/               # Intelligent file parsers module
│       ├── base.py            # Multi-encoding CSV reader & QC flag normalisation
│       └── slope.py           # Sensor-type detection (keywords, units, dataset columns)
├── bulk_upload_scheduler/     # Background bulk file scanner
│   ├── __init__.py            # Source-type routing, scanning & file archiving logic
│   ├── scheduler.py           # Pure background thread scheduler
│   └── source_fetchers.py     # HTTPS/S3/FTP/SFTP remote file fetch functions
├── manual_file_loader/        # API for manual single-file uploads
│   └── __init__.py
├── streaming_data_handler/    # Real-time streaming data processor
│   ├── __init__.py            # Factory router & GaiaBase injector
│   └── stream_handler.py      # Concrete Kafka/Kinesis/OGC consumers
└── tests/                     # Automated integration & unit test suite
    └── assets/isu/            # Real-world dataset fixtures (e.g. Piezometer1.csv, Piezometer2.csv)
```

The test suite includes a full-pipeline integration test (`test_integration_ISU_002`) that runs two
real piezometer datasets end-to-end through ISU → QCL → DPR → SDI: `Piezometer1.csv` (UTF-8,
multi-depth deployment, 18-bit QC flag strings) and `Piezometer2.csv` (GBK-encoded dam monitoring
data with `DataSetI`–`IV` and `X`/`Y`/`H(mm)` displacement columns). It exercises ISU parsing, QCL
validation, DPR STAC packaging, and SDI import in one run.

> **Note:** SDI persistence in this pipeline currently relies on a `store_qc_result()` stub for the
> QCL dispatch contract — it still needs to be replaced with real persistence logic (a PostGIS table
> or STAC record) before the integration is production-ready.

## Key Features

1. **100% GaiaBase Driven:** All core classes inherit from `lib.base.GaiaBase`, enabling out-of-the-box auto-injection of system-level configurations (`self.settings`) and logging (`self.logger`).

2. **Graceful Degradation:** If required streaming parameters (e.g., Kafka brokers) are missing, the system safely logs an error and disables that specific channel without disrupting other functionalities.

3. **Dynamic Factory Routing:** Seamlessly switch underlying data sources (Kafka ↔ Kinesis ↔ OGC) simply by modifying the `stream_source_type` in `settings.yaml`, requiring zero code changes.

4. **Integrated QA/QC Gatekeeping:** Streaming data automatically invokes the injected `qc_layer.check()` for strict metric validation before entering the ETL engine.

## Configuration

All ISU configurations are driven by the global `settings.yaml`. Below is an example of the available ISU parameters:

```yaml
# settings.yaml example
isu:
  # --- Bulk Scanning Config ---
  bulk_scan_interval_sec: 10

  # Valid options: 'local', 'https', 's3', 'ftp', 'sftp'
  bulk_source_type: "local"

  # 'local' specifics (scans input_dir, archives to processed_dir)
  input_dir: "data/input"
  processed_dir: "data/processed"

  # 'https' specifics: a fixed list of file URLs to download each scan
  https_urls:
    - "https://example.org/sensors/latest.csv"

  # 's3' specifics
  s3_bucket: "gaia-tsf-bulk-uploads"
  s3_prefix: "in-situ/"
  s3_region: "eu-central-1"

  # 'ftp' specifics
  ftp_host: "ftp.example.org"
  ftp_user: "gaia"
  ftp_password: "secret"
  ftp_remote_dir: "/incoming"
  ftp_port: 21

  # 'sftp' specifics (use sftp_key_path for key-based auth, otherwise sftp_password)
  sftp_host: "sftp.example.org"
  sftp_user: "gaia"
  sftp_password: "secret"
  sftp_key_path: "/secrets/sftp_id_rsa"
  sftp_remote_dir: "/incoming"
  sftp_port: 22

  # --- Real-time Streaming Config ---
  # Valid options: 'kafka', 'kinesis', 'sensorthings', 'none'
  stream_source_type: "kafka"

  # Kafka specifics
  kafka_broker: "localhost:9092"
  kafka_topics: ["slope_stability", "water_quality"]

  # Kinesis specifics
  kinesis_stream: "tsf_live_stream"
  kinesis_region: "eu-central-1"
  kinesis_iterator_type: "LATEST"

  # OGC SensorThings specifics
  ogc_broker: "mqtt.example.org:1883"
  ogc_datastream_id: "99"
```
