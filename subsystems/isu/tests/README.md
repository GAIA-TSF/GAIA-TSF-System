# In-Situ Data Uploader (ISU)

## Overview
The ISU subsystem is responsible for ingesting data from ground-based sensors.
It supports:
1. Manual file upload via API.
2. Automated file ingestion from remote storage (mocked).
3. Automatic parser selection based on file content.

## Architecture


## Modules
* **parsers**: Handles file content analysis and data extraction.
* **scheduler**: Manages background jobs for automated ingestion.