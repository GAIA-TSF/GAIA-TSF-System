# Spatial Data Infrastructure (SDI)

The **Spatial Data Infrastructure** serves as the core data management
system for the architecture, responsible for the storage,
organization, and retrieval of all geospatial and monitoring data. Its
primary function is to store harmonized Earth Observation (EO) and
in-situ data, along with their associated metadata, in a structured
environment that facilitates efficient access by downstream analytics
and visualization sub-systems. The SDI creates a unified data layer
that supports both public and private data access, ensuring that raw
data is preserved for backup purposes while harmonized datasets are
made readily available for processing. It leverages standard protocols
such as the SpatioTemporal Asset Catalog (STAC) to index data,
allowing systems to easily query and locate relevant datasets based on
time and location. Additionally, the SDI provides specific APIs for
handling different data types, including coverages for raster data and
PostGIS for vector data, ensuring optimizing handling of diverse
monitoring inputs.

-![Spatial Data Infrastructure Architecture](../../images/sdi_subsystem.png)

## Infrastructure

The SDI is based on the following components:
* PostGIS for storing In-Situ data
* PostGIS for storing metadata
* PgSTAC for handling metadata in PostGIS
* STAC-fastapi for handling STAC requests

It is extended for development with:
* localstack for simulating S3 (AWS) services

## How to use it

There are four interfaces:
* import In-Situ data
* import EO data
* read In-Situ data
* read EO data

### Import In-Situ data

For importing In-Situ data is necessary to prepare JSON that describes the
content of archive file.
The archive file (in ZIP format) must contain:
* Just one JSON file
* One or more CSV files

It is necessary to include all items from the example file:
[isu_sample_data_item.json](isu_sample_data_item.json)

Very iportant items are:
* id
* collection
* names of the assets

Collection is used for naming schema in PostGIS.
Id and names are used for naming table inside schema.

There may be another items in the file that are according to STAC specification and
its extensions.

#### Basic usage

```python
from subsystems.sdi.loader import InSituDataLoader
zip_path = 'path/to/zip/file.zip'
importer = InSituDataLoader(zip_path=zip_path)
importer.import_zip()
```

#### Append data into existing table usage

```python
from subsystems.sdi.loader import InSituDataLoader
zip_path = 'path/to/zip/file.zip'
importer = InSituDataLoader(zip_path=zip_path)
importer.import_zip(append_data=True)
```

### Import EO data

For importing EO data is necessary to prepare JSON that describes the
content of archive file.
The archive file (in ZIP format) must contain:
* Just one JSON file
* One or more GeoTIFF or ZIP files

It is necessary to include all items from the example file:
[eou_sample_data_item.json](eou_sample_data_item.json)

Very iportant items are:
* id
* collection
* names of the assets

Collection is used for naming collection in STAC.
Id is used to identify item in collection in STAC.
Names are used to identify assets for the item.

There may be another items in the file that are according to STAC specification and
its extensions.

#### Basic usage

```python
from subsystems.sdi.loader import EarthObservationDataLoader
zip_path = 'path/to/zip/file.zip'
importer = EarthObservationDataLoader(zip_path=zip_path)
importer.import_zip()
```

### Read In-Situ data

Stored In-Situ may be read using PostgreSQL/PostGIS API.
The data are described in STAC, so it is possible to search for them
and then use metadata from asset.

```python
from subsystems.sdi.reader import SdiReader
bbox = [
    14.41200,
    50.08200,
    14.44000,
    50.09200
]
datetime_value = "2026-02-10T00:00:00Z"

query_string = f'bbox={",".join(map(str, bbox))}&datetime={datetime_value}'

# Use SdiReader to search for assets
sdi_client = SdiReader()
assets = sdi_client.search_assets(query_string)
```

The assets contain list of all assets in the specified area and date.
So you can find PostGIS tabes as well.
The metadata for now contains connection string (you have to replace username and password)
and columns description.

```json
{
  "href": "postgresql://user:password@postgis:5432/geodata#prague.measurement_ph_202602_data",
  "type": "application/x-postgresql",
  "roles": [
    "data"
  ],
  "title": "Measurement data stored in PostGIS",
  "table:columns": [
    {
      "name": "lon",
      "type": "number"
    },
    {
      "name": "lat",
      "type": "number"
    },
    {
      "name": "ph",
      "type": "number"
    },
    {
      "name": "timestamp",
      "type": "datetime"
    }
  ]
}
```

### Read EO data

Stored EO data assets can be downloaded using SdiReader.

```python
from subsystems.sdi.reader import SdiReader
bbox = [
    14.41200,
    50.08200,
    14.44000,
    50.09200
]
datetime_value = "2026-02-10T00:00:00Z"

query_string = f'bbox={",".join(map(str, bbox))}&datetime={datetime_value}'

# Use SdiReader to search for assets
sdi_client = SdiReader()
assets = sdi_client.search_assets(query_string, asset_name='B01')

asset = assets[0]
asset_url = asset.get('href')

# Download asset using StacClient
downloaded_path = sdi_client.download_asset(asset_url)

```

The file is downloaded into temporary folder. 
You may copy it anywhere you want.
