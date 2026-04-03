# Earth Observation Data Uploader (EOU)

The **Earth Observation Data Uploader** sub-system is designed to manage
the acquisition of satellite imagery from both public and restricted
repositories. Public repositories are typically accessible via API and
are often covered by a STAC catalogue, while certain datasets that are
not available as open data require manual upload mechanisms. The
sub-system addresses this dual need through two primary pathways. The
**Manual Data Loader** allows operators to import datasets from restricted
sources, such as specific commercial providers or legacy archives,
which cannot be processed automatically. If these files are in
standard formats like GeoTIFF, the system supports automatic
recognition to minimize manual configuration. The **Data Acquisition
Gateway** automates the retrieval of new scenes by connecting to
services like Copernicus, Google Earth Engine, and NASA
Earthdata. This module actively monitors selected data sources and
downloads relevant imagery based on predefined criteria. All ingested
data is subsequently passed to the **Data Extraction** logic for
validation and integration into the processing pipeline.

![EO Data Uploader Architecture](../../images/eou_subsystem.png)

## Usage

Utilize the `DataAcquisitionGateway` module to automatically retrieve
new scenes by connecting to services such as Copernicus, Google Earth
Engine, and NASA Earthdata. By default the `DataAcquisitionGateway` is
using EODAG package to search and download new data products.

```py
dag_module = DataAcquisitionGateway()

wkt_str = 'POLYGON((...))'
search_filter = {
    'provider': 'cop_dataspace',
    'start': '2026-01-01',
    'end': '2026-01-29',
    'productType': 'S2_MSI_L2A',
}

results = dag_module.search(
    geom=wkt_str,
    **self.search_filter,
)
data_path = dag_module.download(results[0], quicklook=True)
```

For data that cannot be automatically retrieved via
`DataAcquisitionGateway` and is available locally, use
`ManualFileLoader` module.

```py
file_path = '/../filename.tif'
loader_module = ManualFileLoader()
result = loader_module.check_file_validity(file_path)
```
