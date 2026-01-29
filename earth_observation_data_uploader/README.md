# Earth Observation Data Uploader

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

![EO Data Uploader Architecture](./images/EO_component.drawio.png)

## Deployment

Build subsystem image:

```sh
docker build --tag gaia_tsf_eou:latest docker/
```

Run unit tests:

```
docker run --rm -v `pwd`/..:/opt/gaia_tsf \
 gaia_tsf_eou:latest \
 python3 -m pytest /opt/gaia_tsf/earth_observation_data_uploader/tests/test_subsystem.py
```
