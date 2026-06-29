# GAIA-TSF System

![Code Quality](https://github.com/GAIA-TSF/GAIA-TSF-System/actions/workflows/code-quality.yml/badge.svg)
![Tests](https://github.com/GAIA-TSF/GAIA-TSF-System/actions/workflows/pytest.yml/badge.svg)

Major repository for GAIA-TSF System

See [CONTRIBUTING](CONTRIBUTING.md) file for details on contributing to the project.

## GAIA-TSF Subsystems

![GAIA Prototype Overall Architecture](./images/prototype_architecture.png)

Individual GAIA-TSF subsystems are located in the `subsystems` directory. Each GAIA-TSF
subsystem corresponds to a subdirectory within `subsystems` directory, named
according to the abbreviation of the respective GAIA-TSF subsystem.

| Abbreviation | GAIA-TSF Subsystem |
|-------------|-------------|
| ALE | Alert & Decision Support Engine |
| DAG | Data Aggregation |
| DPR | Data Processor |
| EOU | Earth Observation Data Uploader |
| ISU | In-Situ Data Uploader |
| MAP | Machine Learning & Predictive Analytics |
| NTF | Notifications |
| QCL | Quality Control and Logging Layer |
| REP | Reporting & Compliance |
| SDI | Spatial Data Infrastructure |
| VID | Visualisation & Dashboard |

## Testing

### Setting Up a Testing Environment

You have two options: either create a Docker environment or set up a
Python virtual environment.

#### Docker

Note: The password is required for downloading Earth Observation (EO)
data through EODAG or ASF. If the `docker/.env` file contains the
`GAIA_EOU_AUTH_CREDENTIALS_PASSWORD` environment variable, there is no
need to specify the password in `config.yaml`, as it will be read from
the environment automatically.

Build docker:

```sh
cd docker
docker compose up --build
```

Run tests for the testfile wished:

```sh
docker compose exec -u $(id -u):$(id -g) gaiatesting python3 -m pytest subsystems/subsystem/tests/testfile.py -v
```

For executing long-running tests (which are excluded from CI), use the slow pytest marker.

```sh
docker compose exec -u $(id -u):$(id -g) gaiatesting python3 -m pytest -m slow subsystems/subsystem/tests/testfile.py -v
```

For more information, see [docker/README.md](docker/README.md).

#### Virtual environment

As an alternative to using Docker, you can run tests or your own
custom scripts by setting up a local Python virtual environment. This
approach is useful if you prefer a lighter-weight setup or need more
direct control over dependencies and execution.

First, create and activate a virtual environment:

```sh
python3 -m venv venv
source venv/bin/activate
```

Before installing the Python dependencies, verify which version of
GDAL is installed on your system. Make sure that the same version is
specified in docker/requirements.txt to avoid compatibility
issues. Once aligned, proceed with installing the Python dependencies.

```sh
pip3 install -r docker/requirements.txt 
```

Next, disable database logging in the `config.yaml` file to avoid
connection issues during local execution. You can do this by
commenting out the relevant section:

```yaml
    # db:
    #   <<: *sdi_db
    #   dbname: 'logging'
```

Finally, run the desired test or your own script. For example, to
execute a specific pytest file:

```sh
python3 -m pytest subsystems/subsystem/tests/testfile.py -v
```

### Examples

See [examples](examples/) for standalone scripts
demonstrating typical GAIA-TSF workflows.

The example can be executed as follows:

```sh
cd docker/
docker compose exec -u $(id -u):$(id -g) gaiatesting python3 -m examples.eou_download_sentinel2
```
