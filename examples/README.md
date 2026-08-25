# GAIA-TSF Examples

The `examples` directory contains standalone scripts demonstrating
typical GAIA-TSF workflows and usage. These scripts can be used as a
starting point for developing custom processing pipelines and for
testing individual subsystem functionalities.

## Docker

```sh
cd docker/
docker compose exec -u $(id -u):$(id -g) gaiatesting python3 -m examples.<script>.py
```

Replace `<script>` with the desired example module name (without the .py
extension), for example:

```sh
docker compose exec -u $(id -u):$(id -g) gaiatesting python3 -m examples.dpr_sentinel2_workflow.py
```