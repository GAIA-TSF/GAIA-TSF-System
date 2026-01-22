# Earth-Observation-Data-Uploader

Mechanisms for importing EO satellite imagery and metadata from multiple sources.

## Deployment

Build subsystem image:

```sh
docker build --tag gaia_tsf_eou:latest docker/
```

Run unit tests:

```
docker run --rm -v `pwd`:/opt/gaia_tsf \
 gaia_tsf_eou:latest \
 python3 -m pytest /opt/gaia_tsf/tests/test_subsystem.py
```

## Project reference

https://github.com/orgs/GAIA-TSF/projects/1
