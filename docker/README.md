## Push docker images into GitHub container repository

Here is the simplest way how to push the images into GitHub container repository.
The user has to have Maintainer/Owner rights of the organization.

```sh
GHCR_PAT=github_token_XXXX
echo $GHCR_PAT | docker login ghcr.io -u username --password-stdin
docker compose build
docker compose push
```

## Persistent storage

This project expects `storage.data_dir` in `config.yaml` to be mounted
inside the `gaiatesting` container.

The compose file defines a named volume `data` configured to bind a
host directory into the container. This keeps a named volume entry
while persisting data on the host. By default, it uses the
`./tests/data` directory on the host.

To use a different host directory, you can either run:

```
HOST_DATA_DIR=/data/gaia_tsf docker compose up
```

or create a `.env` file and run `docker compose up` to pick up
`HOST_DATA_DIR` from the file.
