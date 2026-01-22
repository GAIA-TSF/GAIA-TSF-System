# Example curl commands for collections and searches.

Other examples are in the separate files

## Calling from localhost for testing

```bash
curl -s http://localhost:8888/collections/sentinel-s2-l2a-cogs/items | jq
curl -s "http://localhost:8888/collections/sentinel-s2-l2a-cogs/items?bbox=-180,-90,180,90&datetime=2025-08-31T00:00:00Z/2025-08-31T23:59:59Z" | jq
curl -X DELETE "http://localhost:8888/collections/sentinel-2-l2a/items/S2C_MSIL2A_20250831T075631_R035_T35JNM_20250831T113919"
```

## External STAC (Element 84)   

```bash
curl -s https://earth-search.aws.element84.com/v0/search | jq
```

"query": {
"eo:cloud_cover": {
"lte": 20
}
}

## External STAC (Microsoft)

```bash
curl -X POST "https://planetarycomputer.microsoft.com/api/stac/v1/search" \
-H "Content-Type: application/json" \
-d '{
"collections": ["sentinel-2-l2a"],
"bbox": [-5.0, 50.0, 0.0, 55.0],
"datetime": "2024-01-01T00:00:00Z/2024-02-01T23:59:59Z",
"limit": 5,
"query": {
"eo:cloud_cover": {
"lte": 20
}
}
}'
```

## GAIA-TSF STAC exposed via proxy

```bash
curl -s https://13nxvn6iz1.execute-api.eu-central-1.amazonaws.com/search | jq
curl -s https://13nxvn6iz1.execute-api.eu-central-1.amazonaws.com/collections/landsat-9-l2/items | jq
```
