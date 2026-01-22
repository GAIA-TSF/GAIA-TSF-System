curl -X POST http://localhost:8888/collections \
     -H "Content-Type: application/json" \
     -d '{
           "id": "sentinel-s2-l2a-cogs",
           "title": "Sentinel-2 L2A COG",
           "description": "Sentinel-2 Level-2A COG tiles",
           "extent": {
               "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
               "temporal": {"interval": [["2020-01-01T00:00:00Z", null]]}
           },
           "license": "proprietary"
         }'


curl -X POST http://localhost:8888/collections \
     -H "Content-Type: application/json" \
     -d '{
           "id": "sentinel-2-l2a",
           "title": "Sentinel-2 L2A COG MSP",
           "description": "Sentinel-2 Level-2A COG tiles from MSP",
           "extent": {
               "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
               "temporal": {"interval": [["2020-01-01T00:00:00Z", null]]}
           },
           "license": "proprietary"
         }'

curl -X POST http://localhost:8888/collections \
     -H "Content-Type: application/json" \
     -d '{
           "id": "landsat-9-l2",
           "title": "Landsat-9 L2",
           "description": "Landsat-2 Level-2 GeoTIFF tiles",
           "extent": {
               "spatial": {"bbox": [[-180.0, -90.0, 180.0, 90.0]]},
               "temporal": {"interval": [["2020-01-01T00:00:00Z", null]]}
           },
           "license": "proprietary"
         }'
