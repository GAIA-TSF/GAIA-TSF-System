# Get item from s3

Lambda microservice that returns signed url for gaia-tsf-private bucket

## Setup

### Install serverless-python-requirements
```bash
$ npm install
```

### Deploy!
```bash:development
$ serverless deploy
```

or production
```bash:production
$ serverless deploy --stage live
```

## Testing
curl -H "Authorization: Bearer G2zyGp1OoIx7w5rOEHa8jbu45yfJlem" -H "Accept: application/json" https://41p89gaer6.execute-api.eu-central-1.amazonaws.com/dev/signurl?s3url=s3://gaia-tsf-private/landsat-9-l2/LC92020342024148LGN00_B03.tif
