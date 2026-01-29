# Proxy for STAC running in VPC

Lambda microservice that acts as a proxy for STAC service running in VPN

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

# Testing
```bash
curl -s https://13nxvn6iz1.execute-api.eu-central-1.amazonaws.com/search | jq
```
