import os
import requests

from qcl.logger import Logger

STAC_URL = os.environ.get("STAC_URL", "http://ip-172-31-19-31.eu-central-1.compute.internal:8080")
id = "SDI"

def stac(event, context):
    """
    Simple proxy to resend STAC query to VPN secured STAC
    :param event: System event from API Gateway
    :param context: System context from API Gateway
    :return: Response from STAC running on STAC_URL
    """
    method = event["requestContext"]["http"]["method"]
    path = event["rawPath"]
    query = event.get("queryStringParameters") or {}
    headers = event.get("headers") or {}

    Logger.debug(f"{id} STAC requested with method {method} on {path}")

    # Publicly available is read only
    if method not in ("GET", "HEAD"):
        return {
            "statusCode": 403,
            "body": "Forbidden"
        }

    # Proxy query for STAC server
    url = f"{STAC_URL}{path}"
    resp = requests.request(method, url, headers=headers, params=query)

    return {
        "statusCode": resp.status_code,
        "headers": dict(resp.headers),
        "body": resp.text,
    }
