import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import json

# Add lambda folder to sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent / "lambda"))

from stac.handler import stac

class TestSTACHandler:
    """Test suite for the STAC Lambda handler."""

    # ---------------------------------
    # Helper method to create mock API Gateway event
    # ---------------------------------
    @staticmethod
    def make_event(method="GET", path="/items", query=None, headers=None):
        """
        Create a mock API Gateway event for testing the STAC handler.
        """
        return {
            "requestContext": {"http": {"method": method}},
            "rawPath": path,
            "queryStringParameters": query or {},
            "headers": headers or {},
        }

    # ---------------------------------
    # Test forbidden HTTP methods
    # ---------------------------------
    def test_forbidden_method(self):
        """
        Test that any non-GET/HEAD HTTP methods return 403 Forbidden.
        """
        event = self.make_event(method="POST")
        response = stac(event, None)

        assert response["statusCode"] == 403
        assert response["body"] == "Forbidden"

    # ---------------------------------
    # Test proxy GET request with mocked requests
    # ---------------------------------
    @patch("stac.handler.requests.request")
    def test_get_request(self, mock_request):
        """
        Test that a GET request is correctly proxied and returns expected response.
        """
        # Mock the external STAC response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.text = '{"features": []}'
        mock_request.return_value = mock_resp

        event = self.make_event(method="GET", path="/collections")
        response = stac(event, None)

        # Validate response
        assert response["statusCode"] == 200
        assert response["headers"]["Content-Type"] == "application/json"
        assert response["body"] == '{"features": []}'

        # Validate that requests.request was called correctly
        mock_request.assert_called_once_with(
            "GET",
            "http://ip-172-31-19-31.eu-central-1.compute.internal:8080/collections",
            headers={},
            params={}
        )

    # ---------------------------------
    # Test proxy HEAD request
    # ---------------------------------
    @patch("stac.handler.requests.request")
    def test_head_request(self, mock_request):
        """
        Test that a HEAD request is correctly proxied.
        """
        # Mock the external STAC response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.text = ""
        mock_request.return_value = mock_resp

        event = self.make_event(method="HEAD", path="/status")
        response = stac(event, None)

        # Validate response status
        assert response["statusCode"] == 200

        # Validate that requests.request was called correctly
        mock_request.assert_called_once_with(
            "HEAD",
            "http://ip-172-31-19-31.eu-central-1.compute.internal:8080/status",
            headers={},
            params={}
        )
