from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pystac
import requests
import json

from lib.base import GaiaBase, SubsystemId
from lib.config import SettingsReader


class MetadataValidator(GaiaBase):
    """
    The automatic validation of generated metadata during ingestion.
    """

    def __init__(self):
        """Initialize metadata validator."""
        super().__init__(SubsystemId.DPR)

    def validate(self, metadata_path: Path):
        """
        Validate STAC metadata against a remote STAC API catalog.

        Performs comprehensive validation of STAC metadata files by:
        - Loading metadata from a local file (supports STAC Items and Catalogs)
        - Fetching collection definitions from a remote STAC API
        - Building a temporary catalog with all available collections
        - Verifying that all Items in the metadata reference only defined collections
        - Running PySTAC's built-in validation on the complete catalog

        Checks performed:
        - If metadata file can be read and parsed as valid STAC
        - If STAC API is reachable and returns valid collection data
        - If Items reference only collections that exist in the STAC API
        - Full STAC schema validation via PySTAC

        :param str metadata_path: path to local STAC metadata file (Item or Catalog)
        :return: result dictionary containing validation status and any errors/warnings
        :rtype: dict

        :return dict format:
            - 'path' (str): path to the validated file
            - 'valid' (bool): True if all validations passed, False otherwise
            - 'errors' (list): list of error messages (empty if valid=True)
            - 'warnings' (list): list of warning messages (non-blocking issues)

        Example result:
            {'path': '/data/sample.json', 'valid': True, 'errors': [], 'warnings': []}
            {'path': '/data/sample.json', 'valid': False,
             'errors': ["Item 'foo' references undefined collection 'bar'"],
             'warnings': []}
        """
        self.logger.info(f"Validating metadata: {metadata_path}")
        result = {
            'path': str(metadata_path),
            'valid': True,
            'errors': [],
            'warnings': [],
        }

        try:
            # get collections
            settings = SettingsReader()
            stac_url = settings['sdi']['stac']['url']
            r = requests.get(f'{stac_url}/collections')
            r.raise_for_status()
            data = r.json()

            # read input metadata file
            obj = pystac.read_file(metadata_path)

            # if it's an Item, create a Catalog to hold it
            catalog = pystac.Catalog(
                id='validation-catalog',
                description='Temporary catalog for validation',
            )

            # add collections
            collection_ids = set()
            for col_ref in data['collections']:
                # col_ref is a link/reference, fetch the full collection object
                col_id = col_ref.get('id') or col_ref.get('title')
                if not col_id:
                    # Try to extract ID from rel/href if available
                    continue

                collection_ids.add(col_id)
                col_url = f'{stac_url}/collections/{col_id}'
                col_response = requests.get(col_url)
                col_response.raise_for_status()
                col_json = col_response.json()

                # Ensure the collection has required fields for PySTAC
                if 'type' not in col_json:
                    col_json['type'] = 'Collection'
                if 'stac_version' not in col_json:
                    col_json['stac_version'] = '1.0.0'

                collection = pystac.Collection.from_dict(col_json)
                if col_id == obj.collection_id:
                    collection.add_item(obj)
                catalog.add_child(collection)

            self.logger.debug(f"Collections fetched from STAC: {', '.join(collection_ids)}")
            catalog.normalize_hrefs(stac_url)

            # PySTAC's built-in validation is too lenient - it doesn't enforce that
            # Items reference only collections that actually exist in the catalog.
            # We add this explicit check to catch data errors early.
            if obj.collection_id not in collection_ids:
                result['valid'] = False
                result['errors'].append(
                    f"Item references undefined collection '{obj.collection_id}'. "
                    f'Available collections: {sorted(collection_ids)}'
                )

            for child in catalog.get_children():
                for item in collection.get_items():
                    self.logger.debug(f"Validating item {item.id}")
                    item.validate()
                self.logger.debug(f"Validating collection {child.id}")
                child.validate()

        except requests.RequestException as e:
            result['valid'] = False
            result['errors'].append(
                f'Failed to fetch collections from STAC API: {str(e)}'
            )
        except pystac.STACValidationError as e:
            result['valid'] = False
            result['errors'].append(f'STAC validation failed: {str(e)}')
        except (FileNotFoundError, OSError) as e:
            result['valid'] = False
            result['errors'].append(f'File system error: {str(e)}')
        except json.JSONDecodeError as e:
            result['valid'] = False
            result['errors'].append(f'Invalid JSON in STAC API response: {str(e)}')
        except ValueError as e:
            result['valid'] = False
            result['errors'].append(
                f'Invalid value in metadata or API response: {str(e)}'
            )
        except pystac.STACError as e:
            result['valid'] = False
            result['errors'].append(f'PySTAC processing error: {str(e)}')

        self.logger.info(f"Validation results: {result}")
        return result
