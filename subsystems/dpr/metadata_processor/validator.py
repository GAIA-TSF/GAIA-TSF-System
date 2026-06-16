import pystac
import requests

from lib.base import GaiaBase, SubsystemId
from lib.config import SettingsReader


class MetadataValidator(GaiaBase):
    """
    The automatic validation of generated metadata during ingestion.
    """

    def __init__(self):
        """Initialize metadata validator."""
        super().__init__(SubsystemId.DPR)

    def validate(self, metadata_path):
        """
        Validate provided metadata.

        :param str metadata_path: metadata to be validated
        :return: result dictionary with validation status
        :rtype: dict
        """
        result = {'path': str(metadata_path), 'valid': True, 'errors': [], 'warnings': []}

        try:
            # get collections
            settings = SettingsReader()
            stac_url = settings['sdi']['stac']['url']
            r = requests.get(f'{stac_url}/collections')
            r.raise_for_status()
            data = r.json()

            # read input metadata file
            obj = pystac.read_file(metadata_path)

            # If it's an Item, create a Catalog to hold it
            if isinstance(obj, pystac.Item):
                catalog = pystac.Catalog(
                    id='validation-catalog', description='Temporary catalog for validation'
                )
                catalog.add_item(obj)
            else:
                catalog = obj

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
                catalog.add_child(collection)

            # PySTAC's built-in validation is too lenient - it doesn't enforce that
            # Items reference only collections that actually exist in the catalog.
            # We add this explicit check to catch data errors early.
            for item in catalog.get_items():
                if item.collection_id and item.collection_id not in collection_ids:
                    result['valid'] = False
                    result['errors'].append(
                        f"Item '{item.id}' references undefined collection '{item.collection_id}'. "
                        f"Available collections: {sorted(collection_ids)}"
                    )

            catalog.validate()

        except requests.RequestException as e:
            result['valid'] = False
            result['errors'].append(f'Failed to fetch collections from STAC API: {str(e)}')
        except pystac.STACValidationError as e:
            result['valid'] = False
            result['errors'].append(f'STAC validation failed: {str(e)}')
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f'Metadata validation error: {str(e)}')

        return result
