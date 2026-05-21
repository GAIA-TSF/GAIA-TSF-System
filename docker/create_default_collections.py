#!/usr/bin/env python3
"""
Create default STAC collections directly in pgSTAC database.
This script runs after migrations and before API startup.
"""

import asyncio
import asyncpg
import json

# Database connection string
DSN = 'postgresql://stac:stac@pgstacdb:5432/stac'

# Define your default collections
DEFAULT_COLLECTIONS = [
    {
        'id': 'sentinel-2',
        'type': 'Collection',
        'title': 'Sentinel-2 Data',
        'description': 'Sentinel-2 satellite imagery collection',
        'extent': {
            'spatial': {'bbox': [[-180, -90, 180, 90]]},
            'temporal': {'interval': [['2015-06-23T00:00:00Z', None]]},
        },
        'links': [],
    },
    {
        'id': 'sentinel-1',
        'type': 'Collection',
        'title': 'Sentinel-1 Data',
        'description': 'Sentinel-1 collection',
        'extent': {
            'spatial': {'bbox': [[-180, -90, 180, 90]]},
            'temporal': {'interval': [['2015-06-23T00:00:00Z', None]]},
        },
        'links': [],
    },
    {
        'id': 'in-situ',
        'type': 'Collection',
        'title': 'In-Situ data',
        'description': 'Any In-Situ data collection',
        'license': 'proprietary',
        'extent': {
            'spatial': {'bbox': [[-180, -90, 180, 90]]},
            'temporal': {'interval': [['2015-06-23T00:00:00Z', None]]},
        },
        'links': [],
    },
    {
        'id': 'landsat-8',
        'type': 'Collection',
        'title': 'Landsat 8 Data',
        'description': 'Landsat 8 satellite imagery collection',
        'extent': {
            'spatial': {'bbox': [[-180, -90, 180, 90]]},
            'temporal': {'interval': [['2013-04-11T00:00:00Z', None]]},
        },
        'links': [],
    },
]


async def collection_exists(conn, collection_id):
    """
    Check if collection already exists in database.

    Args:
        conn: Database connection
        collection_id: ID of the collection to check

    Returns:
        bool: True if collection exists, False otherwise
    """
    exists = await conn.fetchval(
        'SELECT EXISTS(SELECT 1 FROM pgstac.collections WHERE id = $1)', collection_id
    )
    return exists


async def create_collection(conn, collection):
    """
    Create a single collection in pgSTAC database.

    Args:
        conn: Database connection
        collection: Collection object as dictionary
    """
    collection_id = collection['id']

    # Check if collection already exists
    if await collection_exists(conn, collection_id):
        print(f"Collection '{collection_id}' already exists, skipping...")
        return

    # Create collection using pgSTAC function
    try:
        await conn.execute(
            'SELECT * FROM pgstac.create_collection($1::jsonb)', json.dumps(collection)
        )
        print(f"Successfully created collection '{collection_id}'")
    except Exception as e:
        print(f"Error creating collection '{collection_id}': {e}")
        raise


async def create_default_collections():
    """
    Main function to create all default collections.
    Connects to database and processes each collection.
    """
    conn = None

    try:
        # Connect to database
        print('Connecting to pgSTAC database...')
        conn = await asyncpg.connect(DSN)
        print('Database connection established')

        # Process each collection
        for collection in DEFAULT_COLLECTIONS:
            await create_collection(conn, collection)

        print('All default collections processed successfully')

    except asyncpg.PostgresError as e:
        print(f'Database error: {e}')
        raise
    except Exception as e:
        print(f'Unexpected error: {e}')
        raise
    finally:
        # Always close connection
        if conn:
            await conn.close()
            print('Database connection closed')


if __name__ == '__main__':
    asyncio.run(create_default_collections())
