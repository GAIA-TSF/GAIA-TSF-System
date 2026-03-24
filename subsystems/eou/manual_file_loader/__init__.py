from pathlib import Path

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

from lib.base import GaiaBase, SubsystemId


class ManualFileLoader(GaiaBase):
    """Manual File Loader module for handling data from restricted
    sources or specific datasets that are not available through
    standard APIs. It provides an interface for operators to upload
    local files, such as JP2 or GeoTIFF images.
    """

    def __init__(self):
        super().__init__(SubsystemId.EOU)

    def check_file_validity(self, file_path: str) -> dict:
        """Open file using GDAL library and perform file validity
        integrity tests.

        Checks:
        - if file exists
        - if file may be open by GDAL
        - if number of rows and columns > 0
        - if number of bands > 0
        - if projection is defined and is valid (warning only)
        - if geotransformation parameters are defined (warning only)
        - if bands are defined
        - if sample data from each band may be read as numpy array
        - for sample data of each channel, it checks whether they do not contain only no-data values.

        :param str file_path: path to EO data file

        :return: result dictionary
        :rtype: dict
        """
        self.logger.info(f'Performing file validity check for {file_path}')
        result = {'path': str(file_path), 'valid': True, 'errors': [], 'warnings': []}

        if not Path(file_path).exists():
            result['valid'] = False
            result['errors'].append('File does not exist')
            return result

        try:
            ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        except RuntimeError as e:
            result['valid'] = False
            result['errors'].append(str(e))
            return result

        # driver check
        driver = ds.GetDriver().ShortName
        result['driver'] = driver

        # raster size
        if ds.RasterXSize <= 0 or ds.RasterYSize <= 0:
            result['errors'].append('Invalid raster dimensions')

        if ds.RasterCount <= 0:
            result['errors'].append('No raster bands found')

        # projection
        proj = ds.GetProjection()
        if not proj:
            result['warnings'].append('Missing projection')
        else:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj)
            if not srs.IsProjected() and not srs.IsGeographic():
                result['warnings'].append('Invalid spatial reference')

        # geo-transform
        gt = ds.GetGeoTransform(can_return_null=True)
        if gt is None:
            result['warnings'].append('Missing geotransform')
        else:
            px_w, px_h = gt[1], gt[5]
            if px_w == 0 or px_h == 0:
                result['errors'].append('Invalid pixel size')

        # band-level checks
        for i in range(1, ds.RasterCount + 1):
            try:
                band = ds.GetRasterBand(i)
            except RuntimeError:
                result['errors'].append(f'Band {i} missing')
                continue

            # try reading a small window
            try:
                arr = band.ReadAsArray(
                    0, 0, min(256, ds.RasterXSize), min(256, ds.RasterYSize)
                )
                if np.isnan(arr).all():
                    result['warnings'].append(f'Band {i} contains only NaN values')
            except RuntimeError as e:
                result['errors'].append(f'Band {i} read error: {str(e)}')

        ds = None  # close dataset

        if result['errors']:
            result['valid'] = False
            self.logger.info(
                f'File validity check failed: {";".join(result["errors"])}'
            )
        else:
            self.logger.info('File validity check: no issues detected')

        return result
