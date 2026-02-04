import pandas as pd
import xarray as xr
import geopandas as gpd 
import numpy as np
import rioxarray as rxr
import os
from datetime import datetime
import re
import warnings

def default(self, row, resolution = None, static = True, **kwargs):
    """
    Default loader function to load data based on file extension.

    Load data from various file formats (CSV, GeoPackage, Shapefile, GeoTIFF, NetCDF)
    and return the appropriate data structure.

    Parameters
    ----------
    self : object
        The class instance.
    row : dict
        A dictionary containing dataset metadata including file path, extension,
        and loading parameters.
    resolution : str, optional
        The resolution to filter files by. Required if the dataset defines
        multiple resolutions. Default is None.
    static : bool, optional
        Flag indicating whether to load static files (True).
        Default is True. The default loader only supports static files.
    **kwargs : dict
        Additional keyword arguments passed to the underlying loading functions
        (e.g., pd.read_csv, gpd.read_file, rxr.open_rasterio, xr.open_mfdataset).

    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame or xr.Dataset
        The loaded data in the appropriate format based on file extension:

        - 'csv': :class:`pandas.DataFrame`
        - 'gpkg' or 'shp': :class:`geopandas.GeoDataFrame`
        - 'tif': :class:`xarray.Dataset`
        - 'nc': :class:`xarray.Dataset`

    Raises
    ------
    FileNotFoundError
        If no files matching the criteria are found.
    ValueError
        If the file extension is not supported or if static flag is not True.
    """
    
    # Ensure static flag is True for default loader
    if static is not True:
        raise ValueError("The default loader only supports static files. "
                         "To load annual files, please use a custom loader function.")

    # Extract parameters from row
    path, ext, skip_lines, no_data, ignore_dirs, ignore_files, loader, resolutions, static_patterns = self._extract_row_params(row)
        
    # Find files
    files = self._recursive_find_files(path, ext, ignore_dirs = ignore_dirs, ignore_files = ignore_files)

    # Filter files based on resolution if specified.
    if resolution is not None:
        files = _filter_resolution_files(
            files,
            resolution = resolution,
            static = static,
            static_patterns = static_patterns,
            resolutions = resolutions
        )

    # If no files found, raise error
    if not files:
        raise FileNotFoundError(
            f"No files found with extension '{ext}' in {path}\n"
            f"Ignoring directories: {ignore_dirs}"
        )

    # Load based on extension type
    # ------------------------------
    # CSV -> Pandas
    if ext == "csv":

        # Get kwargs (or set defaults)
        low_memory = kwargs.pop("low_memory", False)
        
        # Load each CSV into a DataFrame and append to data_list
        data_list = []
        for f in files:
            data = pd.read_csv(f, skiprows = skip_lines, low_memory = low_memory, **kwargs)
            data_list.append(data)
        
        # Concatenate all CSVs into a single DataFrame
        output = pd.concat(data_list, ignore_index = True)

        # Replace no_data values with NaN if specified
        if no_data is not None:
            output = output.replace(no_data, np.nan)

        return output 

    # GPKG / SHP -> GeoPandas
    if ext in ("gpkg", "shp"):
        
        # Load each file into a GeoDataFrame and append to data_list
        data_list = []
        for f in files:
            data = gpd.read_file(f)
            data_list.append(data)
        
        # Concatenate all GeoDataFrames into a single GeoDataFrame
        output = gpd.GeoDataFrame(pd.concat(data_list, ignore_index = True))

        # Replace no_data values with NaN if specified
        if no_data is not None:
            output = output.replace(no_data, np.nan)

        return output

    # TIF -> rioxarray / xarray
    if ext in ("tif"):

        # Get kwargs (or set defaults)
        masked = kwargs.pop("masked", True)
        
        # Load each TIF into an xarray DataArray and store in a dict. Using masked = True automates handling of NaN values.
        data_dict = {}
        for f in files:
            name = f.stem
            data = rxr.open_rasterio(f, masked = masked)

            # Squeeze out single-size 'band' dimension if present
            if "band" in data.dims and data.band.size == 1:
                data = data.squeeze("band", drop = True)

            # Store in dict
            data_dict[name] = data

        # Combine all DataArrays into a single Dataset
        output = xr.Dataset(data_dict)

        return output

    # NetCDF -> xarray
    if ext in ("nc"):

        # Get kwargs (or set defaults)
        combine = kwargs.pop("combine", "by_coords")

        output = xr.open_mfdataset(files,
                                    combine = combine,
                                    **kwargs)

        return output

    raise ValueError(f"Extension '{ext}' is currently not supported for loading. Use one of: csv, gpkg, shp, tif, nc.")


def _extract_year_range_from_filename(filename):
    """
    Extracts the start and end years from a filename.

    Assumes years are represented as four-digit numbers (e.g., 1990, 2020) in the filename.
    For example, from a filename like:
        ``Antarctica_ice_velocity_<start_year>_<end_year>_1km_<version>.nc``
        ``Central_Antarctica_ice_velocity_<year>.nc``
    it will extract the years and return them as integers.

    Parameters
    ----------
    filename : :class:`str`
        The filename to extract years from.

    Returns
    -------
    class:`tuple` of :class:`int`
        A tuple containing the start year and end year as integers.

    Raises
    ------
    ValueError
        If no four-digit years are found in the filename.

    Examples
    --------
    >>> _extract_year_range_from_filename('Antarctica_ice_velocity_1990_2020_1km_v1.nc')
    (1990, 2020)
    >>> _extract_year_range_from_filename('Central_Antarctica_ice_velocity_2015.nc')
    (2015, 2015)
    """

    # Use regex to find all four-digit year patterns in the filename
    base_name = os.path.basename(filename)
    years = re.findall(r'(19\d{2}|20\d{2})', base_name)
    years = [int(y) for y in years]

    # If no years found, raise an error
    if not years:
        raise ValueError(f"No years found in filename: {filename}")

    # Return the first and last years found
    years = sorted(set(years))
    return years[0], years[-1]

def _filter_resolution_files(files, resolution = None, static = None, static_patterns = None, resolutions = None):
    """
    Filter a list of files based on static/annual mode and resolution metadata.

    Parameters
    ----------
    files : :class:`list`
        List of file objects or paths to filter.
    resolution : :class:`str`, optional
        The resolution to filter files by. Required if the dataset defines multiple resolutions.
    static : :class:`bool`, optional
        Flag indicating whether to filter for static files (True) or annual files (False).
    static_patterns : :class:`list` or ``None``, optional
        List of string patterns that identify static files. If None, no static filtering is applied.
    resolutions : :class:`dict` or ``None``, optional
        Dictionary containing resolution metadata for static and/or annual modes. If None, no resolution filtering is applied.

    Returns
    -------
    :class:``list``
        Filtered list of files matching the specified criteria.

    Raises
    ------
    ValueError
        If required parameters are missing, or if filtering criteria are not met.

    Examples
    --------
    >>> files = [Path('file_static_1km.tif'), Path('file_annual_1km_2020.tif')]
    >>> static_patterns = ['static']
    >>> resolutions = {'static': {'1km': '1km'}, 'annual': {'1km': '1km'}}
    >>> _filter_resolution_files(files, resolution='1km', static=True, static_patterns=static_patterns, resolutions=resolutions)
    [Path('file_static_1km.tif')]
    """
    
    # Error checks
    if not files:
        raise ValueError("No files provided for filtering.")

    ## Dataset defines resolution metadata -- user must specify resolution
    if resolutions is not None and resolution is None:
        raise ValueError(
            "This dataset defines multiple resolutions. "
            "You must explicitly specify `resolution=...`."
        )

    ## User specified resolution, but dataset has no resolution metadata
    if resolutions is None and resolution is not None:
        raise ValueError(
            "A resolution was specified, but this dataset does not define "
            "any resolution metadata."
        )

    ## Dataset defines static patterns -- user must specify static flag
    if static_patterns and static is None:
        raise ValueError(
            "This dataset defines static files. "
            "You must explicitly specify `static=True` or `static=False`."
        )

    ## User requested static, but dataset does not support it
    if static and not static_patterns:
        raise ValueError(
            "static data was requested, but this dataset does not "
            "define any static patterns."
        )

    # Ensure static_patterns is a list
    static_patterns = static_patterns or []

    # Isolate static or annual files
    if static:
        # Get only static files
        files = [f for f in files if any(pattern in f.name for pattern in static_patterns)]
    else:
        # Get only annual files (Exclude static patterns)
        files = [f for f in files if not any(pattern in f.name for pattern in static_patterns)]

    # Filter by resolution if specified
    if resolution is not None:

        # Determine mode based on static flag
        mode = "static" if static else "annual"

        # Validate mode and resolution. Ensure resolutions are available for the specified mode.
        if mode not in resolutions:
            raise ValueError(f"Mode '{mode}' not found in resolution metadata.")

        # Ensure the specified resolution exists for the mode
        if resolution not in resolutions[mode]:
            raise ValueError(f"Resolution '{resolution}' not found for mode '{mode}' in resolution metadata."
                            f" Available resolutions: {list(resolutions[mode].keys())}")

        # Filter files by resolution token
        token = resolutions[mode][resolution]
        files = [f for f in files if token in f.name]

    return files

def measures_velocity(self, row, resolution = None, static = None, **kwargs):
    """
    Custom loader for MEaSUREs Velocity datasets.

    Loads MEaSUREs ice velocity NetCDF files, optionally filtering by resolution and static/annual mode,
    and adds a time dimension to annual files based on the year(s) encoded in the filename.

    Parameters
    ----------
    self : object
        The class instance.
    row : dict
        Dictionary containing dataset metadata, including file path, extension, and loading parameters.
    resolution : :class:`str`, optional
        The resolution to filter files by. Required if the dataset defines multiple resolutions.
    static : :class:`bool`, optional
        Flag indicating whether to load static files (True) or annual files (False).
    **kwargs
        Additional keyword arguments passed to ``xr.open_mfdataset``.

    Returns
    -------
    xr.Dataset
        Loaded MEaSUREs velocity data as an xarray Dataset. For annual files, a time dimension is added
        based on the year(s) in the filename (set to July 2nd of the middle year).

    Raises
    ------
    FileNotFoundError
        If no files matching the criteria are found.
    ValueError
        If required parameters are missing or filtering criteria are not met.

    Notes
    -----
        - For annual files, the time dimension is set to July 2nd of the middle year (or the single year if only one is present).
        - Assumes the filename is stored in ``ds.encoding['source']``.
        - Uses parallel loading via ``xr.open_mfdataset``.

    Examples
    --------
    >>> ds = loaders.measures_velocity(row, resolution='1km', static=False)
    """
    
    def _preprocessor(ds):
        """
        Preprocess each dataset to add a time dimension based on the year(s) in the filename.

        This function extracts the year from the filename, calculates the middle year,
        creates a time dimension set to July 2nd of that year, and expands the dataset
        to include this time dimension.

        Parameters
        ----------
        ds : :class:`xarray.Dataset`
            The input dataset to preprocess.

        Returns
        -------
        :class:`xarray.Dataset`
            The preprocessed dataset with an added time dimension.

        Notes
        -----
        - The time is set to July 2nd of the middle year (or the single year if only one is present).
        - Assumes the filename is stored in ``ds.encoding['source']``.
        """
        
        # Extract year(s) from filename
        start_year, end_year = _extract_year_range_from_filename(ds.encoding['source'])
        mid_year = start_year + (end_year - start_year) // 2
        
        # Create time dimension (July 2nd of the extracted year)
        time_da = xr.DataArray([datetime(year = mid_year, month = 7,  day = 2)], dims = 'time')

        # Expand dataset to include time dimension
        return ds.expand_dims(time = time_da)

    # Extract parameters from row
    path, ext, skip_lines, no_data, ignore_dirs, ignore_files, loader, resolutions, static_patterns = self._extract_row_params(row)

    # Normalise lists as needed
    ignore_dirs = self._normalise_list(ignore_dirs)
    ignore_files = self._normalise_list(ignore_files)
    static_patterns = self._normalise_list(static_patterns)
        
    # Find files
    files = self._recursive_find_files(path, ext, ignore_dirs = ignore_dirs, ignore_files = ignore_files)

    # Filter files based on static flag and resolution
    files = _filter_resolution_files(
        files,
        resolution = resolution,
        static = static,
        static_patterns = static_patterns,
        resolutions = resolutions
    )

    # If no files found, raise error
    if not files:
        raise FileNotFoundError(
            f"No files found with extension '{ext}' in {path}"
        )

    # Get kwargs (or set defaults)
    combine = kwargs.pop("combine", "by_coords")

    # Only use preprocessor for annual mode (no time dimension needed in static files)
    preprocess_func = _preprocessor if not static else None

    # Load NetCDF files with preprocessing (if necessary)
    output = xr.open_mfdataset(files,
                            combine = combine,
                            preprocess = preprocess_func,
                            parallel = True,
                            **kwargs)
    
    return output

def racmo(self, row, **kwargs):
    """
    Custom loader for RACMO datasets.

    Loads RACMO NetCDF files, preprocesses them by dropping unnecessary variables
    and setting coordinates, then combines them into a single xarray Dataset.

    Parameters
    ----------
    self : object
        The class instance.
    row : dict
        Dictionary containing dataset metadata, including file path, extension, 
        and loading parameters.
    **kwargs
        Additional keyword arguments passed to ``xr.open_mfdataset``.

    Returns
    -------
    xr.Dataset
        Loaded RACMO data as an xarray Dataset with preprocessed variables 
        and coordinates.

    Raises
    ------
    FileNotFoundError
        If no files matching the criteria are found.

    Warnings
    --------
    UserWarning
        For the 'racmo2.3p2_monthly_27km_1979-2022' dataset, a warning is issued
        noting that timesteps vary between variables/files, which may introduce
        all-NaN arrays for timesteps where a variable lacks data.

    Notes
    -----
        - Drops variables 'block1' and 'block2' if present.
        - Sets 'rlat' and 'rlon' as coordinates.
        - Uses parallel loading via ``xr.open_mfdataset``.
        - Default merge behavior: combine by coordinates, outer join, override compatibility.
    """

    def _preprocessor(ds):
        """
        Preprocess each dataset to drop unnecessary variables and set coordinates.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to preprocess.

        Returns
        -------
        xr.Dataset
            The preprocessed dataset with unnecessary variables dropped and 
            coordinates set for 'rlat' and 'rlon'.
        """
        
        DROP_VARS = {"block1", "block2"}

        ds = ds.set_coords(["rlat", "rlon"])

        return ds.drop_vars(DROP_VARS & set(ds.variables))

    # Extract parameters from row
    path, ext, skip_lines, no_data, ignore_dirs, ignore_files, loader, resolutions, static_patterns = self._extract_row_params(row)

    # Normalise lists as needed
    ignore_dirs = self._normalise_list(ignore_dirs)
    ignore_files = self._normalise_list(ignore_files)
    static_patterns = self._normalise_list(static_patterns)
        
    # Find files
    files = self._recursive_find_files(path, ext, ignore_dirs = ignore_dirs, ignore_files = ignore_files)

    # If no files found, raise error
    if not files:
        raise FileNotFoundError(
            f"No files found with extension '{ext}' in {path}"
        )

    # Get kwargs (or set defaults)
    combine = kwargs.pop("combine", "by_coords")
    join = kwargs.pop("join", "outer")
    compat = kwargs.pop("compat", "override")
    combine_attrs = kwargs.pop("combine_attrs", "drop_conflicts")

    # Load NetCDF files with preprocessing (if necessary)
    output = xr.open_mfdataset(files,
                            combine = combine,
                            join = join,
                            compat = compat,
                            combine_attrs = combine_attrs,
                            preprocess = _preprocessor,
                            parallel = True,
                            **kwargs)

    # Warn users about varying timesteps when loading racmo2.3p2_monthly_27km_1979-2022
    if row['dataset'] == 'racmo2.3p2_monthly_27km_1979-2022':
        warnings.warn("WARNING: Timesteps vary between various variables/files. All data files are loaded using\n"
        "xr.open_mfdataset() and the timestamps are merged. As a result, all-NaN arrays are introduced\n"
        "for timesteps where a given variable does not have data. Users should use caution when considering\n"
        "time-varying analysis and might consider 'trimming' data to isolate all non-NaN arrays for a given variable.")
        
    return output