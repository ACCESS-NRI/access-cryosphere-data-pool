import yaml
from pathlib import Path
from importlib import resources
import pandas as pd
import warnings

from . import loaders

# Ensure UserWarnings are always shown
warnings.simplefilter('always', UserWarning)

class DataCatalog:
    """
    A catalog for managing and loading datasets with versioning and subdataset support.
    
    This class loads dataset configuration from a YAML file and provides methods to
    list, search, and load datasets with support for multiple versions and subdatasets.
    
    Attributes
    ----------
    config_file : :class:`pathlib.Path` or :class:`str`
        Path to the YAML configuration file.
    config : dict
        Parsed YAML configuration content.
    datasets : :class:`pandas.DataFrame`
        DataFrame listing all datasets, versions, and subdatasets with their metadata.
    
    Examples
    --------
    >>> catalog = DataCatalog()
    >>> catalog.help()
    >>> data = catalog.load_dataset('dataset_name', version='v1')
    """

    # Initialize DataPool with key fields
    def __init__(self, yaml_path = None):
        """
        Initialize a DataCatalog.

        This constructor initializes a DataCatalog from a given YAML file. By default, the 
        packaged ``config/datasets.yaml`` file is used.

        Parameters
        ----------
        yaml_path : :class:`pathlib.Path` or :class:`str`, optional
            Path to the YAML configuration file. If not provided, the default packaged 
            configuration file is used. Default is ``None``.

        Raises
        ------
        FileNotFoundError
            If the provided ``yaml_path`` does not exist.

        Notes
        -----
        The YAML file should contain a ``datasets`` key with dataset configurations.
        """

        # If yaml_path is not specified, use to default ../config/datasets.yaml
        if yaml_path is None:
            with resources.as_file(
                resources.files("datapool").joinpath("config/datasets.yaml")
            ) as p:
                self.config_file = Path(p)
        else:
            self.config_file = Path(yaml_path)
            if not self.config_file.exists():
                raise FileNotFoundError(
                    f"DataCatalog YAML file not found:"
                    f"  '{self.config_file}'\n\n"
                    f"Provide a valid path, or omit the argument to use the "
                    f"default packaged catalog."
                )
        self.config = self._load_yaml(self.config_file)
        self.datasets = self._list_datasets()

    # Load YAML configuration
    def _load_yaml(self, path):
        """
        Load a YAML file and return the parsed dictionary.

        Parameters
        ----------
        path : :class:`pathlib.Path` or :class:`str`
            Path to the YAML file.
        
        Returns
        -------
        dict
            Parsed YAML content as a dictionary.
        """

        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _infer_versions_from_directory(self, dataset_path):
        """
        Infer version names from subdirectories inside a dataset directory.

        Parameters
        ----------
        dataset_path : :class:`pathlib.Path` or :class:`str`
            Path to the dataset directory.

        Returns
        -------
        list of str
            Sorted list of version names (subdirectory names).
        """

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return []

        return sorted([
            p.name for p in dataset_path.iterdir()
            if p.is_dir()
        ])

    def _resolve_metadata(self, meta, subds_meta, version, key, default=None):
        """
        Resolve a metadata value for a dataset, supporting dataset-level and
        subdataset-level overrides, including optional per-version dictionaries.

        Resolution priority (highest to lowest):

        1. Subdataset-level value (``subds_meta[key]``)
        2. Dataset-level value (``meta[key]``)
        3. ``default``

        If a metadata value is a dictionary and contains a version key
        (e.g., ``{"v1": ..., "v2": ...}``), the value corresponding to the
        requested version is returned.

        Parameters
        ----------
        meta : dict
            Dataset-level metadata dictionary parsed from the YAML configuration.
        subds_meta : dict
            Subdataset-level metadata dictionary parsed from the YAML configuration.
        version : str
            Dataset version identifier (e.g., ``"v1"``, ``"v2"``). Used to resolve
            version-specific metadata entries when values are dictionaries.
        key : str
            Metadata key to resolve (e.g., ``"resolutions"``,
            ``"static_patterns"``, ``"extension"``).
        default : Any, optional
            Value returned if the key is not found at either dataset or
            subdataset level, or if the key exists but does not define the
            requested version. Default is ``None``.

        Returns
        -------
        Any
            The resolved metadata value. This may be a scalar (e.g., :class:`str`),
            a dictionary, or ``None`` if not found and no default is provided.

        Notes
        -----
        Subdataset-level metadata always takes precedence over dataset-level
        metadata. Dictionary-valued metadata may optionally be keyed by version.
        This method is designed to support flexible YAML layouts where metadata
        may be specified at different hierarchy levels.
        """

        # Ensure subds_meta is a dict
        subds_meta = subds_meta or {}

        # Check subdataset-level metadata first (highest priority)
        subds_value = subds_meta.get(key, None)

        if isinstance(subds_value, dict):
            # If the value is a dict, attempt to resolve by version.
            # If no version key is found, return the entire dict.
            return subds_value.get(version, subds_value)
        elif subds_value is not None:
            # Scalar value found at subdataset level
            return subds_value

        # Fallback to dataset-level metadata
        ds_value = meta.get(key, None)

        if isinstance(ds_value, dict):
            # Dataset-level dic may define version-specific entries
            return ds_value.get(version, default)
        elif ds_value is not None:
            # Scalar value found at dataset level
            return ds_value

        # Nothing found - return default
        return default

    def _normalise_list(self, value):
        """
        Ensure the value is a list. If it's a scalar, wrap it in a list.
        If it's None, return an empty list.

        Parameters
        ----------
        value : Any
            Input value to normalise.

        Returns
        -------
        list
            A list containing the input value, or an empty list if input is None.
        """

        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]

    def _list_datasets(self):
        """
        Return a flattened DataFrame listing datasets, versions, and subdatasets.
        
        This method constructs a DataFrame from the YAML configuration by iterating
        over all datasets and their versions, extracting metadata and constructing
        full file paths for each dataset/version combination.

        Parameters
        ----------
        None

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame with columns: ``dataset``, ``display_name``, ``description``,
            ``tags``, ``version``, ``subdataset``, ``path``, ``full_path``,
            ``extension``, ``skip_lines``, ``no_data_value``, ``ignore_dirs``,
            ``ignore_files``, ``loader``, ``resolutions``, ``static_patterns``.
        """

        # Initialize list to hold dataset records
        records = []

        # Iterate over datasets in config
        for dataset_name, meta in self.config["datasets"].items():
            
            # Extract common metadata
            base_path = Path(meta["path"])
            display_name = meta.get("display_name", dataset_name)
            description = meta.get("description", "")
            tags = meta.get("tags", [])

            # Get dataset versions from directory
            versions = self._infer_versions_from_directory(base_path)

            # VERSIONED DATASETS WITH SUBDATASETS
            if "subdatasets" in meta:
                
                # Iterate over versions (defined in subdatasets, not inferred from dirs)
                for version, subds_dict in meta["subdatasets"].items():

                    # Check version exists in directory
                    if version not in versions:
                        raise ValueError(f"Version '{version}' for dataset '{dataset_name}' not found in directory '{base_path}'. Available versions: {versions}")
                    
                    # Iterate over subdatasets
                    for subds_name, subds_meta in subds_dict.items():
                        
                        # Extract subdataset-specific metadata (or dataset-level fallback)
                        subpath = self._resolve_metadata(meta, subds_meta, version, "subpath")
                        extension = self._resolve_metadata(meta, subds_meta, version, "extension")
                        skip_lines = self._resolve_metadata(meta, subds_meta, version, "skip_lines", 0)
                        no_data_value = self._resolve_metadata(meta, subds_meta, version, "no_data_value", None)
                        ignore_dirs = self._resolve_metadata(meta, subds_meta, version, "ignore_dirs", None)
                        ignore_files = self._resolve_metadata(meta, subds_meta, version, "ignore_files", None)
                        loader = self._resolve_metadata(meta, subds_meta, version, "loader", "default")
                        resolutions = self._resolve_metadata(meta, subds_meta, version, "resolutions")
                        static_patterns = self._resolve_metadata(meta, subds_meta, version, "static_patterns", [])
                        
                        # Normalise lists as needed
                        ignore_dirs = self._normalise_list(ignore_dirs)
                        ignore_files = self._normalise_list(ignore_files)
                        static_patterns = self._normalise_list(static_patterns)

                        # Error checks
                        if not subpath:
                            raise ValueError(f"Subpath must be specified for subdataset '{subds_name}' in dataset '{dataset_name}', version '{version}'. This should be defined in the YAML config.")
                        if not extension:
                            raise ValueError(f"Extension must be specified for subdataset '{subds_name}' in dataset '{dataset_name}', version '{version}'. This should be defined in the YAML config.")                       
                        
                        # Construct full path to subdataset
                        full_path = base_path / version / subpath

                        # Append record
                        records.append({
                            "dataset": dataset_name,
                            "display_name": display_name,
                            "description": description,
                            "tags": tags,
                            "version": version,
                            "subdataset": subds_name,
                            "path": str(base_path),
                            "full_path": str(full_path),
                            "extension": extension,
                            "skip_lines": skip_lines,
                            "no_data_value": no_data_value,
                            "ignore_dirs": ignore_dirs,
                            "ignore_files": ignore_files,
                            "loader": loader,
                            "resolutions": resolutions,
                            "static_patterns": static_patterns,
                        })

            # VERSIONED DATASETS (no subdatasets)
            else:
                
                # Iterate over versions
                for version in versions:
                    
                    # Extract dataset-level metadata (version-specific if applicable)
                    extension = self._resolve_metadata(meta, None, version, "extension")
                    skip_lines = self._resolve_metadata(meta, None, version, "skip_lines", 0)
                    no_data_value = self._resolve_metadata(meta, None, version, "no_data_value", None)
                    ignore_dirs = self._resolve_metadata(meta, None, version, "ignore_dirs", None)
                    ignore_files = self._resolve_metadata(meta, None, version, "ignore_files", None)
                    loader = self._resolve_metadata(meta, None, version, "loader", "default")
                    resolutions = self._resolve_metadata(meta, None, version, "resolutions")
                    static_patterns = self._resolve_metadata(meta, None, version, "static_patterns", [])

                    # Normalise lists as needed
                    ignore_dirs = self._normalise_list(ignore_dirs)
                    ignore_files = self._normalise_list(ignore_files)
                    static_patterns = self._normalise_list(static_patterns)

                    # Error check
                    if not extension:
                        raise ValueError(f"Extension must be specified for dataset '{dataset_name}'. This should be defined in the YAML config.")  

                    # Construct path to version
                    version_path = base_path / version

                    # Append record
                    records.append({
                        "dataset": dataset_name,
                        "display_name": display_name,
                        "description": description,
                        "tags": tags,
                        "version": version,
                        "subdataset": None,
                        "path": str(base_path),
                        "full_path": str(version_path),
                        "extension": extension,
                        "skip_lines": skip_lines,
                        "no_data_value": no_data_value,
                        "ignore_dirs": ignore_dirs,
                        "ignore_files": ignore_files,
                        "loader": loader,
                        "resolutions": resolutions,
                        "static_patterns": static_patterns,
                    })

        return pd.DataFrame(records)


    def _recursive_find_files(self, root, extension, ignore_dirs = None, ignore_files = None):
        """
        Recursively find files with given extension under root directory,
        excluding any whose path contains one of the ignore_dirs or ignore_files substrings.

        Parameters
        ----------
        root : :class:`pathlib.Path` or class:`str`
            Root directory to search.
        extension : :class:`str`
            File extension to search for (e.g., ``'csv'``, ``'tif'``).
        ignore_dirs : :class:`list` of :class:`str`, optional
            List of directory name substrings to ignore. Default is ``None``.
        ignore_files : :class:`list` of :class:`str`, optional
            List of file name substrings to ignore. Default is ``None``.
        
        Returns
        -------
        :class:`list` of :class:`pathlib.Path`
            Sorted list of matching file paths.
        """
        
        # Convert root to Path object
        root = Path(root)
        
        # Ensure provided extension does not start with dot
        ext = extension.lstrip(".")

        # Recursively find all files with the given extension
        files = root.rglob(f"*.{ext}")

        # If ignore_dirs is provided, filter out matching directories
        if ignore_dirs is None:
            pass
        else:
            # Filter out any path containing one of the ignored directory names
            filtered = []
            for f in files:
                reject = any(bad in f.as_posix() for bad in ignore_dirs)
                if not reject:
                    filtered.append(f)
            files = filtered
        
        # If ignore_files is provided, filter out matching file names
        if ignore_files is None:
            pass
        else:
            filtered = []
            # Further filter out any files whose names contain one of the ignored file name substrings
            for f in files:
                reject = any(bad in f.as_posix() for bad in ignore_files)
                if not reject:
                    filtered.append(f)
            files = filtered

        return sorted(files)

    def _get_loader(self, name):
        """
        Retrieve a loader function by name from the loaders module.

        Parameters
        ----------
        name : :class:`str` or None
            Name of the loader function to retrieve from the loaders module. If ``None`, returns ``None``.

        Returns
        -------
        callable or None
            The loader function if found, otherwise ``None`` if name is ``None``.

        Raises
        ------
        ValueError
            If the loader function does not exist in the loaders module or is not callable.
        """

        if name is None:
            return None

        try:
            loader = getattr(loaders, name)
        except AttributeError:
            raise ValueError(f"Loader '{name}' not found in loaders module.")

        if not callable(loader):
            raise ValueError(f"Loader '{name}' exists but is not callable.")

        return loader

    def _extract_row_params(self, row):
        """
        Extract common parameters from a dataset row.

        Parameters
        ----------
        row : pandas.Series or dict
            Row containing dataset metadata and configuration.

        Returns
        -------
        tuple
            Tuple containing:
                path (Path): Full path to the dataset or subdataset.
                ext (str): File extension (without leading dot).
                skip_lines (int): Number of lines to skip when reading files.
                no_data (Any): Value representing missing data.
                ignore_dirs (list or None): List of directory substrings to ignore, or None.
                ignore_files (list or None): List of file substrings to ignore, or None.
                loader (str): Name of the loader function.
                resolutions (Any): Resolution metadata, if available.
                static_patterns (list): List of static file patterns, if available.
        """
    
        # Extract common parameters from row
        path = Path(row["full_path"])
        ext  = row["extension"].lstrip(".")
        skip_lines = row.get("skip_lines", 0)
        no_data = row.get("no_data_value", None)
        ignore_dirs = row.get("ignore_dirs", None)
        ignore_files = row.get("ignore_files", None)
        loader = row.get("loader", "default")
        resolutions = row.get("resolutions")
        static_patterns = row.get("static_patterns", [])
    
        return path, ext, skip_lines, no_data, ignore_dirs, ignore_files, loader, resolutions, static_patterns

    def _load_dataset_row(self, row, **kwargs):
        """
        Load all files for a dataset row, with optional directory filtering.

        Parameters
        ----------
        row : pandas.Series
            A row from the datasets DataFrame containing dataset metadata and configuration.
        **kwargs
            Additional keyword arguments to pass to the loader function.

        Returns
        -------
        pd.DataFrame, gpd.GeoDataFrame, or xr.Dataset
            Loaded dataset in the appropriate format determined by the loader function.

        Raises
        ------
        ValueError
            If no loader function is specified for the dataset.
        """

        loader = getattr(row, "loader", "default")
        loader_func = self._get_loader(loader)

        if loader_func is None:
            raise ValueError(f"No loader function specified for dataset '{row.dataset}'.")

        return loader_func(self, row, **kwargs)

    def load_dataset(self, dataset, version = None, subdataset = None, **kwargs):
        """
        Load any dataset by name/version/subdataset with optional directory filtering.

        Parameters
        ----------
        dataset : :class:`str`
            Name of the dataset to load.
        version : :class:`str`, optional
            Version of the dataset to load. If not provided, the latest version is used.
            Default is ``None``.
        subdataset : :class:`str`, optional
            Name of the subdataset to load (if applicable). Default is ``None``.
        **kwargs
            Additional keyword arguments to pass to the loader function. Common options include:
            
            - ``resolution`` : :class:`str`, optional
                Resolution to load (if supported by the dataset).
            - ``static`` : :class:`bool`, optional
                Whether to load static files (if supported by the dataset).
        
        Returns
        -------
        :class:`pandas.DataFrame`, :class:`geopandas.GeoDataFrame`, or :class:`xarray.Dataset`
            Loaded dataset in the appropriate format determined by the loader function.
        
        Raises
        ------
        KeyError
            If no matching dataset entry is found.
        TypeError
            If ``subdataset`` is specified for a dataset that does not define subdatasets.
        ValueError
            If multiple entries match the criteria or if multiple subdatasets exist 
            and none is specified.

        Examples
        --------
        Load the latest version of a dataset:
        
        >>> data = catalog.load_dataset('dataset_name')
        
        Load a specific version:
        
        >>> data = catalog.load_dataset('dataset_name', version='v1')
        
        Load a specific subdataset:
        
        >>> data = catalog.load_dataset('dataset_name', version='v1', subdataset='sub1')
        """

        # Get dataset Dataframe
        df = self.datasets

        # If version not specified, get latest version
        if version is None:
            version = self._get_latest_version(dataset)

        # Filter by dataset and version
        subset = df[(df.dataset == dataset) &
                    (df.version == version)]

        # Raise error if no matching entry found
        if subset.empty:
            raise KeyError(f"No dataset entry found for:\n"
                            f"'dataset': {dataset}\n"
                            f"'version': {version}")


        # If subdataset specified, filter by subdataset next
        if subdataset is not None:

            # Check if subdataset exists
            if subset.subdataset.isna().all():
                raise TypeError(f"'subdataset' is not applicable for dataset '{dataset}'."
                                " This dataset does not define any subdatasets.")
            
            # Get available subdatasets
            available_subdatasets = self.available_subdatasets(dataset, version)

            # Raise error if specified subdataset not found
            if subdataset not in available_subdatasets:
                raise KeyError(f"Subdataset '{subdataset}' not found for dataset '{dataset}', version '{version}'.\n"
                               f"Available subdatasets: {available_subdatasets}")

            # Filter by subdataset
            subset = subset[subset.subdataset == subdataset]
            
        # Raise error if multiple entries found (should be unique).
        if len(subset) > 1:
            # If multiple subdatasets exist, prompt user to specify one
            if subset.subdataset.unique().size > 1:
                raise ValueError(f"Multiple subdatasets found for dataset '{dataset}', version '{version}'.\n"
                                 f"Available subdatasets: {self.available_subdatasets(dataset, version)}\n"
                                 "Please specify a subdataset to load.")
            else:
                # Generic error for multiple matches
                raise ValueError("Multiple entries matched; dataset table should have unique rows. Refine your query.")

        # Load dataset from the single matching row
        row = subset.iloc[0]

        # Check any additional keywords against the row
        self._check_keywords(row, kwargs)
        
        # Load dataset using the appropriate loader
        data = self._load_dataset_row(row, **kwargs)
        
        return data

    def search(self, keyword):
        """
        Search datasets by keyword in dataset name, display name, or tags.

        Accepts either a single string or a list of keywords and returns a DataFrame
        of datasets matching any of the provided keywords.

        Parameters
        ----------
        keyword : :class:`str` or :class:`list` of :class:`str`
            Keyword(s) to search for in dataset name, display name, or tags.

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame of datasets matching any of the keywords.

        Examples
        --------
        Search for a single keyword:

        >>> results = catalog.search('temperature')

        Search for multiple keywords:

        >>> results = catalog.search(['temperature', 'precipitation'])
        """
        
        # Ensure keywords is a list
        keywords = keyword if isinstance(keyword, list) else [keyword]
        
        # Initialize boolean mask
        mask = pd.Series([False] * len(self.datasets))
        
        # Update mask for each keyword
        for kw in keywords:
            keyword_lower = kw.lower()
            mask |= (
                self.datasets["dataset"].str.lower().str.contains(keyword_lower) |
                self.datasets["display_name"].str.lower().str.contains(keyword_lower) |
                self.datasets.get("tags", pd.Series([])).apply(
                    lambda tags: any(keyword_lower in tag.lower() for tag in tags)
                )
            )
        
        # Return filtered DataFrame
        return self.datasets[mask]

    def available_versions(self, dataset):
        """
        Show available versions for a given dataset.

        Parameters
        ----------
        dataset : :class:`str`
            Name of the dataset.
        
        Returns
        -------
        :class:`list` of :class:`str`
            List of available version names.
        """

        # Get dataset Dataframe
        df = self.datasets

        # Get versions
        versions = df[df.dataset == dataset]["version"].unique().tolist()

        if versions == []:
            raise ValueError(f"No versions found for dataset '{dataset}'.")

        return versions

    def _get_latest_version(self, dataset):
        """
        Get the latest version for a given dataset based on alphanumeric sorting.

        Parameters
        ----------
        dataset : :class:`str`
            Name of the dataset.
        
        Returns
        -------
        :class:`str`
            Latest version name.
        
        Raises
        ------
        ValueError
            If no versions are found for the dataset.
        """

        # Get available versions
        versions = self.available_versions(dataset)
        
        # Raise error if no versions found
        if not versions:
            raise ValueError(f"No versions found for dataset '{dataset}'. All datasets must have at least one version.")

        # Return the latest version (sorted alphanumerically)
        return sorted(versions)[-1]

    def available_subdatasets(self, dataset, version = None):
        """
        Show available subdatasets for a given dataset and version.

        Parameters
        ----------
        dataset : :class:`str`
            Name of the dataset.
        version : :class:`str`, optional
            Version of the dataset. If not provided, the latest version is used.
            Default is ``None``.
        
        Returns
        -------
        :class:`list` of :class:`str`
            List of available subdataset names, or ``None`` if no subdatasets
            are defined for the dataset.
        
        Raises
        ------
        ValueError
            If no versions are found for the dataset.
        
        Warnings
        --------
        UserWarning
            If no subdatasets are defined for the specified dataset and version.
        """
        
        # Get dataset Dataframe
        df = self.datasets

        # If version not specified, get latest version
        if version is None:
            version = self._get_latest_version(dataset)

        # Get subdatasets
        subdatasets = df[
            (df.dataset == dataset) &
            (df.version == version) &
            (df.subdataset.notnull())
        ]["subdataset"].unique().tolist()

        if subdatasets == []:
            warnings.warn('No subdatasets defined for this dataset.')
            return

        return subdatasets

    def available_resolutions(self, dataset, version = None, subdataset = None):
        """
        Show available resolutions for a given dataset/version/subdataset.

        Parameters
        ----------
        dataset : :class:`str`
            Name of the dataset.
        version : :class:`str`, optional
            Version of the dataset. If not provided, the latest version is used.
            Default is ``None``.
        subdataset : :class:`str`, optional
            Name of the subdataset. Default is ``None``.
        
        Returns
        -------
        :class:`list` of :class:`str`
            List of available resolutions, or ``None`` if no resolutions are defined.
        
        Raises
        ------
        KeyError
            If no matching dataset entry is found.
        
        Warnings
        --------
        UserWarning
            If no resolutions are defined for the specified dataset/version/subdataset.
        """

        # Get dataset Dataframe
        df = self.datasets

        # If version not specified, get latest version
        if version is None:
            version = self._get_latest_version(dataset)

        # Filter by dataset/version
        subset = df[
            (df.dataset == dataset) &
            (df.version == version)
        ]

        # Get subdataset if specified
        if subdataset is not None:
            subset = subset[subset.subdataset == subdataset]

        # Raise error if no matching entry found
        if subset.empty:
            raise KeyError(f"No dataset entry found for: {dataset}, {version} ({subdataset})")

        # Get resolutions from the single matching row
        row = subset.iloc[0]
        resolutions = row.get("resolutions", None)
        
        # Warn if no resolutions defined
        if resolutions is None:
            warnings.warn('No resolutions defined for this dataset.')
            return
        
        return resolutions

    def _check_keywords(self, row, kwargs):
        """
        Check if all provided keywords match the dataset row.
        
        Validates that keywords passed to dataset loading operations are supported
        by the specified dataset. Raises appropriate errors if unsupported keywords
        are used (e.g., requesting a resolution for a dataset that doesn't define
        resolutions).

        Parameters
        ----------
        row : pandas.Series
            A row from the datasets DataFrame containing dataset metadata.
        kwargs : dict
            Keyword arguments to validate against the dataset row metadata.
            Recognized keywords: ``resolution``, ``static``, ``subdataset``.

        Raises
        ------
        TypeError
            If ``resolution`` is specified but the dataset does not define resolutions.
        TypeError
            If ``static`` is specified but the dataset does not define static patterns.
        
        Notes
        -----
        This method is designed to provide early validation of user-supplied keywords
        before attempting to load data, preventing confusing errors downstream.
        """

        CATALOG_KEYWORDS = {
            "resolution",
            "static",
            "subdataset"
        }

        used = set(kwargs.keys()) & CATALOG_KEYWORDS

        # Check resolution
        if "resolution" in used:
            if row.resolutions is None:
                raise TypeError(f"'resolution' is not applicable for dataset '{row.dataset}'."
                                " This dataset does not define any resolution metadata.")
        
        # Check static
        if "static" in used:
            if not row.static_patterns:
                raise TypeError(f"'static' is not applicable for dataset '{row.dataset}'."
                                " This dataset does not define any static patterns.")

    def help(self, dataset = None, version = None):
        """
        Describe available datasets and their supported options without loading data.

        This method provides information about available datasets, versions, subdatasets,
        and supported loading options. When called without arguments, it lists all available
        datasets. When a dataset is specified, it lists available versions. When both
        dataset and version are specified, it provides detailed information about the
        dataset/version combination including available subdatasets and supported keywords.

        Parameters
        ----------
        dataset : :class:`str`, optional
            Name of the dataset to describe. If not provided, lists all available datasets.
            Default is ``None``.
        version : :class:`str`, optional
            Version of the dataset to describe. If not provided, lists available versions
            for the specified dataset. Default is ``None``.

        Returns
        -------
        None
            This method prints information to stdout about datasets, versions, subdatasets,
            and supported catalog keywords. No value is returned.

        Raises
        ------
        KeyError
            If the specified dataset does not exist.
        KeyError
            If the specified version does not exist for the given dataset.

        Examples
        --------
        List all available datasets:

        >>> catalog.help()

        List versions for a specific dataset:

        >>> catalog.help(dataset='dataset_name')

        Show detailed information for a dataset version:

        >>> catalog.help(dataset='dataset_name', version='v1')

        Notes
        -----
        The method provides hints for further exploration and example usage based on the
        available metadata for the dataset/version combination.
        """

        # Get dataset Dataframe
        df = self.datasets

        # 1. If no dataset specified, simply list datasets
        # ------------------------------------------------------------------
        if dataset is None:
            datasets = sorted(df.dataset.unique())
            print("Available datasets:")
            for d in datasets:
                print(f"  - {d}")
            return

        # 2. Dataset-level help - list versions
        # ------------------------------------------------------------------
        subset = df[df.dataset == dataset]

        if subset.empty:
            raise KeyError(f"Unknown dataset '{dataset}'")

        print(f"Dataset: {dataset}")

        versions = sorted(subset.version.unique())
        print("\nAvailable versions:")
        for v in versions:
            print(f"  - {v}")

        # If no version specified, stop here
        if version is None:
            print("\nTip:")
            print("  Use catalog.help(dataset=..., version=...) for more details.")
            return

        # 3. Version-level help
        # ------------------------------------------------------------------
        subset = subset[subset.version == version]

        if subset.empty:
            raise KeyError(
                f"Version '{version}' not found for dataset '{dataset}'. "
                f"Available versions: {versions}"
            )

        print(f"\nVersion: {version}")

        # 4. Subdatasets
        # ------------------------------------------------------------------
        # If subdatasets exist, list them
        if not subset.subdataset.isna().all():
            subdatasets = sorted(subset.subdataset.dropna().unique())
            print("\nAvailable subdatasets:")
            for s in subdatasets:
                print(f"  - {s}")
        else:
            print("\nAvailable subdatasets: none")

        # 5. Capabilities (based on row metadata)
        # ------------------------------------------------------------------
        row = subset.iloc[0]

        print("\nSupported catalog keywords:")
        print(f"  - subdataset : {'yes' if not subset.subdataset.isna().all() else 'no'}")
        print(f"  - resolution : {'yes' if row.resolutions is not None else 'no'}")
        print(f"  - static  : {'yes' if bool(row.static_patterns) else 'no'}")

        # 6. Example usage
        # ------------------------------------------------------------------
        print("\nExample usage:")

        example = f"catalog.load_dataset('{dataset}', version = '{version}'"

        if not subset.subdataset.isna().all():
            example += ", subdataset = '...'"

        if row.resolutions is not None:
            example += ", resolution = '...'"

        if row.static_patterns:
            example += ", static = True"

        example += ")"

        print(f"  {example}")
