import yaml
from pathlib import Path
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
    config_file : str or Path
        Path to the YAML configuration file.
    config : dict
        Parsed YAML configuration content.
    datasets : pd.DataFrame
        DataFrame listing all datasets, versions, and subdatasets with their metadata.
    """

    # Initialize DataPool with key fields
    def __init__(self, yaml_path):
        self.config_file = yaml_path
        self.config = self._load_yaml(self.config_file)
        self.datasets = self._list_datasets()

    # Load YAML configuration
    def _load_yaml(self, path):
        """
        Load a YAML file and return the parsed dict.

        Parameters
        ----------
        path : str or Path
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
        dataset_path : str or Path
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

    def _resolve_metadata(self, meta, subds_meta, version, key, default = None):
        """
        Resolve a metadata value for a dataset, supporting dataset-level and
        subdataset-level overrides, including optional per-version dictionaries.

        Resolution priority (highest → lowest):

            1. Subdataset-level value (``subds_meta[key]``)
            2. Dataset-level value (``meta[key]``)
            3. ``default``

        If a metadata value is a dictionary and contains a version key
        (e.g. ``{"v1": ..., "v2": ...}``), the value corresponding to the
        requested version is returned.

        Parameters
        ----------
        meta : dict
            Dataset-level metadata dictionary parsed from the YAML configuration.
        subds_meta : dict
            Subdataset-level metadata dictionary parsed from the YAML configuration.
        version : str
            Dataset version identifier (e.g. ``"v1"``, ``"v2"``). Used to resolve
            version-specific metadata entries when values are dictionaries.
        key : str
            Metadata key to resolve (e.g. ``"resolutions"``,
            ``"static_patterns"``, ``"extension"``).
        default : Any, optional
            Value returned if the key is not found at either dataset or
            subdataset level, or if the key exists but does not define the
            requested version. Default is ``None``.

        Returns
        -------
        resolved : Any
            The resolved metadata value. This may be a scalar (e.g. ``str``),
            a dictionary, or ``None`` if not found and no default is provided.

        Notes
        -----
        - Subdataset-level metadata always takes precedence over dataset-level
        metadata.
        - Dictionary-valued metadata may optionally be keyed by version.
        - Designed to support flexible YAML layouts where metadata may be
        specified at different hierarchy levels.
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
        """

        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]

    def _list_datasets(self):
        """
        Return a flattened DataFrame listing datasets, versions, subdatasets.
        
        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: dataset, display_name, tags, version, subdataset,
            path, full_path, extension, skip_lines, no_data_value.
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
        excluding any whose path contains one of the ignore_dirs substrings.

        Parameters
        ----------
        root : str or Path
            Root directory to search.
        extension : str
            File extension to search for (e.g., 'csv', 'tif').
        ignore_dirs : list of str, optional
            List of directory name substrings to ignore.
        
        Returns
        -------
        list of Path
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
        dataset : str
            Name of the dataset to load.
        version : str
            Version of the dataset to load.
        subdataset : str, optional
            Name of the subdataset to load (if applicable).
        ignore_dirs : list of str, optional
            List of directory name substrings to ignore when loading files.
        
        Returns
        -------
        pd.DataFrame, gpd.GeoDataFrame, or xr.Dataset
            Loaded dataset in appropriate format.
        
        Raises
        ------
        KeyError
            If no matching dataset entry is found.
        ValueError
            If multiple entries match the criteria.
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
        Accepts either a single string or a list of keywords.

        Parameters
        ----------
        keyword : str or list of str
            Keyword(s) to search for.
        
        Returns
        -------
        pd.DataFrame
            DataFrame of datasets matching any of the keywords.
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
        dataset : str
            Name of the dataset.
        
        Returns
        -------
        list of str
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
        dataset : str
            Name of the dataset.
        
        Returns
        -------
        str
            Latest version name.
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
        dataset : str
            Name of the dataset.
        version : str
            Version of the dataset.
        
        Returns
        -------
        list of str
            List of available subdataset names.
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
        dataset : str
            Name of the dataset.
        version : str, optional
            Version of the dataset.
        subdataset : str, optional
            Name of the subdataset.
        
        Returns
        -------
        list of str
            List of available resolutions.
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
        Used for filtering datasets based on arbitrary metadata.
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
        
        if "static" in used:
            if not row.static_patterns:
                raise TypeError(f"'static' is not applicable for dataset '{row.dataset}'."
                                " This dataset does not define any static patterns.")

    def help(self, dataset = None, version = None):
        """
        Describe available datasets and their supported options without loading data.

        Parameters
        ----------
        dataset : str, optional
            Dataset name to describe.

        version : str, optional
            Dataset version to describe.

        Returns
        -------
        None
            Prints information about datasets, versions, subdatasets, and capabilities.
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
