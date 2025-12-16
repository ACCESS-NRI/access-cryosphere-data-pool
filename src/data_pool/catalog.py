import yaml
from pathlib import Path
import pandas as pd
import xarray as xr
import geopandas as gpd 
import numpy as np
import rioxarray as rxr

from . import loaders

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

    def _get_metadata(self, meta, version, key, default = None):
        """
        Resolve metadata value with optional per-version overrides.
        
        meta[key] can be a scalar, or a dict keyed by version.
        """

        value = meta.get(key, default)

        if isinstance(value, dict):
            return value.get(version, default)
        else:
            return value

    def _resolve_metadata(self, meta, subds_meta, version, key, default = None):
        """
        Resolve metadata value with subdataset and per-version overrides.
        
        Priority:
        1. subds_meta[key] (can be scalar or dict by version)
        2. meta[key] (can be scalar or dict by version)
        3. default
        """

        # First check subdataset-level metadata
        subds_value = self._get_metadata(subds_meta, version, key, None)
        if subds_value is not None:
            return subds_value
        
        # Fallback to dataset-level metadata
        ds_value = self._get_metadata(meta, version, key, None)
        if ds_value is not None:
            return ds_value

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
                        loader = self._resolve_metadata(meta, subds_meta, version, "loader", "default")
                        
                        # Normalise lists as needed
                        ignore_dirs = self._normalise_list(ignore_dirs)

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
                            "loader": loader,
                        })

            # VERSIONED DATASETS (no subdatasets)
            else:
                
                # Iterate over versions
                for version in versions:
                    
                    # Extract dataset-level metadata (version-specific if applicable)
                    extension = self._get_metadata(meta, version, "extension")
                    skip_lines = self._get_metadata(meta, version, "skip_lines", 0)
                    no_data_value = self._get_metadata(meta, version, "no_data_value", None)
                    ignore_dirs = self._get_metadata(meta, version, "ignore_dirs", None)
                    loader = self._get_metadata(meta, version, "loader", "default")

                    # Normalise lists as needed
                    ignore_dirs = self._normalise_list(ignore_dirs)

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
                        "loader": loader,
                    })

        return pd.DataFrame(records)


    def _recursive_find_files(self, root, extension, ignore_dirs = None):
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

        # Default ignore_dirs to empty list if None
        if ignore_dirs is None:
            ignore_dirs = []

        # Recursively find all files with the given extension
        files = root.rglob(f"*.{ext}")

        # Filter out any path containing one of the ignored directory names
        filtered = []
        for f in files:
            reject = any(bad in f.as_posix() for bad in ignore_dirs)
            if not reject:
                filtered.append(f)

        return sorted(filtered)

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
    
        return path, ext, skip_lines, no_data, ignore_dirs

    def _load_dataset_row(self, row, **kwargs):
        """
        Load all files for a dataset row, with optional directory filtering.
        """

        loader = getattr(row, "loader", "default")
        loader_func = self._get_loader(loader)

        if loader_func is None:
            raise ValueError(f"No loader function specified for dataset '{row.dataset}'.")

        return loader_func(self, row, **kwargs)

    def load_dataset(self, dataset, version, subdataset = None, **kwargs):
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

        # If no subdataset specified, filter by dataset and version only
        if subdataset is None:
            subset = df[(df.dataset == dataset) &
                        (df.version == version)]

            # Raise error if no matching entry found
            if subset.empty:
                raise KeyError(f"No dataset entry found for: {dataset}, {version}")
        
        # If subdataset specified, filter by all three criteria
        else:
            subset = df[
                (df.dataset == dataset) &
                (df.version == version) &
                (df.subdataset == subdataset)
            ]
            
            # Raise error if no matching entry found
            if subset.empty:
                raise KeyError(f"No dataset entry found for: {dataset}, {version} ({subdataset})")

        # Raise error if multiple entries found (should be unique).
        if len(subset) > 1:
            raise ValueError("Multiple entries matched; dataset table should have unique rows.")

        # Load dataset from the single matching row
        row = subset.iloc[0]
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