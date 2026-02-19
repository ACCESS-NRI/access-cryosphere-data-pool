Cryosphere Community Datapool (CCD) - Features and Capabilities Overview
========

.. warning::
   This is an experimental release. Documentation remains a work in progress. Some sections may be incomplete or under development.


The Cryosphere Community Datapool (CCD) is built around a powerful dataset management system with the following core capabilities:

#### Key Features

1. **Unified Dataset Catalog**
CCD provides a centralized catalog of cryosphere-focused datasets maintained in a flexible YAML configuration. The catalog currently includes elevation and geometry datasets (e.g., BedMachine Antarctica, Bedmap), geospatial boundaries, basal forcing models, and ice velocity measurements from various sources including MEaSUREs, ITS_LIVE, and InSAR observations.

2. **Dataset Discovery and Search**
Users can easily explore available datasets through an interactive catalog interface. The search functionality allows users to find datasets by keywords across multiple metadata fields including dataset names, display names, and descriptive tags (e.g., "antarctica", "ice velocity", "ice thickness").

3. **Version and Subdataset Management**
CCD supports datasets with multiple versions and hierarchical subdatasets. For example, the Bedmap dataset includes three versions (v1, v2, v3), each with multiple subdatasets (geospatial, points, and gridded data). This structure enables researchers to access specific data variants and historical version comparisons.

4. **Flexible Data Loaders**
CCD handles multiple data formats transparently, including CSV, GeoPackage, Shapefile, GeoTIFF, and NetCDF formats. Users specify the format in the catalog configuration, and the appropriate loader automatically handles file discovery and data loading, returning Pandas DataFrames, GeoPandas GeoDataFrames, or Xarray Datasets as appropriate.

5. **Resolution and Parameter Filtering**
For datasets with multiple resolutions or static/annual variants, users can specify desired resolution parameters during loading. CCD automatically filters and loads the correct data subset, supporting complex dataset structures with multiple resolution options.

#### User Workflow

Users interact with CCD in three simple steps:

.. code-block:: python
# 1. Initialize the catalog
catalog = DataCatalog()

# 2. Browse and search for datasets
results = catalog.search('ice velocity')
catalog  # View interactive catalog

# 3. Load data
data = catalog.load_dataset('dataset_name', version='v1', subdataset='sub1')
"""

#### Use Cases

- **Model Initialisation:** Quickly access ice sheet model input datasets (topography, velocities, boundaries)
- **Scientific Analysis:** Discover and load multi-source cryosphere datasets for comparative studies
- **Data Discovery:** Explore available datasets and their properties through keyword search
- **Version Control:** Access specific dataset versions for reproducibility and validation studies
