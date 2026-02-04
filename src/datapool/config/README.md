## ACCESS Cryosphere Datapool (CCD)

CCD Python tools are built around a central `datasets.yaml` file. This file contains all the necessary information to load any given dataset using the `load_dataset` function. The general structure of the YAML is as follows:

```
datasets:
    dataset_name:
        # Dataset-level metadata fields
        display_name: <COMMON DATASET NAME>
        path: <PATH TO DATASET DIRECTORY>
        description: <GENERAL DATASET DESCRIPTION>
        tags:
            - TAG 1
            - TAG 2
            ...
            - TAG N
        loader: <NAME OF CUSTOM LOADER REQUIRED TO LOAD DATASET>
        extension: <EXTENSION FOR FILE(S) TO LOAD>
            v1: <VERSION-SPECIFIC EXTENSION FOR FILE(S) TO LOAD>
            v2: <VERSION-SPECIFIC EXTENSION FOR FILE(S) TO LOAD>
            ...
            vN: <VERSION-SPECIFIC EXTENSION FOR FILE(S) TO LOAD>
                
        # Version-specific metadata fields
        ignore_dirs:
            v1:
                - <IGNORE DIRECTORY IN path>
                - <IGNORE DIRECTORY IN path>
                - <IGNORE DIRECTORY IN path>
        ignore_files:
            v1: 
                - <IGNORE FILE IN path>
                - <IGNORE FILE IN path>
                ...
                - <IGNORE FILE IN path>
        resolutions:
            v1: 
                annual:
                    resolution_1: <PATTERN IN FILENAME(s)>
                    resolution_2: <PATTERN IN FILENAME(s)>
                    ...
                    resolution_n: <PATTERN IN FILENAME(S)>
                composite:
                    resolution_1: <PATTERN IN FILENAME(S)>
                    resolution_2: <PATTERN IN FILENAME(S)>
                    ...
                    resolution_n: <PATTERN IN FILENAME(S)>
        composite_patterns:
            v1:
                - <PATTERN IN FILENAME(S)>
                - <PATTERN IN FILENAME(S)>
                ...
                - <PATTERN IN FILENAME(S)>
        subdatasets:
            v1:
                subdataset_name:
                    subpath: <PATH TO SUBDATASET DIRECTORY>
                    extension: <SUBSET-SPECIFIC EXTENSION FOR FILE(S) TO LOAD>
                    no_data_value: <SUBSET-SPECIFIC VALUE TO USE FOR NO-DATA>
                    skip_lines: <SUBSET-SPECIFIC VALUE TO USE FOR skip_lines WHEN READING CSV FILES>
                    ignore_dirs: <SUBSET-SPECIFIC DIRECTORIES TO IGRNORE WITHIN subpath>
                    
```
**NOTE: any subdataset-specific definitions over-write dataset-level definitions. For example, if `reoslution` is defined at the dataset-level, this resolution is applied to all subdatasets, unless `resolution` is explicitly stated within a specific subdataset.**

Version information is inferred from the directory structure. All datasets must have a `v1` sub-directory, at a minimum. For example, all `elevation_geometry` datasets included in the datapool are structured as follows:
```
.
├── bedmap
│   ├── v1
│   ├── v2
│   └── v3
├── measures_bedmachine_antarctica
│   ├── v1
│   ├── v2
│   └── v3
├── measures_its_live_antarctic_annual_240m_ice_sheet_extent_masks_1997_2021
│   └── v1
├── measures_its_live_antarctic_grounded_ice_sheet_elevation_change
│   └── v1
└── measures_its_live_antarctic_quarterly_1920m_ice_shelf_height_change_and_basal_melt_rates_1992_2017
    └── v1
```

A few example datasets are included below:

- Single-file, static-in-time, datasets:
```
  measures_bedmachine_antarctica:
    display_name: BedMachine Antarctica
    path: /g/data/av17/access-nri/cryosphere-data-pool/elevation_geometry/measures_bedmachine_antarctica
    description: High-resolution bed topography and ice thickness data for Antarctica.
    tags:
      - antarctica
      - bed topography
      - ice thickness
      - surface elevation
      - bed uncertainty
      - ice mask
    extension: nc
```

- Subdatasets, with different formats and versions across versions, and different directories to be ignored in each subdataset:
```
  bedmap:
    display_name: Bedmap
    path: /g/data/av17/access-nri/cryosphere-data-pool/elevation_geometry/bedmap
    description: Gridded, geospatial, and point datasets of Antarctic ice thickness, surface elevation, and bed elevation.
    tags:
      - antarctica
      - bed topography
      - ice thickness
      - surface elevation
    subdatasets:
      # Define version-level subdatasets and subdataset-specific fields (e.g. extensions, skip_lines, no_data_value)
      v1:
        geospatial:
          subpath: geospatial_data
          extension: gpkg
          no_data_value: -9999
        points:
          subpath: point_data
          extension: csv
          skip_lines: 18
          no_data_value: -9999
        gridded:
          subpath: gridded_data
          extension: tif
      v2:
        geospatial:
          subpath: geospatial_data
          extension: gpkg
          no_data_value: -9999
          ignore_dirs:
            - shapeLines
        points:
          subpath: point_data
          extension: csv
          skip_lines: 18
          no_data_value: -9999
        gridded:
          subpath: gridded_data
          extension: tif
          ignore_dirs:
            - resources
      v3:
        geospatial:
          subpath: geospatial_data
          extension: gpkg
          no_data_value: -9999
          ignore_dirs:
            - shapeLines
        points:
          subpath: point_data
          extension: csv
          skip_lines: 18
          no_data_value: -9999
        gridded:
          subpath: gridded_data
          extension: nc
          ignore_dirs:
            - bm3_streamlines_pt
```
- Different extensions at the version-level (no subdatasets):
```
  measures_antarctic_grounding_line:
    display_name: Antarctic Grounding Line from Differential Satellite Radar Interferometry
    path: /g/data/av17/access-nri/cryosphere-data-pool/geospatial/measures_antarctic_grounding_line_from_differential_satellite_radar_interferometry
    description: Grounding line positions for Antarctica derived from differential satellite radar interferometry.
    tags:
      - antarctica
      - grounding line
    # Define version-specific extensions for the dataset
    extension:
      v1: csv
      v2: shp
```
- Custom loader:
```
 measures_insar_based_ice_velocity_maps_of_central_antarctica:
    path: /g/data/av17/access-nri/cryosphere-data-pool/ice_velocity/measures_insar_based_ice_velocity_maps_of_central_antarctica
    display_name: InSAR-based Ice Velocity Maps of Central Antarctica
    description: Ice velocity maps for central Antarctica derived from Interferometric Synthetic Aperture Radar (InSAR) data.
    tags:
      - antarctica
      - ice velocity
      - insar
    extension: nc
    # Define a custom loader for this dataset
    loader: measures_velocity
```
- Different resolutions for annual/composite files available for different datasets:
```
  measures_its_live_regional_glacier_and_ice_sheet_surface_velocities:
    path: /g/data/av17/access-nri/cryosphere-data-pool/ice_velocity/measures_its_live_regional_glacier_and_ice_sheet_surface_velocities
    display_name: ITS_LIVE Regional Glacier and Ice Sheet Surface Velocities
    description: ITS_LIVE regional glacier and ice sheet surface velocities for Antarctica and Greenland.
    tags:
      - antarctica
      - greenland
      - ice velocity
    extension: nc
    loader: measures_velocity

    # Version-sepcific, dataset-level resolutions
    resolutions:
      v1: 
        annual:
          240m: G0240
        composite:
          120m: G0120
          240m: G0240

    # Version-specific file patterns for composite files
    composite_patterns:
      v1: "_0000"
      v2: "_2014-2022_"
      
    subdatasets:
      v1:
        ANT:
          subpath: ANT
        ALA:
          subpath: ALA
          # Overwrite dataset-level resolutions for this subdataset
          resolutions:
            annual:
              120m: G0120
              240m: G0240
            composite:
              120m: G0120
              240m: G0240
        CAN:
          subpath: CAN
        GRE:
          subpath: GRE
        HMA:
          subpath: HMA
        ICE: 
          subpath: ICE
        PAT: 
          subpath: PAT
        SRA:
          subpath: SRA
          # Overwrite dataset-level resolutions for this subdataset
          resolutions:
            annual:
              120m: G0120
              240m: G0240
            composite:
              120m: G0120
              240m: G0240
      v2:
        RGI01A:
          subpath: RGI01A
        RGI02A:
          subpath: RGI02A
        RGI03A: 
          subpath: RGI03A
        RGI04A:
          subpath: RGI04A
        RGI05A:
          subpath: RGI05A
        RGI06A:
          subpath: RGI06A
        RGI07A:
          subpath: RGI07A
        RGI08A:
          subpath: RGI08A
        RGI09A:
          subpath: RGI09A
        RGI10A:
          subpath: RGI10A
        RGI11A:
          subpath: RGI11A
        RGI12A:
          subpath: RGI12A
        RGI13A:
          subpath: RGI13A
        RGI14A:
          subpath: RGI14A
        RGI15A:
          subpath: RGI15A
        RGI16A:
          subpath: RGI16A
        RGI17A:
          subpath: RGI17A
        RGI18A:
          subpath: RGI18A
        RGI19A:
          subpath: RGI19A
```