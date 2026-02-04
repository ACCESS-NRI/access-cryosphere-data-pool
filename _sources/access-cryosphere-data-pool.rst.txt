.. _api:

access-cryosphere-data-pool API Reference
===================
.. automodule:: access-cryosphere-data-pool
.. currentmodule:: access-cryosphere-data-pool

.. _data-catalog:

DataCatalog
-------------------
``access-cryosphere-data-pool`` is developed around a cenral ``DataCatalog()`` class.

.. autosummary::
    :toctree: api

    datapool.catalog.DataCatalog

.. _data-discovery:

Data discovery functions
-------------------
There are various functions to explore the DataCatalog:

.. autosummary::
   :toctree: api
   :recursive:

    datapool.catalog.DataCatalog.available_resolutions
    datapool.catalog.DataCatalog.available_subdatasets
    datapool.catalog.DataCatalog.available_versions
    datapool.catalog.DataCatalog.help
    datapool.catalog.DataCatalog.search

.. _loading_data:

Loading data
-------------------
Datasets are loaded using the ``load_dataset()`` function.

.. autosummary::
   :toctree: api
   :recursive:

   datapool.catalog.DataCatalog.load_dataset

.. _data-loaders:

Data loaders
-------------------
Under-the-hood, datasets are loaded using custom ``loaders``.

.. autosummary::
   :toctree: api
   :recursive:

   datapool.loaders