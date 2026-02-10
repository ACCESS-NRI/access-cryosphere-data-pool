.. _api:

ACCESS Cryosphere Community Datapool API Reference
=========================================
.. automodule:: datapool
.. currentmodule:: datapool

.. _data-catalog:

DataCatalog
-----------
The ACCESS Cryosphere Community Datapool is developed around a cenral ``DataCatalog`` class.

.. autosummary::
    :toctree: api

    datapool.catalog.DataCatalog

.. _data-discovery:

Data discovery
--------------
There are various methods to explore the DataCatalog:

.. autosummary::
   :toctree: api
   :no-index:

    datapool.catalog.DataCatalog.available_resolutions
    datapool.catalog.DataCatalog.available_subdatasets
    datapool.catalog.DataCatalog.available_versions
    datapool.catalog.DataCatalog.help
    datapool.catalog.DataCatalog.search

.. _loading_data:

Loading data
------------
Datasets are loaded using the ``load_dataset`` method.

.. autosummary::
   :toctree: api
   :no-index:

   datapool.catalog.DataCatalog.load_dataset

Under-the-hood, datasets are loaded using custom ``loaders``. 

.. autosummary::
   :toctree: api
   :recursive:

   datapool.loaders