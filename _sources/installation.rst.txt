Installation
============

.. warning::
   This is an experimental release. Documentation remains a work in progress. Some sections may be incomplete or under development.

access-cryosphere-data-pool can be installed using `pip` or `conda`, or by cloning the repository from GitHub. Below are instructions for each method.

Installing via pip
-----------------
You can install access-cryosphere-data-pool using `pip`, the Python package manager.
See the official release on PyPI: https://pypi.org/project/access-cryosphere-data-pool/

Run the following command in your terminal:

.. code-block:: bash

   pip install access-cryosphere-data-pool

Installing via conda
-------------------
If you prefer using `conda`, you can install access-cryosphere-data-pool from the `accessnri` channel.  
https://anaconda.org/channels/accessnri/packages/access-cryosphere-data-pool/overview

.. code-block:: bash

   conda install accessnri::access-cryosphere-data-pool

Installing from GitHub (Development Version)
-------------------------------------------
You can also install the latest development version directly from the GitHub repository:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/ACCESS-NRI/access-cryosphere-data-pool.git

2. Navigate to the cloned directory:

   .. code-block:: bash

      cd access-cryosphere-data-pool

3. Install access-cryosphere-data-pool using pip:

   .. code-block:: bash

      pip install .

.. note::
   Installing from GitHub is recommended only if you want the latest features or are contributing to access-cryosphere-data-pool development.

Verifying the Installation
--------------------------
After installation, you can verify that access-cryosphere-data-pool is installed correctly by running the following commands in a Python shell:

.. code-block:: python

   import datapool
   print(datapool.__version__)