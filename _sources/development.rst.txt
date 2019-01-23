Developing
==========

To develop, we suggest using `virtual environments <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`__ together with ``pip`` or using `pipenv <https://pipenv.readthedocs.io/en/latest/>`__. Once the environment is activated, clone the repo from GitHub

.. code-block:: console

    git clone https://github.com/diana-hep/pyhf.git

and install all necessary packages for development

.. code-block:: console

    pip install --ignore-installed -U -e .[complete]

Then setup the Git pre-commit hook for `Black <https://github.com/ambv/black>`__  by running

.. code-block:: console

    pre-commit install
