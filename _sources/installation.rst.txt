Installation
============

To install, we suggest first setting up a `virtual environment <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`__

.. code-block:: console

    # Python3
    python3 -m venv pyhf


.. code-block:: console

    # Python2
    virtualenv --python=$(which python) pyhf

and activating it

.. code-block:: console

    source pyhf/bin/activate


Install latest stable release from `PyPI <https://pypi.org/project/pyhf/>`__...
-------------------------------------------------------------------------------

... with NumPy backend
++++++++++++++++++++++

.. code-block:: console

    pip install pyhf

... with TensorFlow backend
+++++++++++++++++++++++++++

.. code-block:: console

    pip install pyhf[tensorflow]

... with PyTorch backend
++++++++++++++++++++++++

.. code-block:: console

    pip install pyhf[torch]

... with MXNet backend
++++++++++++++++++++++

.. code-block:: console

    pip install pyhf[mxnet]

... with all backends
+++++++++++++++++++++

.. code-block:: console

    pip install pyhf[tensorflow,torch,mxnet]

Install latest development version from `GitHub <https://github.com/diana-hep/pyhf>`__...
-----------------------------------------------------------------------------------------

... with NumPy backend
++++++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U "git+https://github.com/diana-hep/pyhf.git#egg=pyhf"

... with TensorFlow backend
+++++++++++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U "git+https://github.com/diana-hep/pyhf.git#egg=pyhf[tensorflow]"

... with PyTorch backend
++++++++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U "git+https://github.com/diana-hep/pyhf.git#egg=pyhf[torch]"

... with MXNet backend
++++++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U "git+https://github.com/diana-hep/pyhf.git#egg=pyhf[mxnet]"

... with all backends
+++++++++++++++++++++

.. code-block:: console

    pip install --ignore-installed -U "git+https://github.com/diana-hep/pyhf.git#egg=pyhf[tensorflow,torch,mxnet]"

Updating :code:`pyhf`
---------------------

Rerun the installation command. As the upgrade flag, :code:`-U`, is used then the libraries will be updated.
