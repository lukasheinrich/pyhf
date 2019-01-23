API
===

Top-Level
---------

.. currentmodule:: pyhf

.. autosummary::
   :toctree: _generated/

   default_backend
   default_optimizer
   tensorlib
   optimizer
   get_backend
   set_backend

Making Probability Distribution Functions (PDFs)
------------------------------------------------

.. currentmodule:: pyhf.pdf

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   Model
   _ModelConfig

Backends
--------

The computational backends that :code:`pyhf` provides interfacing for the vector-based calculations.

.. currentmodule:: pyhf.tensor

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   mxnet_backend.mxnet_backend
   numpy_backend.numpy_backend
   pytorch_backend.pytorch_backend
   tensorflow_backend.tensorflow_backend

Optimizers
----------

.. currentmodule:: pyhf.optimize

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   opt_pytorch.pytorch_optimizer
   opt_scipy.scipy_optimizer
   opt_tflow.tflow_optimizer
   opt_minuit.minuit_optimizer

Modifiers
---------

.. currentmodule:: pyhf.modifiers

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   histosys
   normfactor
   normsys
   shapefactor
   shapesys
   staterror

Interpolators
-------------

.. currentmodule:: pyhf.interpolators

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   code0
   code1
   code2
   code4

Exceptions
----------

Various exceptions, apart from standard python exceptions, that are raised from using the :code:`pyhf` API.

.. currentmodule:: pyhf.exceptions

.. autosummary::
   :toctree: _generated/
   :nosignatures:
   :template: modifierclass.rst

   InvalidInterpCode
   InvalidModifier

Utilities
---------

.. currentmodule:: pyhf.utils

.. autosummary::
   :toctree: _generated/

   generate_asimov_data
   loglambdav
   pvals_from_teststat
   qmu
   hypotest
