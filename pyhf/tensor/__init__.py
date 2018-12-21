from .. import exceptions


class _BackendRetriever(object):
    def __getattr__(self, name):
        if name == 'numpy_backend':
            from .numpy_backend import numpy_backend

            assert numpy_backend
            # for autocomplete and dir() calls
            self.numpy_backend = numpy_backend
            return numpy_backend
        elif name == 'jax_backend':
            from .jax_backend import jax_backend

            assert jax_backend
            # for autocomplete and dir() calls
            self.jax_backend = jax_backend
            return jax_backend
        elif name == 'pytorch_backend':
            try:
                from .pytorch_backend import pytorch_backend

                assert pytorch_backend
                # for autocomplete and dir() calls
                self.pytorch_backend = pytorch_backend
                return pytorch_backend
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing PyTorch. The pytorch backend cannot be used.",
                    e,
                )
        elif name == 'tensorflow_backend':
            try:
                from .tensorflow_backend import tensorflow_backend

                assert tensorflow_backend
                # for autocomplete and dir() calls
                self.tensorflow_backend = tensorflow_backend
                return tensorflow_backend
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing TensorFlow. The tensorflow backend cannot be used.",
                    e,
                )
        elif name == 'mxnet_backend':
            try:
                from .mxnet_backend import mxnet_backend

                assert mxnet_backend
                # for autocomplete and dir() calls
                self.mxnet_backend = mxnet_backend
                return mxnet_backend
            except ImportError as e:
                raise exceptions.ImportBackendError(
                    "There was a problem importing MxNet. The mxnet backend cannot be used.",
                    e,
                )


BackendRetriever = _BackendRetriever()
__all__ = ['BackendRetriever']
