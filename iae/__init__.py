"""Interface Autoencoder (IAE).

Spectral encode / decode of an interface `Gamma` and of a function `f` defined on
it, plus the MIONet operators that are trained in the resulting code space.

Sub-packages
------------
``iae.encoders``  the autoencoder (numpy) and a batched GPU encoder (torch)
``iae.models``    the MIONet operator network
``iae.paths``     where datasets and checkpoints are looked up
"""

__version__ = "1.0.0"
