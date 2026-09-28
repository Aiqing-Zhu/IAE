"""IAE encoders.

The numpy reference implementation lives in :mod:`iae.encoders.interface` and is
re-exported here.  :mod:`iae.encoders.gpu` holds a batched torch version used to
encode whole datasets; it is imported explicitly (``from iae.encoders import gpu``)
so that importing this package never pulls in torch.
"""

from .extension import (basis_1d, basis_reconstruct, extend_and_project, mcshane_extend,
                        signed_distance_from_edges)
from .interface import (boundary_moment, decode_interface, decode_state,
                        decode_surface_function, domain_moment, encode_gext,
                        encode_interface, encode_state, encode_surface_function,
                        sdf_geometry_code)
from .utils import poly_sdf, union_sdf

__all__ = [
    "basis_1d", "basis_reconstruct", "extend_and_project", "mcshane_extend",
    "signed_distance_from_edges",
    "encode_state", "decode_state",
    "encode_interface", "decode_interface",
    "encode_surface_function", "decode_surface_function",
    "encode_gext", "sdf_geometry_code", "domain_moment", "boundary_moment",
    "poly_sdf", "union_sdf",
]
