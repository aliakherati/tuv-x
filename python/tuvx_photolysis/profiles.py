# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Vertical profiles on the model height grid: edge/midpoint values and per-layer column densities.

Reference: ``src/profiles/air.F90``, ``o2.F90``, ``o3.F90``, ``from_csv_file.F90``.

Each profile is placed on the height-grid edges and yields:

* ``edge_val`` -- value at every grid interface,
* ``mid_val``  -- midpoint value, ``0.5 * (edge[i] + edge[i+1])``,
* ``layer_dens`` -- per-layer column density [molecule cm-2], including the exospheric top layer.

Faithful details from the Fortran:

* **air / O2** -- log-linear interpolation of number density; per-layer column uses the *geometric*
  mean of the bounding edge densities; an exospheric layer ``edge_top * scale_height`` is added to
  the top layer (scale height 8.01 km). O2 multiplies air by the 0.2095 volume mixing ratio.
* **O3** -- the data are first extended upward in 1 km steps decaying by ``exp(-1/H)`` (H = 4.5 km)
  to the model top, then *linearly* interpolated; per-layer column uses the *arithmetic* midpoint.
  The whole profile is then rescaled so the total column equals a reference value (default 300 DU).
* **temperature (csv)** -- linear interpolation to edges; midpoints are the edge average.

``km2cm = 1e5`` converts the layer thickness (km) to cm for the column density.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import AtmosphereProfile
from .grids import interp_linear

__all__ = ["Profile", "air_profile", "o2_profile", "o3_profile", "temperature_profile"]

_KM2CM = 1.0e5
_DU = 2.687e16  # molecule cm-2 per Dobson Unit
_O2_VMR = 0.2095


@dataclass
class Profile:
    """A vertical profile on the height grid."""

    edge_val: np.ndarray  # (n_levels,)
    mid_val: np.ndarray  # (n_layers,)
    layer_dens: np.ndarray | None = None  # (n_layers,) [molecule cm-2]


def _log_interp_density(height_edges_km, prof: AtmosphereProfile):
    """Log-linear interpolation of a number-density profile onto the grid edges (air/O2)."""
    zdata = prof.altitude_km.copy()
    zdata[-1] = zdata[-1] + 0.001  # matches the Fortran's tiny top bump
    return np.exp(interp_linear(height_edges_km, zdata, np.log(prof.values)))


def _geometric_layer_density(edge_val, delta_km, edge_top, scale_height):
    """Per-layer column density via the geometric mean of edge densities + exospheric top layer."""
    layer = delta_km * np.sqrt(edge_val[:-1]) * np.sqrt(edge_val[1:]) * _KM2CM
    exo = edge_top * scale_height * _KM2CM
    layer = layer.copy()
    layer[-1] = layer[-1] + exo
    return layer


def air_profile(height_edges_km, density: AtmosphereProfile, scale_height: float = 8.01) -> Profile:
    """Air number-density profile (ports ``air.F90``)."""
    edges = np.asarray(height_edges_km, dtype=float)
    delta = np.diff(edges)
    edge_val = _log_interp_density(edges, density)
    mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
    layer_dens = _geometric_layer_density(edge_val, delta, edge_val[-1], scale_height)
    return Profile(edge_val=edge_val, mid_val=mid_val, layer_dens=layer_dens)


def o2_profile(height_edges_km, density: AtmosphereProfile, scale_height: float = 8.01) -> Profile:
    """O2 number-density profile: air density times the 0.2095 VMR (ports ``o2.F90``)."""
    edges = np.asarray(height_edges_km, dtype=float)
    delta = np.diff(edges)
    edge_val = _O2_VMR * _log_interp_density(edges, density)
    mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
    layer_dens = _geometric_layer_density(edge_val, delta, edge_val[-1], scale_height)
    return Profile(edge_val=edge_val, mid_val=mid_val, layer_dens=layer_dens)


def o3_profile(
    height_edges_km,
    ozone: AtmosphereProfile,
    scale_height: float = 4.5,
    reference_column_du: float = 300.0,
) -> Profile:
    """O3 number-density profile (ports ``o3.F90``).

    Extends the data to the model top with an ``exp(-1/H)`` per-km decay, linearly interpolates,
    forms arithmetic-mean layer columns, then rescales to ``reference_column_du`` (default 300 DU).
    """
    edges = np.asarray(height_edges_km, dtype=float)
    delta = np.diff(edges)
    ztop = edges[-1]

    zdata = list(ozone.altitude_km)
    prof = list(ozone.values)
    rfact = np.exp(-1.0 / scale_height) if scale_height != 0.0 else 0.0
    while zdata[-1] <= ztop:
        zdata.append(zdata[-1] + 1.0)
        prof.append(prof[-1] * rfact)
    zdata = np.array(zdata)
    prof = np.array(prof)

    edge_val = interp_linear(edges, zdata, prof)
    mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
    layer_dens = (delta * mid_val * _KM2CM).copy()
    layer_dens[-1] = layer_dens[-1] + edge_val[-1] * scale_height * _KM2CM

    if reference_column_du != 1.0:
        input_du = np.sum(layer_dens) / _DU
        scale = reference_column_du / input_du
        if scale != 1.0:
            edge_val = scale * edge_val
            mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
            layer_dens = (delta * mid_val * _KM2CM).copy()
            layer_dens[-1] = layer_dens[-1] + edge_val[-1] * scale_height * _KM2CM

    return Profile(edge_val=edge_val, mid_val=mid_val, layer_dens=layer_dens)


def temperature_profile(height_edges_km, temperature: AtmosphereProfile) -> Profile:
    """Temperature profile: linear interpolation to edges, midpoints by averaging (ports csv type).

    The ``ussa.temp`` exospheric sentinel row (altitude 1e10) makes the source span the grid so the
    top edge does not extrapolate to zero.
    """
    edges = np.asarray(height_edges_km, dtype=float)
    edge_val = interp_linear(edges, temperature.altitude_km, temperature.values)
    mid_val = 0.5 * (edge_val[:-1] + edge_val[1:])
    return Profile(edge_val=edge_val, mid_val=mid_val)
