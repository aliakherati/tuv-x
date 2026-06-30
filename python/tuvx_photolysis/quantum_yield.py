# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Quantum-yield evaluation.

Reference: ``src/quantum_yield.F90``. The ``base`` quantum yield is either a constant value held
across wavelength, or a tabulated spectrum read from NetCDF and area-conservingly rebinned onto the
model wavelength grid (then replicated across altitude). Reaction-specific quantum-yield recipes
(e.g. the O3 photolysis branching) are separate evaluators to be added as needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import TabulatedData
from .grids import interp_conserving

__all__ = ["ConstantQuantumYield", "TabulatedQuantumYield"]


@dataclass
class ConstantQuantumYield:
    """A wavelength- and temperature-independent quantum yield (``base`` with ``constant value``)."""

    value: float

    def evaluate(self, n_levels: int, n_wavelengths: int) -> np.ndarray:
        return np.full((n_levels, n_wavelengths), float(self.value))


@dataclass
class TabulatedQuantumYield:
    """A tabulated quantum-yield spectrum on the model wavelength grid (replicated across altitude)."""

    array: np.ndarray  # (n_wavelength_cells,)

    @classmethod
    def from_netcdf(
        cls, td: TabulatedData, wl_edges, lower_extrapolation=None, upper_extrapolation=None,
        lower_value=0.0, upper_value=0.0,
    ) -> "TabulatedQuantumYield":
        """Conservingly rebin a single-parameter quantum-yield NetCDF onto the wavelength grid.

        Endpoint extrapolation (e.g. holding the yield at a constant value below the data range,
        as some reactions specify) is applied via :func:`add_points`.
        """
        wl_edges = np.asarray(wl_edges, dtype=float)
        from .cross_section import add_points

        x, y = add_points(
            td.wavelength, td.parameters[:, 0],
            lower_extrapolation, upper_extrapolation, lower_value, upper_value,
        )
        return cls(array=interp_conserving(wl_edges, x, y))

    def evaluate(self, n_levels: int, n_wavelengths: int) -> np.ndarray:
        if self.array.size != n_wavelengths:
            raise ValueError("quantum yield wavelength size mismatch")
        return np.repeat(self.array[None, :], n_levels, axis=0)
