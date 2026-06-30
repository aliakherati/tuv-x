# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Data loaders (NumPy / netCDF4) for cross sections, quantum yields, profiles, grids, flux.

Stage 2. Reference: ``src/cross_section.F90``, ``src/quantum_yield.F90``, ``src/netcdf.F90``,
``src/profiles/*``, ``src/grid.F90``. Loaders return plain NumPy arrays wrapped in small
dataclasses (``CrossSectionData``, ``QuantumYieldData``, ...) so the user's own JPL files can be
swapped in behind the same interface later.
"""
