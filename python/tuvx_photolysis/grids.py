# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Grids (edges / midpoints / deltas) and interpolators.

Stage 3. Reference: ``src/grid.F90``, ``src/interpolate.F90``. Provides a ``Grid`` container
matching TUV-x's edge/midpoint convention and the linear + area-conserving interpolators used to
rebin cross sections, quantum yields, and the extraterrestrial flux onto the model wavelength grid.
"""
