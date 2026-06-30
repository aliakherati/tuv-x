# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Solar and spherical geometry.

Stage 4. Reference: ``src/profiles/earth_sun_distance.F90``, ``src/spherical_geometry.F90``,
``src/profiles/profile_utils.F90``. Provides Earth-Sun distance (1/r^2 scaling) and the
Dahlback-Stamnes slant-path geometry (``dsdh``/``nid``, slant optical depth, ``air_mass``),
including the SZA>90 (twilight) handling. The solar zenith angle itself is taken from
frank-model's existing ``src-python/solar.py`` to avoid duplicating the astronomy.
Also derives the box-model altitude from its pressure via the US Standard Atmosphere.
"""
