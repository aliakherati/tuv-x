# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Top-level orchestrator: PhotolysisCalculator.

Stage 7. Reference: ``src/core.F90`` (``run``) and ``src/tuvx.F90``. Ties the stages together:
solar geometry + Earth-Sun distance -> spherical geometry -> radiator optical properties ->
delta-Eddington solve -> actinic flux -> per-reaction wavelength integration -> J interpolated to
the requested altitude. Designed for frank-model: ``rate_constants(lat, lon, day_of_year,
utc_hour, ...)`` returns J-values (s^-1) keyed by reaction name.
"""
