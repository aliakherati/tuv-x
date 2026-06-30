# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Radiator optical properties and accumulation.

Stage 5. Reference: ``src/radiative_transfer/radiator.F90`` and
``src/radiative_transfer/radiators/`` (rayleigh, o2, o3, aerosol). Each radiator yields layer
optical depth, single-scattering albedo, and asymmetry factor; these are accumulated into the
total ``(OD, SSA, G)`` fed to the solver (air/Rayleigh uses a fixed asymmetry of 0.1).
"""
