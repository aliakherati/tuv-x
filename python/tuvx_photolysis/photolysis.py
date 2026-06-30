# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Photolysis-rate-constant integration.

Stage 7. Reference: ``src/photolysis_rates.F90`` (``get``, lines 321-381). Assembles the actinic
flux ``(fdr + fup + fdn) * extraterrestrial_flux`` (clamped at >= 0), then for each reaction
integrates over wavelength ``J[z] = sum_lambda actinicFlux[lambda, z] * sigma[lambda, z] *
phi[lambda, z] * scale`` as a dot product per altitude.
"""
