# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import os
from pathlib import Path
from typing import Any

import numpy as np

from openstb.i18n.support import translations
from openstb.simulator.plugin.abc import ResultConverter, ResultFormat, SimulationConfig

_ = translations.load("openstb.simulator").gettext


class NumpyConverter(ResultConverter):
    """Convert simulation results to a NumPy file.

    The writes the data as a NumPy .npz file with the following variables:
    * ``baseband_frequency``: the frequency used to baseband the data
    * ``ping_start_time``: seconds since the start of the trajectory that the pings
      were sent
    * ``pressure``: the simulated pressure at each receiver
    * ``pressure_dimensions``: string array giving the order of dimensions in
      ``pressure``.
    * ``sample_time``: seconds since the start of its ping that each sample was captured
    * Optional single-target metadata: ``echo_travel_time_s`` (ping, channel),
      ``echo_target_reflectivity`` and schema/distortion information, when requested
      through the controller's ``save_echo_metadata`` configuration.

    Parameters
    ----------
    filename : path-like
        The path to save the converted results at.
    compress : Boolean
        If True, compress the data while writing.

    """

    def __init__(self, filename: os.PathLike[str] | str, compress=False):
        self.filename = Path(filename)
        self.compress = compress
        if self.filename.exists():
            raise ValueError(
                _("output file {filename} already exists").format(
                    filename=self.filename
                )
            )

    def can_handle(self, format: ResultFormat | str, config: SimulationConfig):
        return format == ResultFormat.ZARR_BASEBAND_PRESSURE

    def convert(
        self, format: ResultFormat | str, result: Any, config: SimulationConfig
    ):
        if format != ResultFormat.ZARR_BASEBAND_PRESSURE:
            return False

        rdict = {
            "ping_start_time": result["ping_start_time"][:],
            "sample_time": result["sample_time"][:],
            "pressure": result["pressure"][:],
            "pressure_dimensions": ["ping", "channel", "sample_time"],
            "baseband_frequency": result.attrs["baseband_frequency"],
        }
        if "echo_travel_time_s" in result:
            rdict.update(
                echo_travel_time_s=result["echo_travel_time_s"][:],
                echo_target_reflectivity=result.attrs["echo_target_reflectivity"],
                echo_metadata_schema=result.attrs["echo_metadata_schema"],
                echo_distortion_classes=result.attrs["echo_distortion_classes"],
                echo_travel_time_dimensions=["ping", "channel"],
            )

        if "echo_scattering_angle_rad" in result:
            rdict["echo_scattering_angle_rad"] = result["echo_scattering_angle_rad"][:]

        if self.compress:
            np.savez_compressed(self.filename, **rdict)
        else:
            np.savez(self.filename, **rdict)

        return True
