# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np

from openstb.simulator.plugin import loader


def build_signal_from_params(sim_params: dict):
    signal_mode = sim_params["signal"]["mode"]

    if signal_mode == "lfm":
        sim_baseband_frequency = 110e3
        signal = loader.signal(
            {
                "name": "lfm_chirp",
                "parameters": {
                    "f_start": 100e3,
                    "f_stop": 120e3,
                    "duration": 0.015,
                    "rms_spl": 190,
                    "rms_after_window": True,
                    "window": {
                        "name": "tukey",
                        "parameters": {"alpha": 0.2},
                    },
                },
            }
        )
        f0 = None

    elif signal_mode == "sine":
        c = sim_params["environment"]["sound_speed_ms"]
        a = sim_params["rigid_sphere"]["radius_m"]
        k0a = sim_params["rigid_sphere"]["k0a"]
        f0 = k0a * c / (2.0 * np.pi * a)

        sim_baseband_frequency = 0.0
        signal = loader.signal(
            {
                "name": "SinusoidBurst:openstb.simulator.system.signal",
                "parameters": {
                    "f0": f0,
                    "n_cycles": sim_params["signal"]["n_cycles"],
                    "amplitude": sim_params["signal"]["amplitude"],
                    "initial_phase": sim_params["signal"]["initial_phase"],
                },
            }
        )
    else:
        raise ValueError(f"Unknown signal_mode '{signal_mode}'")

    return signal, f0, sim_baseband_frequency