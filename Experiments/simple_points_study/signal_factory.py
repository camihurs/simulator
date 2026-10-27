# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

import numpy as np

from openstb.simulator.plugin import loader


def build_signal_from_params(sim_params: dict):
    signal_mode = sim_params["signal"]["mode"]

    if signal_mode == "lfm":
        sim_baseband_frequency = 5.25e3
        signal = loader.signal(
            {
                "name": "lfm_chirp",
                "parameters": {
                    "f_start": 0.5e3,
                    "f_stop": 10e3,
                    "duration": 0.02,
                    "rms_spl": 190,
                    "rms_after_window": True,
                    "window": {
                        "name": "hann",  # Estaba tukey
                        "parameters": {},  # {"alpha": 0.2}
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
    elif signal_mode == "hickling":
        # Positive-exponential counterpart of Hickling's centered pulse.
        # Use the solid radius and actual environment c, not the rigid preset.
        case = sim_params["signal"]["hickling"]
        c = sim_params["environment"]["sound_speed_ms"]
        a = sim_params["solid_sphere"]["radius_m"]
        k0a = case["k0a"]
        n_cycles = case["n_cycles"]
        for name, value in (
            ("sound_speed_ms", c),
            ("radius_m", a),
            ("k0a", k0a),
            ("n_cycles", n_cycles),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Hickling {name} must be finite and positive")
        f0 = k0a * c / (2.0 * np.pi * a)
        sim_baseband_frequency = 0.0
        # SinusoidBurst subtracts pi/2 internally. Choose initial_phase so its
        # exponent is 2*pi*f0*(t - T/2), with T = n_cycles/f0.
        initial_phase = np.pi / 2 - np.pi * n_cycles
        signal = loader.signal(
            {
                "name": "SinusoidBurst:openstb.simulator.system.signal",
                "parameters": {
                    "f0": f0,
                    "n_cycles": n_cycles,
                    "amplitude": case["amplitude"],
                    "initial_phase": initial_phase,
                },
            }
        )
    else:
        raise ValueError(f"Unknown signal_mode '{signal_mode}'")

    return signal, f0, sim_baseband_frequency
