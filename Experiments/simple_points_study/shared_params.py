# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

SIM_PARAMS = {
    "environment": {
        "salinity": 14.5,
        "sound_speed_ms": 1480.0,
        "temperature_c": 11.2,
    },
    "rigid_sphere": {
        "radius_m": 0.25,
        "k0a": 15.0,
        "n_terms": 80,
        "scale": 1.0,
        "ka_eps": 1e-8,
    },
    "signal": {
        "mode": "lfm",  # "sine" or "lfm"
        "n_cycles": 2, #used only for "sine" mode
        "amplitude": 1.0, #used only for "sine" mode
        "initial_phase": 0.0, #used only for "sine" mode
    },
    "debug": {
        "plot_incident": True,
        "plot_incident_spectrum": True,
        "incident_fft_points": 16384, #It is used in the FFT of the incident spectrum graph in ka, in debug_plots.py.
        # 'incident_fft_points' it is not used in the spectrum graph of the incident signal in terms of Hz (in debug_plots.py).
        "plot_form_function": True,
        #"form_function_ka_max": 14,
        #"form_function_points": 2000,
        "dump_form_function_from_plugin": True,
        "plugin_dump_path": str(Path(__file__).resolve().parent / "rigid_sphere_ff_debug.npz"),
        "plot_form_function_from_plugin_dump": True,
    },
}