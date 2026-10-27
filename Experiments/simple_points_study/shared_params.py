# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

SIM_PARAMS = {
    "environment": {
        "salinity": 14.5,
        "sound_speed_ms": 1410.0,  # 1410 for Hickling controlled comparison,
        # 1480 for Rudgers
        "temperature_c": 11.2,
    },
    "rigid_sphere": {
        "radius_m": 0.25,
        "k0a": 15.0,
        "n_terms": 80,
        "scale": 1.0,
        "ka_eps": 1e-8,
    },
    # Armco iron preset from the validated solid-sphere standalone.
    # The active Hickling comparison uses environment.sound_speed_ms = 1410.0.
    "solid_sphere": {
        "radius_m": 0.25,
        "solid_density_kg_m3": 7700.0,
        "compressional_speed_m_s": 5960.0,
        "shear_speed_m_s": 3240.0,
        "fluid_density_kg_m3": 1000.0,
        "n_terms": 80,
        "scale": 1.0,
        "ka_eps": 1e-8,
    },
    "signal": {
        "mode": "hickling",  # "sine", "lfm", or "hickling"
        "n_cycles": 2,  # used only for "sine" mode
        "amplitude": 1.0,  # used only for "sine" mode
        "initial_phase": 0.0,  # used only for "sine" mode
        # Source preset for hickling_fig16_max. For paper comparison also use
        # Armco iron, environment c=1410 m/s and the reference ka band 10..40.
        # Phase is derived in signal_factory.py, not the sine initial_phase.
        "hickling": {
            "k0a": 24.5,
            "n_cycles": 5,
            "amplitude": 1.0,
            "integration_ka_bounds": (10.0, 40.0),  # Reference only, not sphere physics
        },
    },
    "debug": {
        "target_model": "solid",  # "rigid" or "solid"; match the simulation run
        "plot_style": "screen",  # "screen" or "paper"
        "results_path": str(Path(__file__).resolve().parents[2] / "simple_points.npz"),
        "ping_index": 31,
        "receiver_index": 0,
        "solid_form_function_ka_max": 40.0,
        "screen_style": {
            "show_titles": True,
            "figsize_single": (11, 4.5),
            "figsize_multi": (11, 7),
            "label_size": 12,
            "tick_size": 10,
            "legend_size": 10,
        },
        "paper_style": {
            "show_titles": False,
            "figsize_single": (10, 4),
            "figsize_multi": (12, 8),
            "label_size": 38,
            "tick_size": 34,
            "legend_size": 24,
        },
        "plot_incident": True,
        "plot_incident_spectrum": True,
        "incident_fft_points": 16384,
        # Used in the FFT of the incident spectrum graph in ka in debug_plots.py.
        # 'incident_fft_points' is not used in the Hz-spectrum graph of the
        # incident signal in debug_plots.py.
        "plot_form_function": True,
        # "form_function_ka_max": 14,
        # "form_function_points": 2000,
        "dump_form_function_from_plugin": True,
        "plugin_dump_path": str(
            Path(__file__).resolve().parent / "rigid_sphere_ff_debug.npz"
        ),
        "plot_form_function_from_plugin_dump": True,
        # Independent solid dump; does not change the rigid plotting path above.
        "dump_solid_form_function_from_plugin": True,
        "solid_plugin_dump_path": str(
            Path(__file__).resolve().parent / "solid_sphere_ff_debug.npz"
        ),
    },
}
