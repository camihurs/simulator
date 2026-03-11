# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openstb.simulator.plugin import loader


SIM_PARAMS = {
    "environment": {
        "sound_speed_ms": 1480.0,
    },
    "rigid_sphere": {
        "radius_m": 0.25,
        "k0a": 15.0,
    },
    "signal": {
        "mode": "sine",  # "sine" or "lfm"
        "n_cycles": 2,
        "amplitude": 1.0,
        "initial_phase": 0.0,
    },
    "debug": {
        "plot_incident": True,
        "plot_incident_spectrum": True,
        "incident_fft_points": 16384,
        "plot_form_function_from_plugin_dump": True,
        "plugin_dump_path": str(Path(__file__).resolve().parent / "rigid_sphere_ff_debug.npz"),
    },
}


def build_signal(signal_mode: str):
    if signal_mode == "lfm":
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
        c = SIM_PARAMS["environment"]["sound_speed_ms"]
        a = SIM_PARAMS["rigid_sphere"]["radius_m"]
        k0a = SIM_PARAMS["rigid_sphere"]["k0a"]
        f0 = k0a * c / (2.0 * np.pi * a)

        signal = loader.signal(
            {
                "name": "SinusoidBurst:openstb.simulator.system.signal",
                "parameters": {
                    "f0": f0,
                    "n_cycles": SIM_PARAMS["signal"]["n_cycles"],
                    "amplitude": SIM_PARAMS["signal"]["amplitude"],
                    "initial_phase": SIM_PARAMS["signal"]["initial_phase"],
                },
            }
        )
    else:
        raise ValueError(f"Unknown signal_mode '{signal_mode}'")

    return signal, f0


def sample_for_plot(signal, signal_mode: str, f0: float | None):
    if signal_mode == "sine":
        sample_rate_plot = 100.0 * f0
        baseband_frequency_plot = 0.0
        title = f"Incident Signal: {SIM_PARAMS['signal']['n_cycles']}-cycle sinusoid at {f0:.1f} Hz"
    else:
        sample_rate_plot = 10.0 * 30e3
        baseband_frequency_plot = 110e3
        title = f"Incident signal ({signal_mode})"

    t_end = 3.0 * signal.duration
    t = np.arange(0.0, t_end, 1.0 / sample_rate_plot)
    s = signal.sample(t, baseband_frequency_plot)

    return t, s, sample_rate_plot, title


def plot_incident_signal(t: np.ndarray, s: np.ndarray, title: str):
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e3, np.real(s), "b-", linewidth=1.5)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
    plt.xlim(-0.05, (t[-1] if len(t) else 0.0) * 1e3)
    plt.tight_layout()
    plt.show()


def plot_incident_spectrum(s: np.ndarray, sample_rate_plot: float):
    n_fft = int(SIM_PARAMS["debug"].get("incident_fft_points", 16384))
    dt_plot = 1.0 / sample_rate_plot

    incident_fft = np.fft.fft(np.real(s), n_fft) * dt_plot
    incident_fft *= 2.0

    freq = np.fft.fftfreq(n_fft, dt_plot)
    positive = freq >= 0
    freq_positive = freq[positive]
    incident_fft_positive = incident_fft[positive]

    c_plot = SIM_PARAMS["environment"]["sound_speed_ms"]
    a = SIM_PARAMS["rigid_sphere"]["radius_m"]

    k_positive = 2.0 * np.pi * freq_positive / c_plot
    ka_positive = k_positive * a

    magnitude = np.abs(incident_fft_positive)
    phase = np.mod(np.angle(incident_fft_positive) + 2.0 * np.pi, 2.0 * np.pi)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(ka_positive, magnitude, "b-", linewidth=1.5)
    ax1.set_xlabel("ka")
    ax1.set_ylabel("|g(ka)|")
    ax1.set_title("Incident spectrum g(ka) - magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)

    ax2.plot(ka_positive, phase, "r-", linewidth=1.5)
    ax2.set_xlabel("ka")
    ax2.set_ylabel("Phase of g(ka) (radians)")
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 2 * np.pi)

    plt.tight_layout()
    plt.show()


def plot_form_function_from_dump():
    dump_path = Path(SIM_PARAMS["debug"]["plugin_dump_path"])

    if not dump_path.exists():
        print(f"Plugin dump not found: {dump_path}")
        return

    data = np.load(dump_path)
    ka_dump = data["ka"]
    mag_dump = data["ff_magnitude"]
    phase_dump = data["ff_phase"]
    theta_sample = float(data["theta_sample_rad"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(ka_dump, mag_dump, "b-", linewidth=1.5)
    ax1.set_xlabel("ka")
    ax1.set_ylabel("|f(ka)|")
    ax1.set_title("Form Function from plugin dump - Magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 1.5)

    ax2.plot(ka_dump, phase_dump, "r-", linewidth=1.5)
    ax2.set_xlabel("ka")
    ax2.set_ylabel("arg[f(ka)] (radians)")
    ax2.set_title(f"Form Function from plugin dump - Phase (theta={theta_sample:.4f} rad)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])

    plt.tight_layout()
    plt.show()


def main():
    signal_mode = SIM_PARAMS["signal"]["mode"]
    signal, f0 = build_signal(signal_mode)

    if SIM_PARAMS["debug"].get("plot_incident", False):
        t, s, sample_rate_plot, title = sample_for_plot(signal, signal_mode, f0)
        plot_incident_signal(t, s, title)

        if SIM_PARAMS["debug"].get("plot_incident_spectrum", False):
            plot_incident_spectrum(s, sample_rate_plot)

    if SIM_PARAMS["debug"].get("plot_form_function_from_plugin_dump", False):
        plot_form_function_from_dump()


if __name__ == "__main__":
    main()