# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

"""OpenSTB diagnostics, with no standalone imports or reference-echo comparison.

Select debug.target_model and debug.plot_style in shared_params.py. The stored
results must correspond to the chosen source/model; presentation settings do not
rerun a simulation. screen_style/paper_style control dimensions, fonts and titles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shared_params import SIM_PARAMS
from signal_factory import build_signal_from_params


def selected_model():
    model = SIM_PARAMS["debug"].get("target_model", "rigid")
    if model not in ("rigid", "solid"):
        raise ValueError("target_model must be 'rigid' or 'solid'")
    return model


def model_radius():
    return SIM_PARAMS[selected_model() + "_sphere"]["radius_m"]


def results_path():
    return Path(
        SIM_PARAMS["debug"].get(
            "results_path", Path(__file__).resolve().parents[2] / "simple_points.npz"
        )
    )


def simulation_sample_rate(sample_time, signal_f0):
    measured = 1.0 / np.mean(np.diff(sample_time))
    configured = 10.0 * signal_f0 if signal_f0 is not None else 30e3
    if not np.isclose(measured, configured, rtol=1e-10):
        raise ValueError(
            "Stored sample rate disagrees with selected source configuration"
        )
    # Avoid a roundoff-induced one-sample change in ceil(1.1*T*fs).
    return configured


def reconstruct_simulation_source(signal, sample_rate, sample_time, carrier):
    """Sample on the controller's Ns+guard grid, before any diagnostic padding.

    In particular, LFM RMS-after-window scaling depends on the entire record.
    Sampling a shorter plotting interval would change its pressure amplitude.
    """
    ns = len(sample_time)
    guard = int(np.ceil(signal.duration * 1.1 * sample_rate))
    time = np.arange(ns + guard) / sample_rate
    return time, signal.sample(time, carrier)


def finish_figure():
    """Apply presentation only; never change plotted data."""
    style = SIM_PARAMS["debug"].get("plot_style", "screen")
    if style not in ("screen", "paper"):
        raise ValueError("plot_style must be 'screen' or 'paper'")
    fig = plt.gcf()
    paper = style == "paper"
    settings = SIM_PARAMS["debug"].get(style + "_style", {})
    rows = len(fig.axes)
    default_size = (12, 8) if rows > 1 else (10, 4)
    if not paper:
        default_size = (11, 7) if rows > 1 else (11, 4.5)
    fig.set_size_inches(
        settings.get("figsize_multi" if rows > 1 else "figsize_single", default_size)
    )
    label_size = settings.get("label_size", 38 if paper else 12)
    tick_size = settings.get("tick_size", 34 if paper else 10)
    title_size = settings.get("title_size", 14)
    show_titles = settings.get("show_titles", not paper)
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.tick_params(labelsize=tick_size)
        if not ax.get_title():
            ax.set_title(f"{selected_model().title()} sphere — {ax.get_ylabel()}")
        if not show_titles:
            ax.set_title("")
        ax.title.set_fontsize(title_size)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(settings.get("legend_size", 24 if paper else 10))
    fig.tight_layout()
    plt.show()


def plot_incident_spectrum(s: np.ndarray, sample_rate_plot: float, carrier=0.0):
    n_fft = int(SIM_PARAMS["debug"].get("incident_fft_points", 16384))
    dt_plot = 1.0 / sample_rate_plot

    n_fft = max(n_fft, len(s))
    if selected_model() == "solid" or carrier != 0:
        # Physical-frequency view of a complex baseband source (including LFM
        # for either target). Preserve the historical rigid real-tone view below.
        incident_fft = np.fft.fft(s, n_fft) * dt_plot
    else:
        incident_fft = 2 * np.fft.fft(np.real(s), n_fft) * dt_plot

    freq = np.fft.fftfreq(n_fft, dt_plot) + carrier
    positive = freq >= 0
    freq_positive = freq[positive]
    incident_fft_positive = incident_fft[positive]
    order = np.argsort(freq_positive)
    freq_positive = freq_positive[order]
    incident_fft_positive = incident_fft_positive[order]

    c_plot = SIM_PARAMS["environment"]["sound_speed_ms"]
    a = model_radius()

    k_positive = 2.0 * np.pi * freq_positive / c_plot
    ka_positive = k_positive * a

    magnitude = np.abs(incident_fft_positive)
    phase = np.mod(np.angle(incident_fft_positive) + 2.0 * np.pi, 2.0 * np.pi)
    threshold = float(SIM_PARAMS["debug"].get("incident_phase_min_relative", 1e-3))
    if not 0 < threshold <= 1:
        raise ValueError("incident_phase_min_relative must be in (0, 1]")
    # Phase near zero spectral magnitude is not informative. NaNs also prevent
    # lines connecting across excluded bands and across wrapped-phase jumps.
    phase[magnitude < magnitude.max()*threshold] = np.nan
    if not magnitude.max():
        phase[:] = np.nan
    jumps = np.r_[False, np.abs(np.diff(phase)) > np.pi]
    phase[jumps] = np.nan

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(ka_positive, magnitude, "k-", linewidth=1.5)
    ax1.set_xlabel("ka", fontsize=38)
    ax1.set_ylabel("|g(ka)|", fontsize=38)
    ax1.tick_params(axis="x", labelsize=34)
    ax1.tick_params(axis="y", labelsize=34)
    ax1.set_title(
        f"{selected_model().title()} sphere — incident-signal spectrum vs ka: magnitude"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)

    ax2.plot(ka_positive, phase, "k-", linewidth=1.5)
    ax2.set_xlabel("ka", fontsize=38)
    ax2.set_ylabel(r"$\angle g(ka)\ \mathrm{(rad)}$", fontsize=38)
    ax2.set_title(
        "Incident-signal spectrum vs ka: wrapped phase\n"
        f"Shown where magnitude ≥ {20*np.log10(threshold):g} dB relative to peak; wrap jumps not connected"
    )
    ax2.tick_params(axis="x", labelsize=34)
    ax2.tick_params(axis="y", labelsize=34)
    # ax2.set_ylabel("Phase of g(ka) (radians)")
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels(
        [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    )
    # ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 2 * np.pi)

    finish_figure()


def plot_incident_spectrum_hz(
    signal,
    sample_rate: float,
    baseband_frequency: float,
    sample_time: np.ndarray,
):
    # Match simple_points controller logic: Ns + guard band, complex FFT, fftshift.
    t, s = reconstruct_simulation_source(
        signal, sample_rate, sample_time, baseband_frequency
    )
    nfft = len(t)

    S = np.fft.fftshift(np.fft.fft(s))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate)) + baseband_frequency

    mag = np.abs(S)
    mag_norm = mag / (np.max(mag) + 1e-30)
    mag_db = 20.0 * np.log10(np.maximum(mag_norm, 1e-12))

    # Auto frequency window based on the signal band.
    fmin = float(signal.minimum_frequency)
    fmax = float(signal.maximum_frequency)
    bw = max(fmax - fmin, 4.0 / signal.duration)
    pad = 0.25 * bw
    x0 = max(0.0, fmin - pad)
    x1 = fmax + pad

    plt.figure(figsize=(10, 4))
    plt.plot(f, mag_db, "b-", linewidth=1.2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB, normalized]")
    plt.title(
        f"{selected_model().title()} sphere — incident-signal spectrum vs physical frequency (dB)\n"
        "Recalculated using OpenSTB's FFT procedure:\n"
        "complex source, same sampling, recording length + guard samples"
    )
    plt.grid(True, alpha=0.3)
    plt.xlim(x0, x1)
    plt.ylim(-120, 5)
    finish_figure()

    # Linear magnitude view (normalized)
    plt.figure(figsize=(10, 4))
    plt.plot(f, mag_norm, "k-", linewidth=1.2)
    plt.xlabel("Frequency [Hz]", fontsize=38)
    plt.ylabel("Magnitude [linear, normalised]", fontsize=32)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)
    plt.title(
        f"{selected_model().title()} sphere — incident-signal spectrum vs physical frequency\n"
        "Linear magnitude, normalized to its maximum"
    )
    plt.grid(True, alpha=0.3)
    plt.xlim(x0, x1)
    plt.ylim(0.0, 1.05)
    finish_figure()


def plot_form_function_from_dump():
    model = selected_model()
    key = "solid_plugin_dump_path" if model == "solid" else "plugin_dump_path"
    dump_path = Path(SIM_PARAMS["debug"][key])
    if not dump_path.exists():
        print(f"Plugin dump not found: {dump_path}")
        return
    with np.load(dump_path, allow_pickle=False) as data:
        ka = np.asarray(data["ka"]).ravel()
        if model == "solid":
            if str(data["target_model"].item()) != "solid_sphere":
                raise ValueError("Selected dump is not a solid-sphere dump")
            positive = data["f_hz"] > 0
            ka = ka[positive]
            ff = data["ff_hickling_complex"][positive]
            converted = data["ff_openstb_complex"][positive]
            band = data["validation_ka_bounds"].copy()
        else:
            ff = data["ff_complex"].ravel()
            theta = float(data["theta_sample_rad"])
    if ff.size != ka.size:
        raise ValueError("Incompatible form-function dump dimensions")
    order = np.argsort(ka)
    ka, ff = ka[order], ff[order]
    if model == "solid":
        converted = converted[order]
    magnitude = abs(ff)
    phase = np.mod(np.angle(ff), 2 * np.pi)
    # Preserve both historical rigid views: focused and full ka range.
    limits = (
        [14, float(ka.max())]
        if model == "rigid"
        else [SIM_PARAMS["debug"].get("solid_form_function_ka_max", 40)]
    )
    for limit in limits:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(
            ka, magnitude, label="Hickling modal" if model == "solid" else "OpenSTB"
        )
        ax1.set(
            xlabel="ka",
            ylabel="|f(ka)|",
            xlim=(0, limit),
            title=(
                f"{model.title()} sphere — form-function magnitude from OpenSTB plugin dump\n"
                f"Displayed range: 0 ≤ ka ≤ {limit:g}"
            ),
        )
        if model == "rigid":
            ax1.set_ylim(0, 1.5)
        else:
            visible = ka <= limit
            ax1.set_ylim(0, 1.1 * magnitude[visible].max() if np.any(visible) else 1)
        ax2.plot(ka, phase, label="Hickling modal" if model == "solid" else "OpenSTB")
        if model == "solid":
            ax2.plot(
                ka,
                np.mod(np.angle(converted), 2 * np.pi),
                "--",
                label="Conjugated OpenSTB (before frequency mask)",
            )
            ax2.legend()
            ax1.legend()
            if len(band):
                for ax in (ax1, ax2):
                    ax.axvspan(*band, alpha=0.08, color="green")
        ax2.set(
            xlabel="ka",
            ylabel="arg[f(ka)] [rad]",
            xlim=(0, limit),
            ylim=(0, 2 * np.pi),
            title=(
                (
                    "OpenSTB plugin dump — wrapped phase: Hickling and conjugated conventions\n"
                    "Before frequency mask: phase shown before excluded bins are set to zero\n"
                    + (
                        f"Green region: {band[0]:g} ≤ ka ≤ {band[1]:g}, retained for the echo; "
                        "outside: zero response"
                        if len(band)
                        else "No green region: no reference-band restriction"
                    )
                    + "\nOnly positive physical frequencies contribute; phase at zero response is undefined"
                )
                if model == "solid"
                else f"OpenSTB plugin dump — wrapped phase, theta={theta:.5f} rad"
            ),
        )
        for ax in (ax1, ax2):
            ax.grid(True, alpha=0.3)
        finish_figure()
    if model == "solid":
        nonzero = (ka > 0) & (abs(ff) > 1e-12)
        plt.figure()
        plt.plot(ka[nonzero], -np.unwrap(np.angle(ff[nonzero])) / ka[nonzero])
        plt.xlabel("ka")
        plt.ylabel("-unwrapped arg(f) / ka")
        plt.title("Solid sphere — Hickling phase parameter (plugin dump)")
        plt.xlim(0, limits[0])
        plt.grid(True, alpha=0.3)
        finish_figure()


def plot_echo_spectrum(trace: np.ndarray, sample_rate: float, carrier=0.0):
    n_fft = int(SIM_PARAMS["debug"].get("echo_fft_points", 16384))
    dt = 1.0 / sample_rate

    if selected_model() == "solid":
        n_fft = max(n_fft, len(trace))
        echo_fft = np.fft.fft(trace, n_fft) * dt
    else:
        echo_fft = 2 * np.fft.fft(np.real(trace), n_fft) * dt

    freq = np.fft.fftfreq(n_fft, dt) + carrier
    positive = freq >= 0
    freq_positive = freq[positive]
    echo_fft_positive = echo_fft[positive]

    c_plot = SIM_PARAMS["environment"]["sound_speed_ms"]
    a = model_radius()
    k_positive = 2.0 * np.pi * freq_positive / c_plot
    ka_positive = k_positive * a

    magnitude = np.abs(echo_fft_positive)
    phase = np.mod(np.angle(echo_fft_positive) + 2.0 * np.pi, 2.0 * np.pi)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(ka_positive, magnitude, "k-", linewidth=1.5)
    ax1.set_xlabel("ka")
    ax1.set_ylabel("|E(ka)|")
    ax1.set_title("Echo spectrum E(ka) - magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)

    ax2.plot(ka_positive, phase, "k-", linewidth=1.5)
    ax2.set_xlabel("ka")
    ax2.set_ylabel("Phase of E(ka) (radians)")
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 2 * np.pi)

    finish_figure()


def plot_echo_spectrum_hz(trace: np.ndarray, sample_rate: float, carrier=0.0):
    n_fft = int(SIM_PARAMS["debug"].get("echo_fft_points", 16384))
    dt = 1.0 / sample_rate

    if selected_model() == "solid":
        n_fft = max(n_fft, len(trace))
    E = np.fft.fftshift(np.fft.fft(trace, n_fft))
    f = np.fft.fftshift(np.fft.fftfreq(n_fft, dt)) + carrier

    mag = np.abs(E)
    mag_norm = mag / (np.max(mag) + 1e-30)
    mag_db = 20.0 * np.log10(np.maximum(mag_norm, 1e-12))

    # Auto window based on energy support
    keep = mag_norm > 1e-3  # -60 dB
    if np.any(keep):
        fmin = float(f[keep].min())
        fmax = float(f[keep].max())
    else:
        fmin = float(f.min())
        fmax = float(f.max())

    bw = max(fmax - fmin, 1.0)
    pad = 0.25 * bw
    x0 = fmin - pad
    x1 = fmax + pad

    # dB view
    plt.figure(figsize=(10, 4))
    plt.plot(f, mag_db, "k-", linewidth=1.2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB, normalized]")
    plt.title("Echo spectrum in frequency domain")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, x1)
    plt.ylim(-120, 5)
    finish_figure()

    # linear view
    plt.figure(figsize=(10, 4))
    plt.plot(f, mag_norm, "k-", linewidth=1.5)
    plt.xlabel("Frequency [Hz]", fontsize=38)
    plt.ylabel("Magnitude [linear, normalized]", fontsize=32)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)
    plt.title(
        f"{selected_model().title()} sphere — echo spectrum vs physical frequency\n"
        "Linear magnitude, normalized to its maximum"
    )
    plt.grid(True, alpha=0.3)
    plt.xlim(0, x1)
    plt.ylim(0.0, 1.05)
    finish_figure()


def main():
    selected_model()
    signal_mode = SIM_PARAMS["signal"]["mode"]
    signal, f0, sim_baseband_frequency = build_signal_from_params(SIM_PARAMS)

    with np.load(results_path(), allow_pickle=False) as stored:
        result_carrier = float(stored["baseband_frequency"])
    if not np.isclose(result_carrier, sim_baseband_frequency):
        raise ValueError("Result carrier disagrees with selected incident source")
    if SIM_PARAMS["debug"].get("plot_incident", False):
        # Diagnostic: sample the incident signal on the exact simulation time grid
        # used to generate the echoes in simple_points.npz.
        results = np.load(results_path())
        t_sim = results["sample_time"]
        fs_sim = simulation_sample_rate(t_sim, f0)
        _, source_full = reconstruct_simulation_source(
            signal, fs_sim, t_sim, sim_baseband_frequency
        )
        s_sim = source_full[:len(t_sim)]
        if f0 is not None:
            print(
                f"Simulation-grid incident sampling: fs={fs_sim:.1f} Hz, f0={f0:.1f} Hz, samples/cycle={fs_sim / f0:.3f}"
            )

        s_sim_passband = np.real(
            s_sim * np.exp(1j * 2.0 * np.pi * sim_baseband_frequency * t_sim)
        )

        # #Uncomment the following block to plot the incident signal on the simulation grid with both the reconstructed passband view and the baseband magnitude envelope.
        # plt.figure(figsize=(10, 4))
        # plt.plot(t_sim * 1e3, s_sim_passband, "c-", linewidth=1.0, label="reconstructed passband")
        # plt.plot(t_sim * 1e3, np.abs(s_sim), "m--", linewidth=1.0, label="abs(baseband)")
        # plt.plot(t_sim * 1e3, -np.abs(s_sim), "m--", linewidth=1.0)
        # plt.xlabel("Time [ms]")
        # plt.ylabel("Amplitude")
        # plt.title("Incident chirp on simulation grid (passband view for visualization)")
        # plt.grid(True, alpha=0.3)
        # plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        # plt.xlim(0.0, signal.duration * 1e3)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        # # --------------------

        # Durations: emitted chirp vs full hydrophone recording window
        chirp_duration_s = float(signal.duration)
        hydro_duration_s = float(t_sim[-1] - t_sim[0]) if len(t_sim) > 1 else 0.0
        print(f"Incident pulse duration: {chirp_duration_s * 1e3:.2f} ms")
        print(f"Hydrophone trace duration: {hydro_duration_s * 1e3:.2f} ms")

        # Second incident plot: reconstructed passband only, over full recording window
        plt.figure(figsize=(10, 4))
        plt.plot(t_sim * 1e3, s_sim_passband, "k-", linewidth=1.5)

        plt.xlabel("Time [ms]", fontsize=38)
        plt.ylabel(
            "Amplitude [Pa]", fontsize=38
        )  # o "Normalized amplitude" si normalizas
        plt.title(
            f"{selected_model().title()} sphere — incident {signal_mode} (passband)"
        )

        plt.xticks(fontsize=34)
        plt.yticks(fontsize=34)

        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.xlim(0, signal.duration * 1.05e3)
        finish_figure()

        if SIM_PARAMS["debug"].get("plot_incident_spectrum", False):
            plot_incident_spectrum(source_full, fs_sim, sim_baseband_frequency)
            plot_incident_spectrum_hz(
                signal=signal,
                sample_rate=fs_sim,
                baseband_frequency=sim_baseband_frequency,
                sample_time=t_sim,
            )

    if SIM_PARAMS["debug"].get("plot_form_function_from_plugin_dump", False):
        plot_form_function_from_dump()

    if SIM_PARAMS["debug"].get("plot_echo_linear_pressure", True):
        results = np.load(results_path())
        t_echo = results["sample_time"]
        P_echo = results["pressure"]
        fs_echo = simulation_sample_rate(t_echo, f0)

        ping_to_plot = int(SIM_PARAMS["debug"].get("ping_index", 31))
        rx_idx = int(SIM_PARAMS["debug"].get("receiver_index", 0))
        if not (0 <= ping_to_plot < P_echo.shape[0] and 0 <= rx_idx < P_echo.shape[1]):
            raise ValueError("Selected ping/receiver is outside stored results")
        trace_echo_bb = P_echo[ping_to_plot, rx_idx, :]  # complejo en banda base
        trace_echo_pb = np.real(
            trace_echo_bb * np.exp(1j * 2.0 * np.pi * sim_baseband_frequency * t_echo)
        )

        plt.figure(figsize=(10, 4))
        plt.plot(
            t_echo * 1e3,
            trace_echo_pb,
            "k-",
            linewidth=1.2,
            label="passband view for visualization",
        )
        plt.xlabel("Time [ms]", fontsize=38)
        plt.ylabel("Amplitude [Pa]", fontsize=38)
        plt.xticks(fontsize=34)
        plt.yticks(fontsize=34)
        plt.title(
            f"{selected_model().title()} sphere — echo, ping {ping_to_plot}, rx {rx_idx}"
        )
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.legend(fontsize=30)
        finish_figure()

        # Incident signal on the same hydrophone time axis (full trace)-------------------------------------
        _, incident_full = reconstruct_simulation_source(
            signal, fs_echo, t_echo, sim_baseband_frequency
        )
        incident_bb_on_echo_grid = incident_full[:len(t_echo)]
        incident_pb_on_echo_grid = np.real(
            incident_bb_on_echo_grid
            * np.exp(1j * 2.0 * np.pi * sim_baseband_frequency * t_echo)
        )

        # Optional normalization for visual comparison
        inc_peak = np.max(np.abs(incident_pb_on_echo_grid))
        echo_peak = np.max(np.abs(trace_echo_pb))
        incident_plot = (
            incident_pb_on_echo_grid / inc_peak
            if inc_peak > 0
            else incident_pb_on_echo_grid
        )
        echo_plot = trace_echo_pb / echo_peak if echo_peak > 0 else trace_echo_pb

        plt.figure(figsize=(10, 4))
        plt.plot(
            t_echo * 1e3,
            incident_plot,
            "b-",
            linewidth=1.0,
            alpha=0.9,
            label="incident passband (normalized)",
        )
        plt.plot(
            t_echo * 1e3,
            echo_plot,
            "k-",
            linewidth=1.2,
            alpha=0.9,
            label="echo passband (normalized)",
        )
        plt.xlabel("Time [ms]", fontsize=38)
        plt.ylabel("Individually normalized amplitude", fontsize=38)
        plt.title(
            "Incident signal and echo — each divided by its own peak amplitude",
            fontsize=38,
        )
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.xlim(t_echo[0] * 1e3, t_echo[-1] * 1e3)
        plt.legend(fontsize=34)
        finish_figure()

        # Original amplitudes (no normalization):------------------------------
        plt.figure(figsize=(10, 4))
        plt.plot(
            t_echo * 1e3,
            incident_pb_on_echo_grid,
            "b-",
            linewidth=1.0,
            alpha=0.9,
            label="incident passband",
        )
        plt.plot(
            t_echo * 1e3,
            trace_echo_pb,
            "k-",
            linewidth=1.2,
            alpha=0.9,
            label="echo passband",
        )
        plt.xlabel("Time [ms]", fontsize=38)
        plt.ylabel("Pressure [Pa] — not normalized", fontsize=38)
        plt.title(
            "Incident signal and echo — original amplitudes, no normalization",
            fontsize=38,
        )
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.xlim(t_echo[0] * 1e3, t_echo[-1] * 1e3)
        plt.legend(fontsize=34)
        finish_figure()

        if SIM_PARAMS["debug"].get("plot_echo_spectrum", True):
            spectrum_trace = (
                trace_echo_bb if selected_model() == "solid" else trace_echo_pb
            )
            spectrum_carrier = (
                sim_baseband_frequency if selected_model() == "solid" else 0
            )
            plot_echo_spectrum(spectrum_trace, fs_echo, spectrum_carrier)
            plot_echo_spectrum_hz(spectrum_trace, fs_echo, spectrum_carrier)

    # Theta diagnostic across pings (current simple_points_study geometry).
    results = np.load(results_path())
    ping_t = results["ping_start_time"]

    start_pos = np.array([0.0, 0.0, 0.0])
    speed = 1.5
    tx_offset = np.array([0.0, 1.2, 0.3])
    rx_offset = np.array([0.0, 1.2, 0.0])
    target = np.array([5.0, 40.0, 10.0])

    vehicle_pos = start_pos + np.column_stack(
        [speed * ping_t, np.zeros_like(ping_t), np.zeros_like(ping_t)]
    )
    tx_pos = vehicle_pos + tx_offset
    rx_pos = vehicle_pos + rx_offset

    inc = target[np.newaxis, :] - tx_pos
    sca = rx_pos - target[np.newaxis, :]

    cos_th = np.sum(inc * sca, axis=1) / (
        np.linalg.norm(inc, axis=1) * np.linalg.norm(sca, axis=1)
    )
    cos_th = np.clip(cos_th, -1.0, 1.0)
    theta = np.arccos(cos_th)

    print("theta(rad) min/max:", float(theta.min()), float(theta.max()))
    print(
        "theta(deg) min/max:",
        float(np.degrees(theta).min()),
        float(np.degrees(theta).max()),
    )
    print("first 10 theta(deg):", np.degrees(theta[:10]))
    print("theta ping 14 [rad]:", float(theta[14]))

    fs = simulation_sample_rate(results["sample_time"], f0)
    Ns = len(results["sample_time"])
    gb = int(np.ceil(signal.duration * 1.1 * fs))
    nfft_sim = Ns + gb
    df_sim = fs / nfft_sim

    print("FFT diagnostic (sim):")
    print("  Ns:", Ns)
    print("  gb:", gb)
    print("  Nfft_sim = Ns + gb:", nfft_sim)
    print("  df_sim [Hz]:", df_sim)
    print("  Reference standalone FFT points:", 16384)
    print("  df_reference [Hz]:", fs / 16384.0)


if __name__ == "__main__":
    main()
