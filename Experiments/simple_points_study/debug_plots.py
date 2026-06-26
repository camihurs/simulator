# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shared_params import SIM_PARAMS
from signal_factory import build_signal_from_params


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

    ax1.plot(ka_positive, magnitude, "k-", linewidth=1.5)
    ax1.set_xlabel("ka", fontsize=38)
    ax1.set_ylabel("|g(ka)|", fontsize=38)
    ax1.tick_params(axis='x', labelsize=34)
    ax1.tick_params(axis='y', labelsize=34)
    #ax1.set_title("Incident spectrum g(ka) - magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)

    ax2.plot(ka_positive, phase, "k-", linewidth=1.5)
    ax2.set_xlabel("ka", fontsize=38)
    ax2.set_ylabel(r"$\angle g(ka)\ \mathrm{(rad)}$", fontsize=38)
    ax2.tick_params(axis='x', labelsize=34)
    ax2.tick_params(axis='y', labelsize=34)
    #ax2.set_ylabel("Phase of g(ka) (radians)")
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    #ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 2 * np.pi)

    plt.tight_layout()
    plt.show()


def plot_incident_spectrum_hz(
    signal,
    sample_rate: float,
    baseband_frequency: float,
    sample_time: np.ndarray,
):
    # Match simple_points controller logic: Ns + guard band, complex FFT, fftshift.
    Ns = len(sample_time)
    gb = int(np.ceil(signal.duration * 1.1 * sample_rate))
    nfft = Ns + gb

    t = np.arange(nfft) / sample_rate
    s = signal.sample(t, baseband_frequency)

    S = np.fft.fftshift(np.fft.fft(s))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / sample_rate)) + baseband_frequency

    mag = np.abs(S)
    mag_norm = mag / (np.max(mag) + 1e-30)
    mag_db = 20.0 * np.log10(np.maximum(mag_norm, 1e-12))

    # Auto frequency window based on the signal band.
    fmin = float(signal.minimum_frequency)
    fmax = float(signal.maximum_frequency)
    bw = max(fmax - fmin, 1.0)
    pad = 0.25 * bw
    x0 = max(0.0, fmin - pad)
    x1 = fmax + pad

    plt.figure(figsize=(10, 4))
    plt.plot(f, mag_db, "b-", linewidth=1.2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB, normalized]")
    plt.title("Incident spectrum in frequency domain (matches simulation FFT pipeline)")
    plt.grid(True, alpha=0.3)
    plt.xlim(x0, x1)
    plt.ylim(-120, 5)
    plt.tight_layout()
    plt.show()

    # Linear magnitude view (normalized)
    plt.figure(figsize=(10, 4))
    plt.plot(f, mag_norm, "k-", linewidth=1.2)
    plt.xlabel("Frequency [Hz]", fontsize=38)
    plt.ylabel("Magnitude [linear, normalised]", fontsize=32)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)
    #plt.title("Incident spectrum in frequency domain (linear scale)")
    plt.grid(True, alpha=0.3)
    plt.xlim(x0, x1)
    plt.ylim(0.0, 1.05)
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
    print("ka max in dump:", float(ka_dump.max()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(ka_dump, mag_dump, "k-", linewidth=1.5)
    ax1.set_xlabel("ka", fontsize=38)
    ax1.set_ylabel("|f(ka)|", fontsize=38)
    ax1.tick_params(axis='x', labelsize=34)
    ax1.tick_params(axis='y', labelsize=34)
    #ax1.set_title("Form Function from plugin dump - Magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 1.5)

    ax2.plot(ka_dump, phase_dump, "k-", linewidth=1.5)
    ax2.set_xlabel("ka", fontsize=38)
    ax2.set_ylabel(r"$\angle f(ka)\ \mathrm{(rad)}$", fontsize=38)
    #ax2.set_ylabel("arg[f(ka)] (radians)", fontsize=22)
    ax2.tick_params(axis='x', labelsize=34)
    ax2.tick_params(axis='y', labelsize=34)
    #ax2.set_title(f"Form Function from plugin dump - Phase (theta={theta_sample:.4f} rad)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels([r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    #ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])

    plt.tight_layout()
    plt.show()


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(ka_dump, mag_dump, "b-", linewidth=1.5)
    ax1.set_xlabel("ka")
    ax1.set_ylabel("|f(ka)|")
    ax1.set_title("Form Function from plugin dump - Magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, float(ka_dump.max()))
    ax1.set_ylim(0, 1.5)

    ax2.plot(ka_dump, phase_dump, "r-", linewidth=1.5)
    ax2.set_xlabel("ka")
    ax2.set_ylabel("arg[f(ka)] (radians)")
    ax2.set_title(f"Form Function from plugin dump - Phase (theta={theta_sample:.4f} rad)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, float(ka_dump.max()))
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])

    plt.tight_layout()
    plt.show()


def plot_echo_spectrum(trace: np.ndarray, sample_rate: float):
    n_fft = int(SIM_PARAMS["debug"].get("echo_fft_points", 16384))
    dt = 1.0 / sample_rate

    echo_fft = np.fft.fft(np.real(trace), n_fft) * dt
    echo_fft *= 2.0

    freq = np.fft.fftfreq(n_fft, dt)
    positive = freq >= 0
    freq_positive = freq[positive]
    echo_fft_positive = echo_fft[positive]

    c_plot = SIM_PARAMS["environment"]["sound_speed_ms"]
    a = SIM_PARAMS["rigid_sphere"]["radius_m"]
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

    plt.tight_layout()
    plt.show()


def plot_echo_spectrum_hz(trace: np.ndarray, sample_rate: float):
    n_fft = int(SIM_PARAMS["debug"].get("echo_fft_points", 16384))
    dt = 1.0 / sample_rate

    E = np.fft.fftshift(np.fft.fft(trace, n_fft))
    f = np.fft.fftshift(np.fft.fftfreq(n_fft, dt))

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
    plt.tight_layout()
    plt.show()

    # linear view
    plt.figure(figsize=(10, 4))
    plt.plot(f, mag_norm, "k-", linewidth=1.5)
    plt.xlabel("Frequency [Hz]", fontsize=38)
    plt.ylabel("Magnitude [linear, normalized]", fontsize=32)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)
    #plt.title("Echo spectrum in frequency domain (linear scale)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, x1)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.show()


def main():
    signal_mode = SIM_PARAMS["signal"]["mode"]
    signal, f0, sim_baseband_frequency = build_signal_from_params(SIM_PARAMS)

    if SIM_PARAMS["debug"].get("plot_incident", False):

        # Diagnostic: sample the incident signal on the exact simulation time grid
        # used to generate the echoes in simple_points.npz.
        results = np.load("simple_points.npz")
        t_sim = results["sample_time"]
        s_sim = signal.sample(t_sim, sim_baseband_frequency)

        fs_sim = 1.0 / np.mean(np.diff(t_sim))
        if f0 is not None:
            print(f"Simulation-grid incident sampling: fs={fs_sim:.1f} Hz, f0={f0:.1f} Hz, samples/cycle={fs_sim/f0:.3f}")

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
        print(f"Chirp duration: {chirp_duration_s*1e3:.2f} ms")
        print(f"Hydrophone trace duration: {hydro_duration_s*1e3:.2f} ms")

        # Second incident plot: reconstructed passband only, over full recording window
        plt.figure(figsize=(10, 4))
        plt.plot(t_sim * 1e3, s_sim_passband, "k-", linewidth=1.5)

        plt.xlabel("Time [ms]", fontsize=38)
        plt.ylabel("Amplitude [Pa]", fontsize=38)  # o "Normalized amplitude" si normalizas
        #plt.title("Incident signal", fontsize=38)

        plt.xticks(fontsize=34)
        plt.yticks(fontsize=34)

        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.xlim(t_sim[0] * 1e3, t_sim[-1] * 1e3)
        plt.tight_layout()
        plt.show()


        if SIM_PARAMS["debug"].get("plot_incident_spectrum", False):
            plot_incident_spectrum(s_sim, fs_sim)
            plot_incident_spectrum_hz(
                signal=signal,
                sample_rate=fs_sim,
                baseband_frequency=sim_baseband_frequency,
                sample_time=t_sim,
            )

    if SIM_PARAMS["debug"].get("plot_form_function_from_plugin_dump", False):
        plot_form_function_from_dump()

    if SIM_PARAMS["debug"].get("plot_echo_linear_pressure", True):
        results = np.load("simple_points.npz")
        t_echo = results["sample_time"]
        P_echo = results["pressure"]
        fs_echo = 1.0 / np.mean(np.diff(t_echo))

        ping_to_plot = 31   # cambia aquí el ping que quieras ver
        rx_idx = 0
        trace_echo_bb = P_echo[ping_to_plot, rx_idx, :]  # complejo en banda base
        trace_echo_pb = np.real(
            trace_echo_bb * np.exp(1j * 2.0 * np.pi * sim_baseband_frequency * t_echo)
        )

        plt.figure(figsize=(10, 4))
        plt.plot(t_echo * 1e3, trace_echo_pb, "k-", linewidth=1.2, label="passband view for visualization")
        plt.xlabel("Time [ms]", fontsize=38)
        plt.ylabel("Amplitude [Pa]", fontsize=38)
        plt.xticks(fontsize=34)
        plt.yticks(fontsize=34)
        #plt.title(f"Echo (passband view), ping {ping_to_plot}, rx {rx_idx}")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.legend(fontsize=30)
        plt.tight_layout()
        plt.show()


        # Incident signal on the same hydrophone time axis (full trace)-------------------------------------
        incident_bb_on_echo_grid = signal.sample(t_echo, sim_baseband_frequency)
        incident_pb_on_echo_grid = np.real(
            incident_bb_on_echo_grid * np.exp(1j * 2.0 * np.pi * sim_baseband_frequency * t_echo)
        )

        # Optional normalization for visual comparison
        inc_peak = np.max(np.abs(incident_pb_on_echo_grid))
        echo_peak = np.max(np.abs(trace_echo_pb))
        incident_plot = incident_pb_on_echo_grid / inc_peak if inc_peak > 0 else incident_pb_on_echo_grid
        echo_plot = trace_echo_pb / echo_peak if echo_peak > 0 else trace_echo_pb

        plt.figure(figsize=(10, 4))
        plt.plot(t_echo * 1e3, incident_plot, "b-", linewidth=1.0, alpha=0.9, label="incident passband (normalized)")
        plt.plot(t_echo * 1e3, echo_plot, "k-", linewidth=1.2, alpha=0.9, label="echo passband (normalized)")
        plt.xlabel("Time [ms]", fontsize=38)
        plt.ylabel("Amplitude", fontsize=38)
        plt.title("Incident and echo on full hydrophone trace (normalized for comparison)", fontsize=38)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.xlim(t_echo[0] * 1e3, t_echo[-1] * 1e3)
        plt.legend(fontsize=34)
        plt.tight_layout()
        plt.show()


        #With real amplitudes (no normalization):------------------------------
        plt.figure(figsize=(10, 4))
        plt.plot(t_echo * 1e3, incident_pb_on_echo_grid, "b-", linewidth=1.0, alpha=0.9, label="incident passband")
        plt.plot(t_echo * 1e3, trace_echo_pb, "k-", linewidth=1.2, alpha=0.9, label="echo passband")
        plt.xlabel("Time [ms]", fontsize=38)
        plt.ylabel("Amplitude", fontsize=38)
        plt.title("Incident and echo on full hydrophone trace (with real amplitudes)", fontsize=38)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.xlim(t_echo[0] * 1e3, t_echo[-1] * 1e3)
        plt.legend(fontsize=34)
        plt.tight_layout()
        plt.show()


        if SIM_PARAMS["debug"].get("plot_echo_spectrum", True):
            plot_echo_spectrum(trace_echo_pb, fs_echo)
            plot_echo_spectrum_hz(trace_echo_pb, fs_echo)


    # Theta diagnostic across pings (current simple_points_study geometry).
    results = np.load("simple_points.npz")
    ping_t = results["ping_start_time"]

    start_pos = np.array([0.0, 0.0, 0.0])
    speed = 1.5
    tx_offset = np.array([0.0, 1.2, 0.3])
    rx_offset = np.array([0.0, 1.2, 0.0])
    target = np.array([5.0, 40.0, 10.0])

    vehicle_pos = start_pos + np.column_stack([speed * ping_t, np.zeros_like(ping_t), np.zeros_like(ping_t)])
    tx_pos = vehicle_pos + tx_offset
    rx_pos = vehicle_pos + rx_offset

    inc = target[np.newaxis, :] - tx_pos
    sca = rx_pos - target[np.newaxis, :]

    cos_th = np.sum(inc * sca, axis=1) / (np.linalg.norm(inc, axis=1) * np.linalg.norm(sca, axis=1))
    cos_th = np.clip(cos_th, -1.0, 1.0)
    theta = np.arccos(cos_th)

    print("theta(rad) min/max:", float(theta.min()), float(theta.max()))
    print("theta(deg) min/max:", float(np.degrees(theta).min()), float(np.degrees(theta).max()))
    print("first 10 theta(deg):", np.degrees(theta[:10]))
    print("theta ping 14 [rad]:", float(theta[14]))

    fs = 1.0 / np.mean(np.diff(results["sample_time"]))
    Ns = len(results["sample_time"])
    gb = int(np.ceil(signal.duration * 1.1 * fs))
    nfft_sim = Ns + gb
    df_sim = fs / nfft_sim

    print("FFT diagnostic (sim):")
    print("  Ns:", Ns)
    print("  gb:", gb)
    print("  Nfft_sim = Ns + gb:", nfft_sim)
    print("  df_sim [Hz]:", df_sim)
    print("  n_fft in RigidSphereEcho.py:", 16384)
    print("  df_RigidSphereEcho [Hz]:", fs / 16384.0)


if __name__ == "__main__":
    main()