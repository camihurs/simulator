# SPDX-FileCopyrightText: openSTB contributors
# SPDX-License-Identifier: BSD-2-Clause-Patent

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openstb.simulator.plugin import loader
from shared_params import SIM_PARAMS
from signal_factory import build_signal_from_params


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
    signal, f0, sim_baseband_frequency = build_signal_from_params(SIM_PARAMS)

    if SIM_PARAMS["debug"].get("plot_incident", False):
        #t, s, sample_rate_plot, title = sample_for_plot(signal, signal_mode, f0)
        #plot_incident_signal(t, s, title)

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

        plt.figure(figsize=(10, 4))
        plt.plot(t_sim * 1e3, s_sim_passband, "c-", linewidth=1.0, label="reconstructed passband")
        plt.plot(t_sim * 1e3, np.abs(s_sim), "m--", linewidth=1.0, label="abs(baseband)")
        plt.plot(t_sim * 1e3, -np.abs(s_sim), "m--", linewidth=1.0)
        plt.xlabel("Time [ms]")
        plt.ylabel("Amplitude")
        plt.title("Incident chirp on simulation grid (passband view for visualization)")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.xlim(0.0, signal.duration * 1e3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # s_sim_real = np.real(s_sim)
        # s_sim_mag = np.abs(s_sim)

        # plt.figure(figsize=(10, 4))
        # plt.plot(t_sim * 1e3, s_sim_real, "g-", linewidth=1.0, label="real(s_sim)")
        # plt.plot(t_sim * 1e3, s_sim_mag, "m--", linewidth=1.0, label="abs(s_sim)")
        # plt.plot(t_sim * 1e3, -s_sim_mag, "m--", linewidth=1.0)
        # plt.xlabel("Time [ms]")
        # plt.ylabel("Amplitude")
        # plt.title("Incident signal on simulation grid: real part and magnitude envelope")
        # plt.grid(True, alpha=0.3)
        # plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # plt.figure(figsize=(10, 4))
        # plt.plot(t_sim * 1e3, np.real(s_sim), "g-", linewidth=1.2)
        # plt.xlabel("Time [ms]")
        # plt.ylabel("Amplitude")
        # plt.title("Incident signal sampled on simulation grid (same grid as echo)")
        # plt.grid(True, alpha=0.3)
        # plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        # plt.tight_layout()
        # plt.show()

        if SIM_PARAMS["debug"].get("plot_incident_spectrum", False):
            #plot_incident_spectrum(s, sample_rate_plot)
            plot_incident_spectrum(s_sim, fs_sim)

    if SIM_PARAMS["debug"].get("plot_form_function_from_plugin_dump", False):
        plot_form_function_from_dump()

    if SIM_PARAMS["debug"].get("plot_echo_linear_pressure", True):
        results = np.load("simple_points.npz")
        t_echo = results["sample_time"]
        P_echo = results["pressure"]

        ping_to_plot = 5   # cambia aquí el ping que quieras ver
        rx_idx = 0
        trace_echo_bb = P_echo[ping_to_plot, rx_idx, :]  # complejo en banda base
        trace_echo_pb = np.real(
            trace_echo_bb * np.exp(1j * 2.0 * np.pi * sim_baseband_frequency * t_echo)
        )

        #Uncomment the following lines to normalize the echo trace to compare with the paper.
        #peak = np.max(np.abs(trace_echo_pb))
        #trace_echo_pb_norm = trace_echo_pb / peak if peak > 0 else trace_echo_pb

        plt.figure(figsize=(10, 4))
        plt.plot(t_echo, trace_echo_pb, "k-", linewidth=1.2, label="echo passband (viz)")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Echo (passband view), ping {ping_to_plot}, rx {rx_idx}")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        # trace_echo = np.real(P_echo[ping_to_plot, rx_idx, :])
        # peak = np.max(np.abs(trace_echo))
        # trace_echo_norm = trace_echo / peak if peak > 0 else trace_echo

        # plt.figure(figsize=(10, 4))
        # plt.plot(t_echo, trace_echo_norm, "k-", linewidth=1.2)
        # plt.xlabel("Time [s]")
        # plt.ylabel("Normalized amplitude (signed)")
        # plt.title(f"Echo (signed, normalized), ping {ping_to_plot}, rx {rx_idx}")
        # plt.grid(True, alpha=0.3)
        # plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
        # plt.tight_layout()
        # plt.show()

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