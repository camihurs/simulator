import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, eval_legendre


# ============================================================================
# STEP 1: Define Physical Parameters (edit these values directly)
# ============================================================================
N_TERMS = 80

# Water properties
C_WATER = 1480.0          # m/s
RHO_WATER = 1000.0        # kg/m^3

# Seabed properties
C_SEABED = 1600.0         # m/s
RHO_SEABED = 1800.0       # kg/m^3

# Rigid sphere and geometry
SPHERE_RADIUS = 0.25      # m
SLANT_RANGE = 10.0        # m
GRAZING_ANGLE_DEG = 20.0  # degrees

# Signal definition
# Supported: "sine", "chirp", "ricker"
SIGNAL_TYPE = "chirp"

# For sine/ricker
K0A = 15.0
N_CYCLES = 2

# For chirp
CHIRP_F_START = 30e3      # Hz
CHIRP_F_END = 150e3       # Hz
CHIRP_DURATION = 5e-3     # s

# FFT and sampling
FFT_POINTS = 16384
SAMPLE_RATE_FACTOR = 100

# Plot limits
KA_XMAX = 30
FORM_FUNCTION_XMAX = 14


# ============================================================================
# Core physics helpers
# ============================================================================
def hankel2_spherical(n, x):
    return spherical_jn(n, x) - 1j * spherical_yn(n, x)


def reflection_coefficient(theta_i, freqs, rho1, rho2, c1, c2):
    """Fresnel reflection coefficient for water/seabed interface."""
    freqs_safe = np.where(freqs == 0.0, 1e-15, freqs)

    rho = rho2 / rho1
    k2 = 2 * np.pi * freqs_safe / c2
    k1 = 2 * np.pi * freqs_safe / c1
    kappa = k2 / k1

    sqrt_term = np.lib.scimath.sqrt(kappa**2 - np.cos(theta_i) ** 2)
    gamma = (rho * np.sin(theta_i) - sqrt_term) / (rho * np.sin(theta_i) + sqrt_term)
    return gamma


def modes_rigid(n, k, freqs, theta, flag, a, r, field="near-field"):
    """
    flag=0: monostatic (theta = pi)
    flag=1: bistatic   (theta = theta_i)
    """
    fn_array = np.zeros(freqs.size, dtype=complex)

    for i in range(len(freqs)):
        j_n = spherical_jn(n, k[i] * a, derivative=True)
        y_n = spherical_yn(n, k[i] * a, derivative=True)

        if np.abs(y_n) < 1e-15:
            eta_n = 0.0
        else:
            eta_n = np.arctan(-j_n / y_n)

        if flag == 0:
            if field == "near-field":
                fn_array[i] = (
                    ((-1j) ** (n + 1))
                    * hankel2_spherical(n, k[i] * r)
                    * eval_legendre(n, np.cos(theta))
                    * (2 * n + 1)
                    * np.sin(eta_n)
                    * np.exp(1j * eta_n)
                )
            else:
                fn_array[i] = (
                    eval_legendre(n, np.cos(theta))
                    * (2 * n + 1)
                    * np.sin(eta_n)
                    * np.exp(1j * eta_n)
                )
        else:
            fn_array[i] = (
                ((-1j) ** (n + 1))
                * hankel2_spherical(n, k[i] * r)
                * eval_legendre(n, -np.cos(2 * theta))
                * (2 * n + 1)
                * np.sin(eta_n)
                * np.exp(1j * eta_n)
            )

    return np.nan_to_num(fn_array, nan=0.0, posinf=0.0, neginf=0.0)


def form_function_above_seabed(f1, f2, gamma_11, k, a, theta_i):
    """Equation (3-term) for rigid sphere above seabed."""
    gamma_sq = gamma_11**2
    term1 = f1
    term2 = 2 * gamma_11 * f2 * np.exp(2j * k * a * np.sin(theta_i))
    term3 = gamma_sq * f1 * np.exp(4j * k * a * np.sin(theta_i))
    f_above = term1 + term2 + term3
    return np.nan_to_num(f_above, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# Signal helpers
# ============================================================================
def build_incident_signal():
    if SIGNAL_TYPE in ("sine", "ricker"):
        f0 = K0A * C_WATER / (2 * np.pi * SPHERE_RADIUS)
        pulse_duration = N_CYCLES / f0
        sample_rate = SAMPLE_RATE_FACTOR * f0

        t_max = 3 * pulse_duration
        t = np.arange(0, t_max, 1 / sample_rate)

        if SIGNAL_TYPE == "sine":
            signal = np.where(t < pulse_duration, np.sin(2 * np.pi * f0 * t), 0.0)
        else:
            t_centered = t - pulse_duration / 2
            signal = (1 - 2 * (np.pi * f0 * t_centered) ** 2) * np.exp(
                -(np.pi * f0 * t_centered) ** 2
            )

        label = f"{SIGNAL_TYPE} (f0={f0:.1f} Hz)"
        return signal, t, sample_rate, f0, label

    if SIGNAL_TYPE == "chirp":
        sample_rate = max(2.5 * CHIRP_F_END, 1e5)
        t = np.arange(0, CHIRP_DURATION, 1 / sample_rate)
        k_chirp = (CHIRP_F_END - CHIRP_F_START) / CHIRP_DURATION
        phase = 2 * np.pi * (CHIRP_F_START * t + 0.5 * k_chirp * t**2)
        signal = np.sin(phase)

        f0 = 0.5 * (CHIRP_F_START + CHIRP_F_END)
        label = f"chirp ({CHIRP_F_START/1e3:.1f}-{CHIRP_F_END/1e3:.1f} kHz)"
        return signal, t, sample_rate, f0, label

    raise ValueError("SIGNAL_TYPE must be one of: 'sine', 'chirp', 'ricker'")


def build_bilateral_form_function(f_positive):
    f_conj = np.conjugate(f_positive[1:])
    f_conj_inv = f_conj[::-1]
    return np.concatenate([f_positive[:-1], f_conj_inv])


# ============================================================================
# Plot helpers
# ============================================================================
def plot_incident_signal_time(t, signal, label):
    plt.figure(figsize=(10, 4))
    plt.plot(t * 1e3, signal, "b-", linewidth=1.4)
    plt.xlabel("Time [ms]")
    plt.ylabel("Amplitude")
    plt.title(f"Incident signal in time: {label}")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.0, color="k", linewidth=0.5)
    plt.tight_layout()


def plot_incident_signal_frequency(freq_positive, spectrum_positive, a):
    ka_positive = (2 * np.pi * freq_positive / C_WATER) * a

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(ka_positive, np.abs(spectrum_positive), "b-", linewidth=1.3)
    ax1.set_xlabel("ka")
    ax1.set_ylabel("|S(ka)|")
    ax1.set_title("Incident signal spectrum - magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, KA_XMAX)

    phase = np.mod(np.angle(spectrum_positive), 2 * np.pi)
    ax2.plot(ka_positive, phase, "r-", linewidth=1.3)
    ax2.set_xlabel("ka")
    ax2.set_ylabel("Phase [rad]")
    ax2.set_title("Incident signal spectrum - phase")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, KA_XMAX)
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])

    plt.tight_layout()


def plot_form_function(freq_positive, f_above, a, grazing_angle_deg):
    ka = (2 * np.pi * freq_positive / C_WATER) * a

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(ka, np.abs(f_above), "b-", linewidth=1.3)
    ax1.set_xlabel("ka")
    ax1.set_ylabel("|f(ka)|")
    ax1.set_title(
        f"Rigid sphere above seabed - form function magnitude ({grazing_angle_deg:.1f} deg)"
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, FORM_FUNCTION_XMAX)

    phase = np.mod(np.angle(f_above), 2 * np.pi)
    ax2.plot(ka, phase, "r-", linewidth=1.3)
    ax2.set_xlabel("ka")
    ax2.set_ylabel("arg[f(ka)] [rad]")
    ax2.set_title("Rigid sphere above seabed - form function phase")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, FORM_FUNCTION_XMAX)
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])

    plt.tight_layout()


def plot_echo_time_and_frequency(echo, sample_rate):
    n = len(echo)

    echo_shifted = np.fft.fftshift(np.real(echo))
    t_shifted = (np.arange(n) - n // 2) / sample_rate

    echo_fft = np.fft.fft(echo_shifted)
    freq = np.fft.fftfreq(n, d=1 / sample_rate)
    pos = freq >= 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(t_shifted * 1e3, echo_shifted, "b-", linewidth=1.2)
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Scattered echo in time")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.0, color="k", linewidth=0.5)

    ax2.plot(freq[pos] / 1e3, np.abs(echo_fft[pos]), "m-", linewidth=1.2)
    ax2.set_xlabel("Frequency [kHz]")
    ax2.set_ylabel("|E(f)|")
    ax2.set_title("Scattered echo spectrum magnitude")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


# ============================================================================
# Main
# ============================================================================
def main():
    print("Rigid sphere above seabed")
    print("Script style aligned with free_field/RigidSphereEcho.py")

    # ========================================================================
    # STEP 2: Generate Incident Signal (time domain)
    # ========================================================================
    theta_i = np.deg2rad(90.0 - GRAZING_ANGLE_DEG)

    incident_signal, t, sample_rate, f0, signal_label = build_incident_signal()
    dt = 1 / sample_rate

    print(f"Signal type: {SIGNAL_TYPE}")
    print(f"Sample rate: {sample_rate:.2f} Hz")
    print(f"Center frequency (reference): {f0:.2f} Hz")
    print(f"Wavelength: {C_WATER / f0:.4f} m")
    print(f"Period: {(1 / f0) * 1e3:.4f} ms")
    print(f"Sphere radius: {SPHERE_RADIUS:.3f} m")
    print(f"Grazing angle: {GRAZING_ANGLE_DEG:.2f} deg")
    print(f"Incidence angle: {np.rad2deg(theta_i):.2f} deg")
    print(f"Slant range: {SLANT_RANGE:.2f} m")

    print("\nSignal generated:")
    print(f"  - Samples: {len(t)}")
    print(f"  - Duration: {t[-1] * 1e3:.2f} ms")

    # ========================================================================
    # STEP 3: Calculate Spectrum of Incident Signal
    # ========================================================================
    spectrum = np.fft.fft(incident_signal, FFT_POINTS) * dt
    spectrum *= 2

    freq = np.fft.fftfreq(FFT_POINTS, dt)
    mask_positive = freq >= 0
    freq_positive = freq[mask_positive]
    spectrum_positive = spectrum[mask_positive]

    print("\nSpectrum computed:")
    print(f"  - FFT points: {FFT_POINTS}")
    print(f"  - Frequency resolution: {freq_positive[1] - freq_positive[0]:.2f} Hz")
    print(f"  - Max frequency: {freq_positive[-1] / 1e3:.2f} kHz")

    # ========================================================================
    # STEP 4: Calculate Form Function f(ka) for Above Seabed
    # ========================================================================
    k = 2 * np.pi * np.abs(freq_positive) / C_WATER
    ka_positive = k * SPHERE_RADIUS

    gamma_11 = reflection_coefficient(
        theta_i,
        np.abs(freq_positive),
        RHO_WATER,
        RHO_SEABED,
        C_WATER,
        C_SEABED,
    )

    print("Computing monostatic form function f1...")
    f1 = np.zeros(freq_positive.size, dtype=complex)
    for n in range(N_TERMS):
        f1 += modes_rigid(n, k, freq_positive, np.pi, 0, SPHERE_RADIUS, SLANT_RANGE)

    print("Computing bistatic form function f2...")
    f2 = np.zeros(freq_positive.size, dtype=complex)
    for n in range(N_TERMS):
        f2 += modes_rigid(n, k, freq_positive, theta_i, 1, SPHERE_RADIUS, SLANT_RANGE)

    f_above = form_function_above_seabed(f1, f2, gamma_11, k, SPHERE_RADIUS, theta_i)

    print("\nForm function computed:")
    print(f"  - Number of terms: {N_TERMS}")
    print(f"  - ka min/max: {ka_positive.min():.6f} / {ka_positive.max():.6f}")

    # ========================================================================
    # STEP 5: Compute Scattered Echo
    # ========================================================================
    f_bilateral = build_bilateral_form_function(f_above)
    echo_spectrum = spectrum * f_bilateral
    echo = np.fft.ifft(echo_spectrum, FFT_POINTS)
    echo = echo / (np.max(np.abs(echo)) + 1e-15)

    print("\nScattered echo computed:")
    print(f"  - Samples: {len(echo)}")
    print(f"  - Finite ratio: {np.isfinite(echo).mean():.3f}")

    # ========================================================================
    # Visualize requested outputs
    # ========================================================================
    plot_incident_signal_time(t, incident_signal, signal_label)
    plot_incident_signal_frequency(freq_positive, spectrum_positive, SPHERE_RADIUS)
    plot_form_function(freq_positive, f_above, SPHERE_RADIUS, GRAZING_ANGLE_DEG)
    plot_echo_time_and_frequency(echo, sample_rate)

    plt.show()


if __name__ == "__main__":
    main()
