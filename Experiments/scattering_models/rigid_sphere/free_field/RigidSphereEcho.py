from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn


# Versión robusta para evitar NaN/Inf cerca de ka=0
def compute_form_function(ka_values, theta=np.pi, ka_eps=1e-8):
    """
    Compute form function f(ka) for rigid sphere (robust near ka=0).
    """
    ka_values = np.asarray(ka_values, dtype=float)
    ka_safe = np.maximum(ka_values, ka_eps)

    f = np.zeros_like(ka_safe, dtype=np.complex128)
    cos_theta = np.cos(theta)

    for n in range(N_terms):
        jn_prime = spherical_jn(n, ka_safe, derivative=True)
        yn_prime = spherical_yn(n, ka_safe, derivative=True)

        # Robust equivalent of arctan(jn_prime / -yn_prime)
        eta_n = np.arctan2(jn_prime, -yn_prime)
        eta_n = np.nan_to_num(eta_n, nan=0.0, posinf=0.0, neginf=0.0)

        Pn = eval_legendre(n, cos_theta)

        coeff_n = (2 * n + 1) * np.sin(eta_n) * np.exp(1j * eta_n)
        coeff_n = np.nan_to_num(coeff_n, nan=0.0, posinf=0.0, neginf=0.0)

        f += coeff_n * Pn

    f *= 2.0 / ka_safe
    f = -f
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return f


# def compute_form_function(ka_values, theta=np.pi):
#     """
#     Compute form function f(ka) for rigid sphere.

#     Parameters:
#     - ka_values: array of ka values
#     - theta: scattering angle (π for backscattering)

#     Returns:
#     - f(ka): complex form function
#     """

#     f = np.zeros_like(ka_values, dtype=complex)
#     cos_theta = np.cos(theta)

#     for n in range(N_terms):
#         # Derivatives of spherical Bessel functions
#         jn_prime = spherical_jn(n, ka_values, derivative=True)
#         yn_prime = spherical_yn(n, ka_values, derivative=True)

#         # Phase shift
#         eta_n = np.arctan(jn_prime / -yn_prime)

#         # Legendre polynomial
#         Pn = eval_legendre(n, cos_theta)

#         # Add term to sum
#         f += (2*n + 1) * Pn * np.sin(eta_n) * np.exp(1j * eta_n)

#     f = f * (2 / ka_values)
#     return -f


# ============================================================================
# STEP 1: Define Physical Parameters
# ============================================================================
N_terms = 80  # Number of terms in series (same as your original code)

# Physical constants
c = 1480.0  # Sound speed [m/s]
a = 0.25  # Sphere radius [m]

# Signal parameters (from paper: k0a = 15.0, 2 cycles)
k0a = 15.0
f0 = k0a * c / (2 * np.pi * a)  # Center frequency [Hz]
n_cycles = 2

print(f"Center frequency: f0 = {f0:.1f} Hz")
print(f"Wavelength: λ = {c / f0:.3f} m")
print(f"Period: T = {1 / f0 * 1e3:.3f} ms")
print(f"Pulse duration: {n_cycles / f0 * 1e3:.3f} ms")


# ============================================================================
# STEP 2: Generate Incident Signal (time domain)
# ============================================================================

# Time parameters
T_pulse = n_cycles / f0  # Pulse duration [s]
sample_rate = 10 * f0  # Sampling rate [Hz] (10x Nyquist)
dt = 1 / sample_rate  # Time step [s]

# Create time vector (start at t=0)
t_max = 3 * T_pulse  # Extended for visualization
t = np.arange(0, t_max, dt)  # Time vector [s]

# Generate 2-cycle truncated sinusoid
incident_signal = np.where(t < T_pulse, np.sin(2 * np.pi * f0 * t), 0)


# ============================================================================
# Visualize incident signal
# ============================================================================
plt.figure(figsize=(10, 4))
plt.plot(t * 1e3, incident_signal, "b-", linewidth=1.5)
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.title(f"Incident Signal: {n_cycles}-cycle sinusoid at {f0:.1f} Hz", fontsize=38)
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, t_max * 1e3)
plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
plt.show()

print("\nSignal generated:")
print(f"  - Samples: {len(t)}")
print(f"  - Duration: {t_max * 1e3:.2f} ms")
print(f"  - Sample rate: {sample_rate:.1f} Hz")


# ============================================================================
# STEP 3: Calculate Spectrum of Incident Signal
# ============================================================================

# FFT parameters
# n_fft = 282880. Para comparar con SimulatorSTB
n_fft = 16384  # FFT points (Originally 8192). 16384 is to get a good resolution in the form function.
freq = np.fft.fftfreq(n_fft, dt)  # Frequency vector [Hz]

# Compute FFT
incident_fft = np.fft.fft(incident_signal, n_fft) * dt  # Scale by dt
incident_fft *= 2  # Factor of 2 (same as your original code)

# Keep only positive frequencies
positive_freq_mask = freq >= 0
freq_positive = freq[positive_freq_mask]
incident_fft_positive = incident_fft[positive_freq_mask]

# Convert frequency to wavenumber k = 2πf/c
k_positive = 2 * np.pi * freq_positive / c


# Calculate magnitude and phase
magnitude = np.abs(incident_fft_positive)
phase = np.angle(incident_fft_positive)
phase = np.mod(phase + 2 * np.pi, 2 * np.pi)  # Wrap to [0, 2π]


# ============================================================================
# Visualize spectrum
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Magnitude vs ka
ka_positive = k_positive * a
print(
    f"RigidSphereEcho ka min/max used for FF: {ka_positive.min():.6f} / {ka_positive.max():.6f}"
)
ax1.plot(ka_positive, magnitude, "b-", linewidth=1.5)
ax1.set_xlabel("ka")
ax1.set_ylabel("|g(ka)|")
ax1.set_title(f"Spectrum g(ka) of a {n_cycles}-cycle pulse with k₀a = {k0a}")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)

# Phase vs ka
ax2.plot(ka_positive, phase, "r-", linewidth=1.5)
ax2.set_xlabel("ka")
ax2.set_ylabel("Phase of g(ka) (radians)")
ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax2.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 2 * np.pi)

plt.tight_layout()
plt.show()

print("\nSpectrum computed:")
print(f"  - FFT points: {n_fft}")
print(f"  - Frequency resolution: {freq_positive[1]:.2f} Hz")
print(f"  - Max frequency: {freq_positive[-1] / 1e3:.1f} kHz")
print(f"  - k₀a value at f₀: {(2 * np.pi * f0 / c) * a:.2f}")


# ============================================================================
# STEP 4: Calculate Form Function f(ka)
# ============================================================================
# Compute f(ka) for the spectrum range
f_ka = compute_form_function(ka_positive[1:], theta=np.pi)  # Skip ka=0.
f_ka = np.concatenate([[0], f_ka])  # Add zero at ka=0

# ============================================================================
# Visualize form function (magnitude and phase)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Magnitude
ax1.plot(ka_positive, np.abs(f_ka), "b-", linewidth=1.5)
ax1.set_xlabel("ka")
ax1.set_ylabel("|f(ka)|")
ax1.set_title("Form Function for Rigid Sphere (Backscattering) - Magnitude")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 1.5)

# Phase
phase_f = np.angle(f_ka)
# Phase (wrap to [0, 2π] like in the paper)
phase_f = np.mod(phase_f, 2 * np.pi)  # Wrap to [0, 2π]

ax2.plot(ka_positive, phase_f, "r-", linewidth=1.5)
ax2.set_xlabel("ka")
ax2.set_ylabel("arg[f(ka)] (radians)")
ax2.set_title("Form Function for Rigid Sphere (Backscattering) - Phase")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 2 * np.pi)
ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax2.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])

plt.tight_layout()
plt.show()

print("\nForm function computed:")
print(f"  - Number of terms: {N_terms}")
print("  - Scattering angle: θ = π (backscattering)")


# ============================================================================
# STEP 4B: Compare form function vs SimulatorSTB plugin dump
# ============================================================================
dump_path = (
    Path(__file__).resolve().parents[3]
    / "simple_points_study"
    / "rigid_sphere_ff_debug.npz"
)

if dump_path.exists():
    d = np.load(dump_path)
    ka_sim = d["ka"]
    print(
        f"SimulatorSTB dump ka min/max used for FF: {ka_sim.min():.6f} / {ka_sim.max():.6f}"
    )
    ff_sim = d["ff_complex"]
    theta_sim = float(d["theta_sample_rad"])

    # Compute reference FF with same ka grid and same theta as plugin dump
    ff_ref = compute_form_function(ka_sim, theta=theta_sim)

    # Magnitude error (relative L2)
    num = np.linalg.norm(np.abs(ff_ref) - np.abs(ff_sim))
    den = np.linalg.norm(np.abs(ff_sim)) + 1e-15
    rel_mag_l2 = num / den

    # Phase error (wrapped) in radians
    phase_diff = np.angle(np.exp(1j * (np.angle(ff_ref) - np.angle(ff_sim))))
    rms_phase = np.sqrt(np.mean(phase_diff**2))

    print("\nFF comparison vs SimulatorSTB dump:")
    print(f"  dump path: {dump_path}")
    print(f"  theta_sim [rad]: {theta_sim:.10f}")
    print(f"  relative L2 magnitude error: {rel_mag_l2:.3e}")
    print(f"  RMS wrapped phase error [rad]: {rms_phase:.3e}")

    # Plot visual comparison (sorted by ka for readability)
    order = np.argsort(ka_sim)
    ka_s = ka_sim[order]
    ff_sim_s = ff_sim[order]
    ff_ref_s = ff_ref[order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(ka_s, np.abs(ff_sim_s), "k-", lw=1.2, label="SimulatorSTB dump")
    ax1.plot(ka_s, np.abs(ff_ref_s), "r--", lw=1.2, label="RigidSphereEcho recompute")
    ax1.set_xlabel("ka", fontsize=18)
    ax1.set_ylabel("|f(ka)|", fontsize=18)
    ax1.set_title("Form function magnitude: plugin vs reference", fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=18)

    ax2.plot(
        ka_s,
        np.mod(np.angle(ff_sim_s), 2 * np.pi),
        "k-",
        lw=1.2,
        label="SimulatorSTB dump",
    )
    ax2.plot(
        ka_s,
        np.mod(np.angle(ff_ref_s), 2 * np.pi),
        "r--",
        lw=1.2,
        label="RigidSphereEcho recompute",
    )
    ax2.set_xlabel("ka")
    ax2.set_ylabel("arg[f(ka)]")
    ax2.set_title("Form function phase: plugin vs reference")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()
else:
    print(f"\nPlugin dump not found: {dump_path}")


# ============================================================================
# STEP 4C: Compare input spectrum S (SimulatorSTB dump) vs RigidSphereEcho
# ============================================================================
if dump_path.exists():
    d = np.load(dump_path)
    f_sim = np.asarray(d["f_hz"], dtype=float)
    S_sim = np.asarray(d["S_in_complex"], dtype=np.complex128)

    # RigidSphereEcho spectrum using simulator-like convention:
    # no dt scaling, no factor 2, bilateral + fftshift
    f_rs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dt))
    S_rs = np.fft.fftshift(np.fft.fft(incident_signal, n_fft))

    # Interpolate RigidSphereEcho spectrum onto simulator frequency grid
    S_rs_interp = np.interp(
        f_sim, f_rs, np.real(S_rs), left=0.0, right=0.0
    ) + 1j * np.interp(f_sim, f_rs, np.imag(S_rs), left=0.0, right=0.0)

    # Relative magnitude error
    num = np.linalg.norm(np.abs(S_rs_interp) - np.abs(S_sim))
    den = np.linalg.norm(np.abs(S_sim)) + 1e-15
    rel_mag_l2_S = num / den

    # Phase error only where spectrum is significant
    mask = np.abs(S_sim) > (np.max(np.abs(S_sim)) * 1e-6)
    phase_diff = np.angle(
        np.exp(1j * (np.angle(S_rs_interp[mask]) - np.angle(S_sim[mask])))
    )
    rms_phase_S = np.sqrt(np.mean(phase_diff**2)) if np.any(mask) else np.nan

    print("\nS spectrum comparison (SimulatorSTB input vs RigidSphereEcho):")
    print(f"  relative L2 magnitude error: {rel_mag_l2_S:.3e}")
    print(f"  RMS wrapped phase error [rad]: {rms_phase_S:.3e}")


# ============================================================================
# STEP 5: Compute Scattered Echo
# ============================================================================

# Multiply spectrum with form function (in frequency domain)
product = incident_fft_positive * f_ka

# Inverse FFT to get scattered pulse in time domain
scattered_fft_raw = np.fft.ifft(product)
scattered_pulse = np.real(scattered_fft_raw)

# Normalize
scattered_pulse = scattered_pulse / np.max(np.abs(scattered_pulse))

# Shift to center the zero frequency component
scattered_pulse = np.fft.fftshift(scattered_pulse)


# --- Diagnostic branch: rebuild echo using S_in_complex dumped from SimulatorSTB ---
dump_path = (
    Path(__file__).resolve().parents[3]
    / "simple_points_study"
    / "rigid_sphere_ff_debug.npz"
)
if dump_path.exists():
    d2 = np.load(dump_path)
    f_sim = np.asarray(
        d2["f_hz"], dtype=float
    )  # shifted frequency grid used in simulator
    S_sim = np.asarray(
        d2["S_in_complex"], dtype=np.complex128
    )  # actual input spectrum seen by plugin
    theta_sim = float(d2["theta_sample_rad"])

    ka_sim_full = np.abs(2.0 * np.pi * f_sim * a / c)
    ff_sim_full = compute_form_function(ka_sim_full, theta=theta_sim)

    E_sim = S_sim * ff_sim_full
    echo_from_dumpS = np.real(np.fft.ifft(np.fft.ifftshift(E_sim)))

    # normalize + center for shape comparison
    echo_from_dumpS = echo_from_dumpS / (np.max(np.abs(echo_from_dumpS)) + 1e-15)
    echo_from_dumpS = np.fft.fftshift(echo_from_dumpS)

    # build time axis from simulator frequency grid spacing
    Nsim = len(f_sim)
    f_sorted = np.sort(f_sim)
    df_sim = np.median(np.diff(f_sorted))
    fs_sim = Nsim * df_sim
    dt_sim = 1.0 / fs_sim
    t_dumpS = (np.arange(Nsim) - Nsim // 2) * dt_sim
    tau_dumpS = t_dumpS * c / a

    print("echo_from_dumpS finite ratio:", np.isfinite(echo_from_dumpS).mean())
    print("echo_from_dumpS max abs:", np.max(np.abs(echo_from_dumpS)))


# Create proper time axis
n_samples = len(scattered_pulse)
dk = ka_positive[1] - ka_positive[0]  # Step in ka
dt_scattered = 2 * np.pi / (n_samples * dk * c / a)  # Time step
t_scattered = np.arange(-n_samples // 2, n_samples // 2) * dt_scattered

print("finite ratio scattered_pulse:", np.isfinite(scattered_pulse).mean())
print("max abs scattered_pulse:", np.max(np.abs(scattered_pulse)))
imax = int(np.argmax(np.abs(scattered_pulse)))
print("peak index:", imax)
print("peak time [ms]:", t_scattered[imax] * 1e3)

# Normalized time τ = tc/a (for comparison with paper)
tau = t_scattered * c / a

# ============================================================================
# Visualize: Physical units
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Physical time (seconds)
ax1.plot(
    t_scattered * 1e3, scattered_pulse, "b-", linewidth=1.5, label="Standalone script"
)
ax1.set_xlabel("Time [ms]", fontsize=30)
ax1.set_ylabel("Amplitude (norm.)", fontsize=30)
ax1.tick_params(axis="x", labelsize=30)
ax1.tick_params(axis="y", labelsize=30)
# ax1.set_title('Scattered Pulse - Physical Units')
ax1.grid(True, linestyle="--", alpha=0.7)
ax1.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
ax1.legend(fontsize=26)

# ax1.plot(t_sim_like * 1e3, echo_sim_like, 'k--', linewidth=1.2, label='Simulator-like pipeline')
# ax1.legend()
if dump_path.exists():
    ax1.plot(
        t_dumpS * 1e3, echo_from_dumpS, "k--", linewidth=1.2, label="Simulator pipeline"
    )
    ax1.legend(fontsize=26)

ax1.set_ylim(-1.1, 1.1)
ax1.set_xlim(-1.5, 1.5)


# Plot 2: Normalized time τ (comparison with paper)
ax2.plot(tau, scattered_pulse, "b-", linewidth=1.5, label="Standalone script")
ax2.set_xlabel("τ (normalized time: tc/a)", fontsize=30)
ax2.set_ylabel("Ψ(τ) Amplitude (norm.)", fontsize=30)
ax2.tick_params(axis="x", labelsize=30)
ax2.tick_params(axis="y", labelsize=30)
# ax2.set_title('Scattered Pulse - Normalized Units (Fig. 6 from Paper)')
ax2.grid(True, linestyle="--", alpha=0.7)

# ax2.plot(tau_sim_like, echo_sim_like, 'k--', linewidth=1.2, label='Simulator-like pipeline')
# ax2.legend()
if dump_path.exists():
    ax2.plot(
        tau_dumpS, echo_from_dumpS, "k--", linewidth=1.2, label="Simulator pipeline"
    )
    ax2.legend(fontsize=26)

ax2.set_xlim(-3, 8)
ax2.set_ylim(-1.1, 1.1)
ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
ax2.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
ax2.legend(fontsize=26)

plt.tight_layout()
plt.show()

print("\nScattered pulse computed:")
print(f"  - Samples: {len(scattered_pulse)}")
print(f"  - τ range: [{tau[0]:.2f}, {tau[-1]:.2f}]")
print(
    f"  - Physical time range: [{t_scattered[0] * 1e3:.2f}, {t_scattered[-1] * 1e3:.2f}] ms"
)
print(f"  - Expected specular at: τ ≈ -2 (t ≈ {-2 * a / c * 1e3:.3f} ms)")
print(f"  - Expected creeping wave at: τ ≈ π (t ≈ {np.pi * a / c * 1e3:.3f} ms)")
