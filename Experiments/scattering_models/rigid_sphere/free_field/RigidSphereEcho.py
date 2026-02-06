import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn, eval_legendre


def compute_form_function(ka_values, theta=np.pi):
    """
    Compute form function f(ka) for rigid sphere.

    Parameters:
    - ka_values: array of ka values
    - theta: scattering angle (π for backscattering)

    Returns:
    - f(ka): complex form function
    """

    f = np.zeros_like(ka_values, dtype=complex)
    cos_theta = np.cos(theta)

    for n in range(N_terms):
        # Derivatives of spherical Bessel functions
        jn_prime = spherical_jn(n, ka_values, derivative=True)
        yn_prime = spherical_yn(n, ka_values, derivative=True)

        # Phase shift
        eta_n = np.arctan(jn_prime / -yn_prime)

        # Legendre polynomial
        Pn = eval_legendre(n, cos_theta)

        # Add term to sum
        f += (2*n + 1) * Pn * np.sin(eta_n) * np.exp(1j * eta_n)

    f = f * (2 / ka_values)
    return -f


# ============================================================================
# STEP 1: Define Physical Parameters
# ============================================================================
N_terms = 80  # Number of terms in series (same as your original code)

# Physical constants
c = 1480.0              # Sound speed [m/s]
a = 0.25                # Sphere radius [m]

# Signal parameters (from paper: k0a = 15.0, 2 cycles)
k0a = 15.0
f0 = k0a * c / (2 * np.pi * a)  # Center frequency [Hz]
n_cycles = 2

print(f"Center frequency: f0 = {f0:.1f} Hz")
print(f"Wavelength: λ = {c/f0:.3f} m")
print(f"Period: T = {1/f0*1e3:.3f} ms")
print(f"Pulse duration: {n_cycles/f0*1e3:.3f} ms")



# ============================================================================
# STEP 2: Generate Incident Signal (time domain)
# ============================================================================

# Time parameters
T_pulse = n_cycles / f0          # Pulse duration [s]
sample_rate = 100 * f0            # Sampling rate [Hz] (100x Nyquist)
dt = 1 / sample_rate             # Time step [s]

# Create time vector (start at t=0)
t_max = 3 * T_pulse              # Extended for visualization
t = np.arange(0, t_max, dt)      # Time vector [s]

# Generate 2-cycle truncated sinusoid
incident_signal = np.where(t < T_pulse, np.sin(2 * np.pi * f0 * t), 0)


# ============================================================================
# Visualize incident signal
# ============================================================================
plt.figure(figsize=(10, 4))
plt.plot(t * 1e3, incident_signal, 'b-', linewidth=1.5)
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
plt.title(f'Incident Signal: {n_cycles}-cycle sinusoid at {f0:.1f} Hz')
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, t_max*1e3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.show()

print(f"\nSignal generated:")
print(f"  - Samples: {len(t)}")
print(f"  - Duration: {t_max*1e3:.2f} ms")
print(f"  - Sample rate: {sample_rate:.1f} Hz")



# ============================================================================
# STEP 3: Calculate Spectrum of Incident Signal
# ============================================================================

# FFT parameters
n_fft = 16384                     # FFT points (Originally 8192). 16384 is to get a good resolution in the form function.
freq = np.fft.fftfreq(n_fft, dt) # Frequency vector [Hz]

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
phase = np.mod(phase + 2*np.pi, 2*np.pi)  # Wrap to [0, 2π]



# ============================================================================
# Visualize spectrum
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Magnitude vs ka
ka_positive = k_positive * a
ax1.plot(ka_positive, magnitude, 'b-', linewidth=1.5)
ax1.set_xlabel('ka')
ax1.set_ylabel('|g(ka)|')
ax1.set_title(f'Spectrum g(ka) of a {n_cycles}-cycle pulse with k₀a = {k0a}')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)

# Phase vs ka
ax2.plot(ka_positive, phase, 'r-', linewidth=1.5)
ax2.set_xlabel('ka')
ax2.set_ylabel('Phase of g(ka) (radians)')
ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax2.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 2*np.pi)

plt.tight_layout()
plt.show()

print(f"\nSpectrum computed:")
print(f"  - FFT points: {n_fft}")
print(f"  - Frequency resolution: {freq_positive[1]:.2f} Hz")
print(f"  - Max frequency: {freq_positive[-1]/1e3:.1f} kHz")
print(f"  - k₀a value at f₀: {(2*np.pi*f0/c)*a:.2f}")



# ============================================================================
# STEP 4: Calculate Form Function f(ka)
# ============================================================================
# Compute f(ka) for the spectrum range
f_ka = compute_form_function(ka_positive[1:], theta=np.pi)  # Skip ka=0
f_ka = np.concatenate([[0], f_ka])  # Add zero at ka=0

# ============================================================================
# Visualize form function (magnitude and phase)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Magnitude
ax1.plot(ka_positive, np.abs(f_ka), 'b-', linewidth=1.5)
ax1.set_xlabel('ka')
ax1.set_ylabel('|f(ka)|')
ax1.set_title('Form Function for Rigid Sphere (Backscattering) - Magnitude')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 14)
ax1.set_ylim(0, 1.5)

# Phase
phase_f = np.angle(f_ka)
# Phase (wrap to [0, 2π] like in the paper)
phase_f = np.mod(phase_f, 2*np.pi)  # Wrap to [0, 2π]

ax2.plot(ka_positive, phase_f, 'r-', linewidth=1.5)
ax2.set_xlabel('ka')
ax2.set_ylabel('arg[f(ka)] (radians)')
ax2.set_title('Form Function for Rigid Sphere (Backscattering) - Phase')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 2*np.pi)
ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax2.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

plt.tight_layout()
plt.show()

print(f"\nForm function computed:")
print(f"  - Number of terms: {N_terms}")
print(f"  - Scattering angle: θ = π (backscattering)")


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

# Create proper time axis
n_samples = len(scattered_pulse)
dk = ka_positive[1] - ka_positive[0]  # Step in ka
dt_scattered = 2*np.pi / (n_samples * dk * c / a)  # Time step
t_scattered = np.arange(-n_samples//2, n_samples//2) * dt_scattered

# Normalized time τ = tc/a (for comparison with paper)
tau = t_scattered * c / a

# ============================================================================
# Visualize: Physical units
# ============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Physical time (seconds)
ax1.plot(t_scattered * 1e3, scattered_pulse, 'b-', linewidth=1.5)
ax1.set_xlabel('Time [ms]')
ax1.set_ylabel('Normalized Amplitude')
ax1.set_title('Scattered Pulse - Physical Units')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.set_ylim(-1.1, 1.1)

# Plot 2: Normalized time τ (comparison with paper)
ax2.plot(tau, scattered_pulse, 'b-', linewidth=1.5)
ax2.set_xlabel('τ (normalized time: tc/a)')
ax2.set_ylabel('Ψ(τ) (normalized amplitude)')
ax2.set_title('Scattered Pulse - Normalized Units (Fig. 6 from Paper)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xlim(-3, 8)
ax2.set_ylim(-1.1, 1.1)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

print(f"\nScattered pulse computed:")
print(f"  - Samples: {len(scattered_pulse)}")
print(f"  - τ range: [{tau[0]:.2f}, {tau[-1]:.2f}]")
print(f"  - Physical time range: [{t_scattered[0]*1e3:.2f}, {t_scattered[-1]*1e3:.2f}] ms")
print(f"  - Expected specular at: τ ≈ -2 (t ≈ {-2*a/c*1e3:.3f} ms)")
print(f"  - Expected creeping wave at: τ ≈ π (t ≈ {np.pi*a/c*1e3:.3f} ms)")