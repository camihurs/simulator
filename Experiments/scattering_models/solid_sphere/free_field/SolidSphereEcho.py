import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn

warnings.filterwarnings("ignore", category=RuntimeWarning)


def hankel1_spherical(n, x):
    return spherical_jn(n, x) + 1j * spherical_yn(n, x)


def hankel1_sph_deriv(n, x):
    return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(
        n, x, derivative=True
    )


def modes_solid(n, k1, freqs, x, x1, x2, theta, rho1, rho2, r, field="near-field"):
    """
    Modal contribution for a solid elastic sphere.

    This is adapted from ModesSolid in AcousticScattering_Menu_FF_spheres.py.
    For validation against Hickling (1962), use field="far-field".
    For echo synthesis, use the near-field expression used by the original menu script.
    """
    fn_array = np.zeros(freqs.size, dtype=np.complex128)

    for i in range(len(freqs)):
        d11 = (rho1 / rho2) * (x2[i] ** 2) * hankel1_spherical(n, x[i])
        d12 = ((2 * n * (n + 1) - x2[i] ** 2) * spherical_jn(n, x1[i])) - (
            4 * x1[i] * spherical_jn(n, x1[i], derivative=True)
        )
        d13 = (
            2
            * n
            * (n + 1)
            * (x2[i] * spherical_jn(n, x2[i], derivative=True) - spherical_jn(n, x2[i]))
        )
        d21 = -x[i] * hankel1_sph_deriv(n, x[i])
        d22 = x1[i] * spherical_jn(n, x1[i], derivative=True)
        d23 = n * (n + 1) * spherical_jn(n, x2[i])
        d32 = 2 * (
            spherical_jn(n, x1[i]) - x1[i] * spherical_jn(n, x1[i], derivative=True)
        )
        d33 = 2 * x2[i] * spherical_jn(n, x2[i], derivative=True) + (
            (x2[i] ** 2 - 2 * n * (n + 1) + 2) * spherical_jn(n, x2[i])
        )
        d10 = -(rho1 / rho2) * (x2[i] ** 2) * spherical_jn(n, x[i])
        d20 = x[i] * spherical_jn(n, x[i], derivative=True)

        b_matrix = np.array([[d10, d12, d13], [d20, d22, d23], [0.0, d32, d33]])
        d_matrix = np.array([[d11, d12, d13], [d21, d22, d23], [0.0, d32, d33]])

        b_determinant = np.linalg.det(b_matrix)
        d_determinant = np.linalg.det(d_matrix)
        r_determinant = -b_determinant / d_determinant

        if not np.isfinite(r_determinant):
            r_determinant = 0.0

        if field == "near-field":
            fn_array[i] = (
                (1j**n)
                * (2 * n + 1)
                * r_determinant
                * hankel1_spherical(n, k1[i] * r)
                * eval_legendre(n, np.cos(theta))
            )
        else:
            fn_array[i] = ((-1) ** n) * r_determinant * (2 * n + 1)

    return np.nan_to_num(fn_array, nan=0.0, posinf=0.0, neginf=0.0)


def compute_modal_sum(freqs, c1, a, cd2, cs2, rho1, rho2, r, n_terms, field):
    freqs = np.asarray(freqs, dtype=float)
    k1 = 2 * np.pi * np.abs(freqs) / c1
    x = k1 * a
    x1 = (c1 / cd2) * x
    x2 = (c1 / cs2) * x

    f_sum = np.zeros(freqs.size, dtype=np.complex128)
    for n in range(n_terms):
        if n % 10 == 0:
            print(f"  mode {n}/{n_terms}")
        f_sum += modes_solid(
            n=n,
            k1=k1,
            freqs=freqs,
            x=x,
            x1=x1,
            x2=x2,
            theta=np.pi,
            rho1=rho1,
            rho2=rho2,
            r=r,
            field=field,
        )

    return np.nan_to_num(f_sum, nan=0.0, posinf=0.0, neginf=0.0), x


def compute_hickling_form_function(freqs, c1, a, cd2, cs2, rho1, rho2, r, n_terms):
    """
    Far-field form function used for comparison with Hickling (1962).
    """
    f_sum, ka = compute_modal_sum(
        freqs=freqs,
        c1=c1,
        a=a,
        cd2=cd2,
        cs2=cs2,
        rho1=rho1,
        rho2=rho2,
        r=r,
        n_terms=n_terms,
        field="far-field",
    )

    form_function = np.zeros_like(f_sum)
    nonzero = np.abs(ka) > 0
    form_function[nonzero] = (2 * f_sum[nonzero]) / (1j * ka[nonzero])
    return np.nan_to_num(form_function, nan=0.0, posinf=0.0, neginf=0.0), ka


def get_form_function_echo(f_positive):
    """
    Rebuild the bilateral spectrum convention used in the original menu script.
    """
    f_positive = np.nan_to_num(f_positive, nan=0.0, posinf=0.0, neginf=0.0)
    f_conj = np.conjugate(f_positive[1:])
    f_conj_inv = f_conj[::-1]
    return np.concatenate([f_positive[:-1], f_conj_inv])


# ============================================================================
# STEP 1: Define Physical Parameters
# ============================================================================

# Parameters from CheckSolidSphereFormFunction in AcousticScattering_Menu_FF_spheres.py.
reference_paper = (
    "Hickling, R. (1962). Analysis of echoes from a solid elastic sphere in water. "
    "Journal of the Acoustical Society of America, 34(10), 1582-1592."
)

medium1 = "Water"
material_sphere = "Beryllium"

rho1 = 1000.0  # Water density [kg/m^3]
c1 = 1410.0  # Water sound speed [m/s]
rho2 = 1870.0  # Beryllium density [kg/m^3]
cd2 = 12890.0  # Compressional wave speed in Beryllium [m/s]
cs2 = 8880.0  # Shear wave speed in Beryllium [m/s]
a = 0.25  # Sphere radius [m]
r = 1.73  # Slant range used in the original validation script [m]
n_terms = 80

xmin = 0.0
xmax = 30.0
form_function_points = 4096
fmin = xmin * c1 / (2 * np.pi * a)
fmax = xmax * c1 / (2 * np.pi * a)

print("Reference paper:")
print(f"  {reference_paper}")
print("\nSolid sphere parameters:")
print(f"  surrounding medium: {medium1}")
print(f"  sphere material: {material_sphere}")
print(f"  radius: {a:.3f} m")
print(f"  water sound speed: {c1:.1f} m/s")
print(f"  rho1/rho2: {rho1:.1f} / {rho2:.1f} kg/m^3")
print(f"  cd2/cs2: {cd2:.1f} / {cs2:.1f} m/s")
print(f"  modes: {n_terms}")


# ============================================================================
# STEP 2: Generate Incident Signal (time domain)
# ============================================================================

# First echo source: same style as RigidSphereEcho.py, a 2-cycle truncated sine.
k0a = 15.0
f0 = k0a * c1 / (2 * np.pi * a)
n_cycles = 2
T_pulse = n_cycles / f0
sample_rate = 10 * f0
dt = 1 / sample_rate
t_max = 3 * T_pulse
t = np.arange(0, t_max, dt)
incident_signal = np.where(t < T_pulse, np.sin(2 * np.pi * f0 * t), 0.0)

print("\nIncident signal:")
print(f"  center frequency: {f0:.1f} Hz")
print(f"  k0a: {k0a:.1f}")
print(f"  cycles: {n_cycles}")
print(f"  pulse duration: {T_pulse * 1e3:.3f} ms")
print(f"  sample rate: {sample_rate:.1f} Hz")

plt.figure(figsize=(10, 4))
plt.plot(t * 1e3, incident_signal, "k-", linewidth=1.5)
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude")
plt.title(f"Incident Signal: {n_cycles}-cycle sinusoid at {f0:.1f} Hz")
plt.grid(True, alpha=0.3)
plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
plt.tight_layout()
plt.show()


# ============================================================================
# STEP 3: Calculate Spectrum of Incident Signal
# ============================================================================

n_fft = 16384
freq = np.fft.fftfreq(n_fft, dt)
incident_fft = np.fft.fft(incident_signal, n_fft)

# The solid-sphere echo reconstruction follows the original menu script:
# keep the first n_fft//2 + 1 bins, including the Nyquist bin, then rebuild the
# negative-frequency half with conjugate symmetry.
freq_positive = np.abs(freq[: n_fft // 2 + 1])
incident_fft_positive = incident_fft[: n_fft // 2 + 1]
ka_positive = 2 * np.pi * freq_positive * a / c1

magnitude = np.abs(incident_fft_positive)
phase = np.mod(np.angle(incident_fft_positive), 2 * np.pi)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(ka_positive, magnitude, "k-", linewidth=1.5)
ax1.set_xlabel("ka")
ax1.set_ylabel("|g(ka)|")
ax1.set_title(f"Spectrum g(ka) of a {n_cycles}-cycle pulse with k0a = {k0a}")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)

ax2.plot(ka_positive, phase, "k-", linewidth=1.5)
ax2.set_xlabel("ka")
ax2.set_ylabel("Phase of g(ka) [rad]")
ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 2 * np.pi)
plt.tight_layout()
plt.show()

print("\nSpectrum computed:")
print(f"  FFT points: {n_fft}")
print(f"  frequency resolution: {freq_positive[1]:.2f} Hz")
print(
    f"  ka range for echo form function: {ka_positive.min():.4f}\
       / {ka_positive.max():.4f}"
)


# ============================================================================
# STEP 4: Calculate and Validate Far-Field Form Function
# ============================================================================

freq_validation = np.linspace(fmin, fmax, form_function_points)
print("\nComputing Hickling far-field form function for validation...")
print(f"  frequency step: {(fmax - fmin) / form_function_points:.3f} Hz")
f_hickling, ka_validation = compute_hickling_form_function(
    freqs=freq_validation,
    c1=c1,
    a=a,
    cd2=cd2,
    cs2=cs2,
    rho1=rho1,
    rho2=rho2,
    r=r,
    n_terms=n_terms,
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(ka_validation, np.abs(f_hickling), "k-", linewidth=1.5)
ax1.set_xlabel("ka")
ax1.set_ylabel("|f(ka)|")
ax1.set_title("Solid Beryllium Sphere Form Function - Hickling Far Field")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)

phase_hickling = np.mod(np.angle(f_hickling), 2 * np.pi)
ax2.plot(ka_validation, phase_hickling, "k-", linewidth=1.5)
ax2.set_xlabel("ka")
ax2.set_ylabel("arg[f(ka)] [rad]")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 2 * np.pi)
ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
plt.tight_layout()
plt.show()

print("\nValidation form function computed:")
print(f"  material: {material_sphere}")
print(f"  ka range: {ka_validation.min():.2f} / {ka_validation.max():.2f}")
print("  Compare this figure against Hickling (1962) solid Beryllium sphere results.")


# ============================================================================
# STEP 5: Compute Solid-Sphere Form Function for Echo Synthesis
# ============================================================================

print("\nComputing near-field modal response for echo synthesis...")
f_echo_positive, x_echo = compute_modal_sum(
    freqs=freq_positive,
    c1=c1,
    a=a,
    cd2=cd2,
    cs2=cs2,
    rho1=rho1,
    rho2=rho2,
    r=r,
    n_terms=n_terms,
    field="near-field",
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(x_echo, np.abs(f_echo_positive), "k-", linewidth=1.5)
ax1.set_xlabel("ka")
ax1.set_ylabel("|modal sum|")
ax1.set_title("Solid Beryllium Sphere Modal Response Used for Echo")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)

phase_echo = np.mod(np.angle(f_echo_positive), 2 * np.pi)
ax2.plot(x_echo, phase_echo, "k-", linewidth=1.5)
ax2.set_xlabel("ka")
ax2.set_ylabel("Phase [rad]")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 2 * np.pi)
plt.tight_layout()
plt.show()


# ============================================================================
# STEP 6: Compute Scattered Echo
# ============================================================================

target = "SolBerylliumSphere"
f_echo_full = get_form_function_echo(f_echo_positive)

# Keep the same convention as GetEchoFF in AcousticScattering_Menu_FF_spheres.py:
# non-rigid spherical targets multiply by the reversed rebuilt form-function array.
echo_spectrum = incident_fft * f_echo_full[::-1]
scattered_echo = np.fft.ifft(echo_spectrum, n_fft)
scattered_echo_normalized = scattered_echo / (np.max(np.abs(scattered_echo)) + 1e-15)

t_echo = np.arange(n_fft) / sample_rate
tau_echo = t_echo * c1 / a

print("\nScattered pulse computed:")
print(f"  target: {target}")
print(f"  samples: {len(scattered_echo_normalized)}")
print(f"  finite ratio: {np.isfinite(scattered_echo_normalized).mean():.3f}")
print(f"  max abs before normalization: {np.max(np.abs(scattered_echo)):.3e}")


# ============================================================================
# STEP 7: Visualize Echo
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(t_echo * 1e3, np.real(scattered_echo_normalized), "k-", linewidth=1.5)
ax1.set_xlabel("Time [ms]")
ax1.set_ylabel("Amplitude (norm.)")
ax1.set_title("Solid Beryllium Sphere Echo - Physical Time")
ax1.grid(True, linestyle="--", alpha=0.7)
ax1.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
ax1.set_xlim(0, 3.0)
ax1.set_ylim(-1.1, 1.1)

ax2.plot(tau_echo, np.real(scattered_echo_normalized), "k-", linewidth=1.5)
ax2.set_xlabel("Normalized time tc/a")
ax2.set_ylabel("Amplitude (norm.)")
ax2.set_title("Solid Beryllium Sphere Echo - Normalized Time")
ax2.grid(True, linestyle="--", alpha=0.7)
ax2.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
ax2.set_xlim(0, 18)
ax2.set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.show()
