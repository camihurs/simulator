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


def modes_solid(
    n,
    k1,
    freqs,
    x,
    x1,
    x2,
    theta,
    rho1,
    rho2,
    r,
    field="far-field",
    valid_mask=None,
):
    """
    Modal contribution for a solid elastic sphere.

    This is adapted from ModesSolid in AcousticScattering_Menu_FF_spheres.py,
    but vectorized over the frequency axis.
    """
    fn_array = np.zeros(freqs.size, dtype=np.complex128)
    if valid_mask is None:
        valid_mask = np.ones(freqs.shape, dtype=bool)
    if not np.any(valid_mask):
        return fn_array

    xv = x[valid_mask]
    x1v = x1[valid_mask]
    x2v = x2[valid_mask]
    k1v = k1[valid_mask]

    jn_x = spherical_jn(n, xv)
    jn_x_deriv = spherical_jn(n, xv, derivative=True)
    jn_x1 = spherical_jn(n, x1v)
    jn_x1_deriv = spherical_jn(n, x1v, derivative=True)
    jn_x2 = spherical_jn(n, x2v)
    jn_x2_deriv = spherical_jn(n, x2v, derivative=True)

    d11 = (rho1 / rho2) * (x2v**2) * hankel1_spherical(n, xv)
    d12 = ((2 * n * (n + 1) - x2v**2) * jn_x1) - (4 * x1v * jn_x1_deriv)
    d13 = 2 * n * (n + 1) * (x2v * jn_x2_deriv - jn_x2)
    d21 = -xv * hankel1_sph_deriv(n, xv)
    d22 = x1v * jn_x1_deriv
    d23 = n * (n + 1) * jn_x2
    d32 = 2 * (jn_x1 - x1v * jn_x1_deriv)
    d33 = 2 * x2v * jn_x2_deriv + ((x2v**2 - 2 * n * (n + 1) + 2) * jn_x2)
    d10 = -(rho1 / rho2) * (x2v**2) * jn_x
    d20 = xv * jn_x_deriv

    b_matrix = np.zeros((xv.size, 3, 3), dtype=np.complex128)
    b_matrix[:, 0, 0] = d10
    b_matrix[:, 0, 1] = d12
    b_matrix[:, 0, 2] = d13
    b_matrix[:, 1, 0] = d20
    b_matrix[:, 1, 1] = d22
    b_matrix[:, 1, 2] = d23
    b_matrix[:, 2, 1] = d32
    b_matrix[:, 2, 2] = d33

    d_matrix = np.zeros((xv.size, 3, 3), dtype=np.complex128)
    d_matrix[:, 0, 0] = d11
    d_matrix[:, 0, 1] = d12
    d_matrix[:, 0, 2] = d13
    d_matrix[:, 1, 0] = d21
    d_matrix[:, 1, 1] = d22
    d_matrix[:, 1, 2] = d23
    d_matrix[:, 2, 1] = d32
    d_matrix[:, 2, 2] = d33

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r_determinant = -np.linalg.det(b_matrix) / np.linalg.det(d_matrix)

    r_determinant = np.nan_to_num(r_determinant, nan=0.0, posinf=0.0, neginf=0.0)

    if field == "near-field":
        values = (
            (1j**n)
            * (2 * n + 1)
            * r_determinant
            * hankel1_spherical(n, k1v * r)
            * eval_legendre(n, np.cos(theta))
        )
    elif field == "far-field":
        values = ((-1) ** n) * r_determinant * (2 * n + 1)
    else:
        raise ValueError("field must be 'far-field' or 'near-field'")

    fn_array[valid_mask] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(fn_array, nan=0.0, posinf=0.0, neginf=0.0)


def compute_modal_sum(
    freqs,
    c1,
    a,
    cd2,
    cs2,
    rho1,
    rho2,
    r,
    n_terms,
    field,
    ka_eps=1e-8,
):
    freqs = np.asarray(freqs, dtype=float)
    k1 = 2 * np.pi * np.abs(freqs) / c1
    x_raw = k1 * a
    valid_mask = x_raw > ka_eps
    x = np.maximum(x_raw, ka_eps)
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
            valid_mask=valid_mask,
        )

    return np.nan_to_num(f_sum, nan=0.0, posinf=0.0, neginf=0.0), x_raw


def scale_far_field_form_function(f_sum, ka, ka_eps=1e-8):
    form_function = np.zeros_like(f_sum)
    nonzero = np.abs(ka) > ka_eps
    form_function[nonzero] = (2 * f_sum[nonzero]) / (1j * ka[nonzero])
    return np.nan_to_num(form_function, nan=0.0, posinf=0.0, neginf=0.0)


def compute_hickling_form_function(
    freqs, c1, a, cd2, cs2, rho1, rho2, r, n_terms, ka_eps=1e-8
):
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
        ka_eps=ka_eps,
    )

    return scale_far_field_form_function(f_sum, ka, ka_eps=ka_eps), ka


def compute_echo_response(
    freqs, c1, a, cd2, cs2, rho1, rho2, r, n_terms, field, ka_eps
):
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
        field=field,
        ka_eps=ka_eps,
    )
    if field == "far-field":
        return scale_far_field_form_function(f_sum, ka, ka_eps=ka_eps), ka
    return f_sum, ka


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

HICKLING_MATERIALS = {
    "Beryllium": {"rho2": 1870.0, "cd2": 12890.0, "cs2": 8880.0},
    "Fused silica": {"rho2": 2200.0, "cd2": 5968.0, "cs2": 3764.0},
    "Heavy silicate, flint glass": {"rho2": 3880.0, "cd2": 3980.0, "cs2": 2380.0},
    "Armco iron": {"rho2": 7700.0, "cd2": 5960.0, "cs2": 3240.0},
    "Monel metal": {"rho2": 8900.0, "cd2": 5350.0, "cs2": 2720.0},
    "Aluminum": {"rho2": 2700.0, "cd2": 6420.0, "cs2": 3040.0},
    "Yellow brass": {"rho2": 8600.0, "cd2": 4700.0, "cs2": 2110.0},
    "Lucite": {"rho2": 1180.0, "cd2": 2680.0, "cs2": 1100.0},
    "Lead": {"rho2": 11340.0, "cd2": 1960.0, "cs2": 690.0},
    "Ice": {"rho2": 917.0, "cd2": 2743.0, "cs2": 1433.0},
}

SOURCE_CASES = {
    "general": {
        "description": "General 2-cycle truncated sinusoid, not tied to a paper echo.",
        "n_cycles": 2,
        "k0a": 15.0,
        "integration_ka_bounds": None,
    },
    "hickling_fig16_max": {
        "description": "Hickling Fig. 16, maximum of |f|, Armco iron.",
        "n_cycles": 5,
        "k0a": 24.5,
        "integration_ka_bounds": (10.0, 40.0),
    },
    "hickling_fig16_min": {
        "description": "Hickling Fig. 16, minimum of |f|, Armco iron.",
        "n_cycles": 5,
        "k0a": 25.5,
        "integration_ka_bounds": (10.0, 40.0),
    },
    "hickling_fig17_max": {
        "description": "Hickling Fig. 17, maximum of |f|, Armco iron.",
        "n_cycles": 25,
        "k0a": 24.5,
        "integration_ka_bounds": (15.0, 35.0),
    },
    "hickling_fig17_min": {
        "description": "Hickling Fig. 17, minimum of |f|, Armco iron.",
        "n_cycles": 25,
        "k0a": 25.5,
        "integration_ka_bounds": (15.0, 35.0),
    },
    "hickling_fig18_max": {
        "description": "Hickling Fig. 18, maximum of |f|, Armco iron.",
        "n_cycles": 50,
        "k0a": 24.5,
        "integration_ka_bounds": (15.0, 35.0),
    },
    "hickling_fig18_min": {
        "description": "Hickling Fig. 18, minimum of |f|, Armco iron.",
        "n_cycles": 50,
        "k0a": 25.5,
        "integration_ka_bounds": (15.0, 35.0),
    },
}

medium1 = "Water"
selected_material = "Armco iron"
if selected_material not in HICKLING_MATERIALS:
    raise ValueError(
        "selected_material must be one of: " + ", ".join(sorted(HICKLING_MATERIALS))
    )

material_sphere = selected_material
material_properties = HICKLING_MATERIALS[selected_material]

rho1 = 1000.0  # Water density [kg/m^3]
c1 = 1410.0  # Water sound speed [m/s]
rho2 = material_properties["rho2"]  # Sphere density [kg/m^3]
cd2 = material_properties["cd2"]  # Compressional wave speed in the sphere [m/s]
cs2 = material_properties["cs2"]  # Shear wave speed in the sphere [m/s]
a = 0.25  # Sphere radius [m]
r = 1.73  # Slant range used in the original validation script [m]
n_terms = 80
ka_eps = 1e-8
echo_field = "far-field"  # Options: "far-field" or "near-field".
selected_source_case = "hickling_fig16_max"
if selected_source_case not in SOURCE_CASES:
    raise ValueError(
        "selected_source_case must be one of: " + ", ".join(sorted(SOURCE_CASES))
    )
source_case = SOURCE_CASES[selected_source_case]
is_hickling_source_case = selected_source_case.startswith("hickling")
if is_hickling_source_case and selected_material != "Armco iron":
    warnings.warn(
        "Hickling Figs. 16-18 use Armco iron. "
        f"Current selected_material is {selected_material!r}.",
        UserWarning,
    )

# Observation: the far-field form function is validated against Hickling (1962).
# The far-field echo synthesis is still pending direct validation against a paper echo.

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
print(f"  echo field: {echo_field}")
print(f"  source case: {selected_source_case}")
print(f"  source description: {source_case['description']}")


# ============================================================================
# STEP 2: Generate Incident Signal (time domain)
# ============================================================================

# The "general" case preserves the first source used in this script. The Hickling
# cases reproduce the truncated sinusoid parameters from Figs. 16-18.
k0a = source_case["k0a"]
f0 = k0a * c1 / (2 * np.pi * a)
n_cycles = source_case["n_cycles"]
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
source_plot_ka_max = 30
if source_case["integration_ka_bounds"] is None:
    print("  integration ka bounds: full FFT positive-frequency range")
else:
    ka_min, ka_max = source_case["integration_ka_bounds"]
    print(f"  integration ka bounds: {ka_min:g}-{ka_max:g}")
    source_plot_ka_max = max(source_plot_ka_max, ka_max + 5)

if is_hickling_source_case:
    source_time_centered = t - (T_pulse / 2)
    source_tau_centered = source_time_centered * c1 / a
    source_tau_complete = t * c1 / a
    half_duration_tau = (T_pulse * c1 / a) / 2
    half_duration_ms = (T_pulse * 1e3) / 2

    hickling_tau_xlim = (
        -max(1.0, 1.15 * half_duration_tau),
        max(1.0, 1.15 * half_duration_tau),
    )
    hickling_ms_xlim = (
        -1.15 * half_duration_ms,
        1.15 * half_duration_ms,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(source_tau_centered, incident_signal, "k-", linewidth=1.5)
    ax1.set_xlabel(r"$\tau - R$")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Incident signal, Hickling-normalized view")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
    ax1.set_xlim(*hickling_tau_xlim)

    ax2.plot(source_time_centered * 1e3, incident_signal, "k-", linewidth=1.5)
    ax2.set_xlabel("Time relative to pulse center [ms]")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Incident signal, centered physical view")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
    ax2.set_xlim(*hickling_ms_xlim)

    ax3.plot(source_tau_complete, incident_signal, "k-", linewidth=1.5)
    ax3.set_xlabel(r"$\tau = ct/a$")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Incident signal, full pulse duration")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
    ax3.set_xlim(0, T_pulse * c1 / a)

    ax4.plot(t * 1e3, incident_signal, "k-", linewidth=1.5)
    ax4.set_xlabel("Time [ms]")
    ax4.set_ylabel("Amplitude")
    ax4.set_title("Incident signal, full pulse duration")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
    ax4.set_xlim(0, T_pulse * 1e3)

    fig.suptitle(
        f"Incident Signal: {n_cycles}-cycle sinusoid, k0a={k0a:.1f}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()
else:
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
ax1.set_xlim(0, source_plot_ka_max)

ax2.plot(ka_positive, phase, "k-", linewidth=1.5)
ax2.set_xlabel("ka")
ax2.set_ylabel("Phase of g(ka) [rad]")
ax2.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax2.set_yticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, source_plot_ka_max)
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
    ka_eps=ka_eps,
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(ka_validation, np.abs(f_hickling), "k-", linewidth=1.5)
ax1.set_xlabel("ka")
ax1.set_ylabel("|f(ka)|")
ax1.set_title(f"Solid {material_sphere} Sphere Form Function - Hickling Far Field")
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

# Hickling Fig. 6 does not plot the wrapped phase directly. It uses the
# continuous phase parameter -arg(f_inf)/(ka), which is shown separately here
# so that both phase representations remain available for comparison.
phase_hickling_unwrapped = np.unwrap(np.angle(f_hickling))
phase_hickling_parameter = np.full_like(ka_validation, np.nan, dtype=float)
phase_valid = np.abs(ka_validation) > ka_eps
phase_hickling_parameter[phase_valid] = (
    -phase_hickling_unwrapped[phase_valid] / ka_validation[phase_valid]
)

plt.figure(figsize=(12, 5))
plt.plot(ka_validation, phase_hickling_parameter, "k-", linewidth=1.5)
plt.xlabel("ka")
plt.ylabel(r"$-\mathrm{arg}(f_\infty)/(ka)$")
plt.title(f"Solid {material_sphere} Sphere Phase Parameter - Hickling Fig. 6")
plt.grid(True, alpha=0.3)
plt.xlim(0, 30)
plt.ylim(0, 2)
plt.tight_layout()
plt.show()

print("\nValidation form function computed:")
print(f"  material: {material_sphere}")
print(f"  ka range: {ka_validation.min():.2f} / {ka_validation.max():.2f}")
print(
    f"  Compare this figure against Hickling (1962) solid {material_sphere} sphere results."
)


# ============================================================================
# STEP 5: Compute Solid-Sphere Response for Echo Synthesis
# ============================================================================

print(f"\nComputing {echo_field} response for echo synthesis...")
f_echo_positive, x_echo = compute_echo_response(
    freqs=freq_positive,
    c1=c1,
    a=a,
    cd2=cd2,
    cs2=cs2,
    rho1=rho1,
    rho2=rho2,
    r=r,
    n_terms=n_terms,
    field=echo_field,
    ka_eps=ka_eps,
)

integration_ka_bounds = source_case["integration_ka_bounds"]
if integration_ka_bounds is None:
    f_echo_for_synthesis = f_echo_positive
    response_title_suffix = "full ka range"
    response_plot_ka_max = 30
else:
    ka_min, ka_max = integration_ka_bounds
    integration_mask = (x_echo >= ka_min) & (x_echo <= ka_max)
    f_echo_for_synthesis = np.where(integration_mask, f_echo_positive, 0.0)
    response_title_suffix = f"ka {ka_min:g}-{ka_max:g}"
    response_plot_ka_max = ka_max + 5

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(x_echo, np.abs(f_echo_for_synthesis), "k-", linewidth=1.5)
ax1.set_xlabel("ka")
ax1.set_ylabel("|response|")
ax1.set_title(
    f"Solid {material_sphere} Sphere {echo_field} Response Used for Echo "
    f"({response_title_suffix})"
)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, response_plot_ka_max)

phase_echo = np.mod(np.angle(f_echo_for_synthesis), 2 * np.pi)
ax2.plot(x_echo, phase_echo, "k-", linewidth=1.5)
ax2.set_xlabel("ka")
ax2.set_ylabel("Phase [rad]")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, response_plot_ka_max)
ax2.set_ylim(0, 2 * np.pi)
plt.tight_layout()
plt.show()


# ============================================================================
# STEP 6: Compute Scattered Echo
# ============================================================================

material_id = "".join(ch for ch in material_sphere if ch.isalnum())
target = f"Sol{material_id}Sphere"
f_echo_full = get_form_function_echo(f_echo_for_synthesis)

# Keep the same convention as GetEchoFF in AcousticScattering_Menu_FF_spheres.py:
# non-rigid spherical targets multiply by the reversed rebuilt form-function array.
echo_spectrum = incident_fft * f_echo_full[::-1]
scattered_echo = np.fft.ifft(echo_spectrum, n_fft)
scattered_echo_real = np.real(scattered_echo)

t_echo = np.arange(n_fft) / sample_rate
if is_hickling_source_case:
    tau_echo = t_echo * c1 / a
    R = r / a
    echo_time_axis = tau_echo - 2 * R
    echo_time_label = r"$\tau - 2R$"
    echo_time_xlim = (-3, 5)
else:
    echo_time_axis = t_echo * 1e3
    echo_time_label = "Time [ms]"
    echo_time_xlim = (0, max(3.0, 1.5 * T_pulse * 1e3))

if is_hickling_source_case:
    normalization_mask = (echo_time_axis >= echo_time_xlim[0]) & (
        echo_time_axis <= echo_time_xlim[1]
    )
else:
    normalization_mask = np.ones_like(echo_time_axis, dtype=bool)

normalization_reference = np.max(np.abs(scattered_echo_real[normalization_mask]))
scattered_echo_normalized = scattered_echo_real / (normalization_reference + 1e-15)

print("\nScattered pulse computed:")
print(f"  target: {target}")
print(f"  echo field: {echo_field}")
print(f"  source case: {selected_source_case}")
print(f"  response range: {response_title_suffix}")
if is_hickling_source_case:
    print(f"  normalized range reference R = r/a: {R:.3f}")
    print("  echo time axis: tau - 2R")
print(f"  samples: {len(scattered_echo_normalized)}")
print(f"  finite ratio: {np.isfinite(scattered_echo_normalized).mean():.3f}")
print(f"  max abs before normalization: {np.max(np.abs(scattered_echo)):.3e}")


# ============================================================================
# STEP 7: Visualize Echo
# ============================================================================

plt.figure(figsize=(12, 6))
plt.plot(echo_time_axis, scattered_echo_normalized, "k-", linewidth=1.5)
plt.xlabel(echo_time_label)
plt.ylabel("Amplitude (norm.)")
plt.title(f"Solid {material_sphere} Sphere Echo - {echo_field}, {selected_source_case}")
plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
plt.xlim(*echo_time_xlim)
plt.ylim(-1.1, 1.1)

plt.tight_layout()
plt.show()


# ============================================================================
# STEP 8: Direct Hickling Eq. (14) Echo Synthesis
# ============================================================================

# This is an independent diagnostic for the Hickling far-field cases. It uses
# the analytical source spectrum from Eq. (16) and integrates Eq. (14) directly,
# without FFT/IFFT, bilateral-spectrum reconstruction, or array reversal.
if is_hickling_source_case and echo_field == "far-field":
    ka_min, ka_max = integration_ka_bounds
    direct_integration_mask = (x_echo >= ka_min) & (x_echo <= ka_max)
    x_direct = x_echo[direct_integration_mask]
    f_direct = f_echo_positive[direct_integration_mask]

    # Hickling's pulse occupies -Delta_t < t < Delta_t. For a total of
    # n_cycles, its dimensionless half-duration is Delta_tau = N*pi/x0.
    delta_tau = n_cycles * np.pi / k0a
    source_offset = x_direct - k0a
    g_direct = (
        (2.0 / np.pi)
        * delta_tau
        * np.sinc(source_offset * delta_tau / np.pi)
    )

    direct_echo_time_axis = np.linspace(*echo_time_xlim, 1601)
    direct_echo_complex = np.empty_like(direct_echo_time_axis, dtype=complex)
    spectral_product = g_direct * f_direct

    # Evaluate exp[-i*x*(tau - 2R)] in small blocks to avoid allocating one
    # large time-by-frequency matrix.
    integration_block_size = 256
    for block_start in range(0, direct_echo_time_axis.size, integration_block_size):
        block_stop = min(
            block_start + integration_block_size, direct_echo_time_axis.size
        )
        phase_kernel = np.exp(
            -1j
            * np.outer(
                direct_echo_time_axis[block_start:block_stop],
                x_direct,
            )
        )
        direct_echo_complex[block_start:block_stop] = np.trapezoid(
            phase_kernel * spectral_product,
            x=x_direct,
            axis=1,
        )

    direct_echo_real = np.real(direct_echo_complex)
    direct_normalization = np.max(np.abs(direct_echo_real))
    direct_echo_normalized = direct_echo_real / (direct_normalization + 1e-15)

    print("\nDirect Hickling Eq. (14) echo computed:")
    print("  source spectrum: Hickling Eq. (16), 2/pi prefactor")
    print(f"  Delta_tau: {delta_tau:.6f}")
    print(f"  integration range: ka {ka_min:g}-{ka_max:g}")
    print("  Fourier kernel: exp[-i*x*(tau - 2R)]")

    plt.figure(figsize=(12, 6))
    plt.plot(
        direct_echo_time_axis,
        direct_echo_normalized,
        "k-",
        linewidth=1.5,
    )
    plt.xlabel(r"$\tau - 2R$")
    plt.ylabel("Amplitude (norm.)")
    plt.title(
        "Solid "
        f"{material_sphere} Sphere Echo - Hickling Eq. (14), "
        f"{selected_source_case}"
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axhline(y=0.0, color="k", linestyle="-", linewidth=0.5)
    plt.xlim(*echo_time_xlim)
    plt.ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.show()


# ============================================================================
# STEP 9: OpenSTB-Style Complex FFT/IFFT Echo Synthesis
# ============================================================================

# This second diagnostic follows the OpenSTB signal-processing path while keeping
# the same Hickling case and ka integration bounds used by the direct synthesis.
# The form function is evaluated at the physical FFT frequencies and multiplied
# directly, without bilateral reconstruction or array reversal.
if is_hickling_source_case and echo_field == "far-field":
    openstb_baseband_frequency = 0.0
    openstb_initial_phase = 0.0
    openstb_source = np.where(
        t < T_pulse,
        np.exp(
            1j
            * (
                2
                * np.pi
                * (f0 - openstb_baseband_frequency)
                * t
                + openstb_initial_phase
                - np.pi / 2
            )
        ),
        0.0j,
    )

    # Figure 1: analytic complex source used by OpenSTB's SinusoidBurst.
    source_tau_centered = t * c1 / a - delta_tau
    plt.figure(figsize=(12, 5))
    plt.plot(
        source_tau_centered,
        np.real(openstb_source),
        "k-",
        linewidth=1.5,
        label="Real part",
    )
    plt.plot(
        source_tau_centered,
        np.imag(openstb_source),
        color="0.55",
        linestyle="--",
        linewidth=1.2,
        label="Imaginary part",
    )
    plt.xlabel(r"Source time, $\tau$ (centered)")
    plt.ylabel("Amplitude")
    plt.title(
        "OpenSTB-Style Analytic Source - "
        f"{selected_source_case}, {n_cycles:g} cycles"
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axhline(y=0.0, color="k", linewidth=0.5)
    plt.xlim(-delta_tau - 0.5, delta_tau + 0.5)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    openstb_frequency_offset = np.fft.fftshift(np.fft.fftfreq(n_fft, dt))
    openstb_physical_frequency = (
        openstb_frequency_offset + openstb_baseband_frequency
    )
    openstb_source_spectrum = np.fft.fftshift(
        np.fft.fft(openstb_source, n_fft)
    )
    openstb_ka = 2 * np.pi * openstb_physical_frequency * a / c1

    # Figure 2: the causal FFT source and Hickling's centered analytical source
    # have the same normalized magnitude; their phases include different time origins.
    positive_source_mask = (openstb_ka >= 0.0) & (
        openstb_ka <= source_plot_ka_max
    )
    openstb_source_magnitude = np.abs(
        openstb_source_spectrum[positive_source_mask]
    )
    openstb_source_magnitude /= np.max(openstb_source_magnitude) + 1e-15
    hickling_source_magnitude = np.abs(g_direct)
    hickling_source_magnitude /= np.max(hickling_source_magnitude) + 1e-15

    plt.figure(figsize=(12, 5))
    plt.plot(
        openstb_ka[positive_source_mask],
        openstb_source_magnitude,
        "k-",
        linewidth=1.5,
        label="OpenSTB complex source: FFT magnitude",
    )
    plt.plot(
        x_direct,
        hickling_source_magnitude,
        color="0.55",
        linestyle="--",
        linewidth=1.5,
        label="Hickling Eq. (16): analytical magnitude",
    )
    plt.xlabel("ka")
    plt.ylabel("Normalized magnitude")
    plt.title(f"Complex Source Spectrum Comparison - {selected_source_case}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(ka_min, ka_max)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    openstb_response = np.zeros(n_fft, dtype=complex)
    openstb_response_mask = (
        (openstb_physical_frequency > 0.0)
        & (openstb_ka >= ka_min)
        & (openstb_ka <= ka_max)
    )
    openstb_response_active, _ = compute_echo_response(
        freqs=openstb_physical_frequency[openstb_response_mask],
        c1=c1,
        a=a,
        cd2=cd2,
        cs2=cs2,
        rho1=rho1,
        rho2=rho2,
        r=r,
        n_terms=n_terms,
        field=echo_field,
        ka_eps=ka_eps,
    )
    openstb_response[openstb_response_mask] = openstb_response_active

    travel_time = 2 * r / c1
    travel_phase = np.exp(
        -2j * np.pi * openstb_physical_frequency * travel_time
    )
    openstb_echo_spectrum = (
        openstb_source_spectrum * openstb_response * travel_phase
    )
    openstb_echo = np.fft.ifft(np.fft.ifftshift(openstb_echo_spectrum))

    openstb_echo_time = np.arange(n_fft) / sample_rate
    openstb_echo_time_axis = (
        openstb_echo_time * c1 / a - 2 * R - delta_tau
    )
    openstb_echo_real = np.real(openstb_echo)
    openstb_normalization_mask = (
        (openstb_echo_time_axis >= echo_time_xlim[0])
        & (openstb_echo_time_axis <= echo_time_xlim[1])
    )
    openstb_normalization = np.max(
        np.abs(openstb_echo_real[openstb_normalization_mask])
    )
    openstb_echo_normalized = openstb_echo_real / (
        openstb_normalization + 1e-15
    )

    # Hickling and OpenSTB use opposite complex-exponential conventions. Test
    # that conversion by changing only the form-function phase convention.
    openstb_conjugated_echo_spectrum = (
        openstb_source_spectrum * np.conjugate(openstb_response) * travel_phase
    )
    openstb_conjugated_echo = np.fft.ifft(
        np.fft.ifftshift(openstb_conjugated_echo_spectrum)
    )
    openstb_conjugated_echo_real = np.real(openstb_conjugated_echo)
    openstb_conjugated_normalization = np.max(
        np.abs(openstb_conjugated_echo_real[openstb_normalization_mask])
    )
    openstb_conjugated_echo_normalized = openstb_conjugated_echo_real / (
        openstb_conjugated_normalization + 1e-15
    )

    print("\nOpenSTB-style complex FFT/IFFT echo computed:")
    print("  source: analytic SinusoidBurst convention")
    print(f"  baseband frequency: {openstb_baseband_frequency:.1f} Hz")
    print(f"  initial phase: {openstb_initial_phase:.3f} rad")
    print(f"  response range: ka {ka_min:g}-{ka_max:g}")
    print("  form-function multiplication: direct (no reversal)")
    print("  comparison variant: conjugated form function (no reversal)")
    print(f"  travel time: {travel_time:.6e} s")

    # Figure 3: compare the validated direct integral with the OpenSTB-style
    # direct and conjugated form-function variants on identical normalized axes.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    ax1.plot(
        direct_echo_time_axis,
        direct_echo_normalized,
        "k-",
        linewidth=1.5,
    )
    ax1.set_ylabel("Amplitude (norm.)")
    ax1.set_title("Reference: Direct Hickling Eq. (14)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.axhline(y=0.0, color="k", linewidth=0.5)
    ax1.set_ylim(-1.1, 1.1)

    ax2.plot(
        openstb_echo_time_axis,
        openstb_echo_normalized,
        "k-",
        linewidth=1.5,
    )
    ax2.set_ylabel("Amplitude (norm.)")
    ax2.set_title("OpenSTB-Style Complex FFT/IFFT: Direct Form Function")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.axhline(y=0.0, color="k", linewidth=0.5)
    ax2.set_ylim(-1.1, 1.1)

    ax3.plot(
        openstb_echo_time_axis,
        openstb_conjugated_echo_normalized,
        "k-",
        linewidth=1.5,
    )
    ax3.set_xlabel(r"$\tau - 2R$")
    ax3.set_ylabel("Amplitude (norm.)")
    ax3.set_title("OpenSTB-Style Complex FFT/IFFT: Conjugated Form Function")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.axhline(y=0.0, color="k", linewidth=0.5)
    ax3.set_xlim(*echo_time_xlim)
    ax3.set_ylim(-1.1, 1.1)

    fig.suptitle(
        f"Solid {material_sphere} Sphere Echo Comparison - "
        f"{selected_source_case}"
    )
    plt.tight_layout()
    plt.show()
