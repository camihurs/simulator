"""Solid elastic sphere in free field: analytic FFT/IFFT standalone.

Historical SolidSphereEcho.py is preserved. Modal equations retain Hickling's
convention; explicit conjugation adapts the response to NumPy/OpenSTB synthesis.
Only far-field monostatic backscattering is supported here.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn

# USER CONFIGURATION
SOURCE_CASE = "lfm_chirp"  # six Hickling presets
# (the first one hickling_fig16_max), tone_burst, lfm_chirp
SELECTED_MATERIAL = "Armco iron"
ECHO_FIELD = "far-field"  # near field is not validated or supported by this version
SHOW_VALIDATION_PLOTS = True
COMPARE_DIRECT_HICKLING = True
COMPARE_OPENSTB_DUMP = True  # Independent of source case / Hickling diagnostics
COMPARE_OPENSTB_ECHO = True
# This comparison requires the SAME selected source/material as the simulation.
OPENSTB_RESULTS = Path(__file__).resolve().parents[4] / "simple_points.npz"
OPENSTB_ECHO_PING_INDEX = 30
OPENSTB_ECHO_RECEIVER_INDEX = 0
OPENSTB_ECHO_PLOT_WINDOW_MS = None  # None: source duration plus target-response window
OPENSTB_DUMP = (
    Path(__file__).resolve().parents[3]
    / "simple_points_study"
    / "solid_sphere_ff_debug.npz"
)
SOUND_SPEED_M_S = 1410.0
WATER_DENSITY_KG_M3 = 1000.0
SPHERE_RADIUS_M = 0.25
N_TERMS = 80
KA_EPS = 1e-8
N_FFT = 16384
RECORD_DURATION_FACTOR = 3.0
FORM_FUNCTION_PLOT_KA_MAX = 30.0
TONE_FREQUENCY_HZ = 14_000.0
TONE_CYCLES = 4.0
CHIRP_START_HZ = 500.0
CHIRP_STOP_HZ = 10_000.0
CHIRP_DURATION_S = 0.020
CHIRP_SAMPLE_RATE_HZ = 30_000.0
CHIRP_USE_HANN_WINDOW = True

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


def form_function_hickling(ka):
    """Backscattering form function in the historical Hickling convention."""
    if ECHO_FIELD != "far-field":
        raise ValueError("Only validated far-field backscattering is supported")
    if SELECTED_MATERIAL not in HICKLING_MATERIALS:
        raise ValueError(f"Unknown material: {SELECTED_MATERIAL}")
    material = HICKLING_MATERIALS[SELECTED_MATERIAL]
    ka = np.asarray(ka, dtype=float)
    if np.any(ka < 0):
        raise ValueError("ka must be non-negative")
    frequency = ka * SOUND_SPEED_M_S / (2 * np.pi * SPHERE_RADIUS_M)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ff, _ = compute_hickling_form_function(
            frequency,
            SOUND_SPEED_M_S,
            SPHERE_RADIUS_M,
            material["cd2"],
            material["cs2"],
            WATER_DENSITY_KG_M3,
            material["rho2"],
            1.0,
            N_TERMS,
            KA_EPS,
        )
    return ff


def build_source() -> dict[str, object]:
    """Build one analytic source using the same sign convention as OpenSTB."""

    if SOURCE_CASE.startswith("hickling_"):
        case = SOURCE_CASES[SOURCE_CASE]
        k0a = case["k0a"]
        f0 = k0a * SOUND_SPEED_M_S / (2 * np.pi * SPHERE_RADIUS_M)
        n_cycles = float(case["n_cycles"])
        duration = n_cycles / f0
        sample_rate = 10.0 * f0
        baseband_frequency = 0.0
        label = f"{SOURCE_CASE}: {n_cycles:g} cycles, k0a={k0a:g}"

        def evaluate(t: np.ndarray) -> np.ndarray:
            valid = (t >= 0.0) & (t < duration)
            signal = np.zeros_like(t, dtype=np.complex128)
            # Analytic equivalent of sin(2*pi*f0*t), matching OpenSTB SinusoidBurst.
            phase = 2 * np.pi * f0 * (t[valid] - duration / 2)
            signal[valid] = np.exp(1j * phase)
            return signal

    elif SOURCE_CASE == "tone_burst":
        f0 = TONE_FREQUENCY_HZ
        n_cycles = TONE_CYCLES
        duration = n_cycles / f0
        sample_rate = 10.0 * f0
        baseband_frequency = 0.0
        k0a = 2 * np.pi * f0 * SPHERE_RADIUS_M / SOUND_SPEED_M_S
        label = f"Analytic {n_cycles:g}-cycle tone burst, k0a={k0a:.3f}"

        def evaluate(t: np.ndarray) -> np.ndarray:
            valid = (t >= 0.0) & (t < duration)
            signal = np.zeros_like(t, dtype=np.complex128)
            phase = 2 * np.pi * f0 * t[valid] - np.pi / 2
            signal[valid] = np.exp(1j * phase)
            return signal

    elif SOURCE_CASE == "lfm_chirp":
        f0 = None
        n_cycles = None
        k0a = None
        duration = CHIRP_DURATION_S
        sample_rate = CHIRP_SAMPLE_RATE_HZ
        baseband_frequency = 0.5 * (CHIRP_START_HZ + CHIRP_STOP_HZ)
        label = (
            f"Analytic LFM chirp, {CHIRP_START_HZ / 1e3:g}-{CHIRP_STOP_HZ / 1e3:g} kHz"
        )

        def evaluate(t: np.ndarray) -> np.ndarray:
            valid = (t >= 0.0) & (t <= duration)
            tv = t[valid]
            signal = np.zeros_like(t, dtype=np.complex128)
            fd = CHIRP_START_HZ - baseband_frequency
            chirp_rate = (CHIRP_STOP_HZ - CHIRP_START_HZ) / duration
            signal[valid] = np.exp(1j * np.pi * tv * (2 * fd + chirp_rate * tv))
            if CHIRP_USE_HANN_WINDOW:
                # Continuous Hann window, zero at both pulse endpoints.
                signal[valid] *= 0.5 * (1.0 - np.cos(2 * np.pi * tv / duration))
            return signal

    else:
        raise ValueError(f"Unknown SOURCE_CASE: {SOURCE_CASE!r}")

    record_duration = RECORD_DURATION_FACTOR * duration
    n_samples = max(2, int(np.ceil(record_duration * sample_rate)))
    t = np.arange(n_samples) / sample_rate
    signal = evaluate(t)

    return {
        "label": label,
        "time_s": t,
        "signal": signal,
        "sample_rate_hz": sample_rate,
        "baseband_frequency_hz": baseband_frequency,
        "duration_s": duration,
        "f0_hz": f0,
        "n_cycles": n_cycles,
        "k0a": k0a,
    }


def synthesize_echo(source: dict[str, object]) -> dict[str, np.ndarray]:
    """Apply the rigid-sphere response with the OpenSTB FFT/IFFT convention."""

    signal = np.asarray(source["signal"], dtype=np.complex128)
    sample_rate = float(source["sample_rate_hz"])
    baseband_frequency = float(source["baseband_frequency_hz"])

    if signal.size > N_FFT:
        raise ValueError("N_FFT must cover the complete source record")

    frequency_offset = np.fft.fftshift(np.fft.fftfreq(N_FFT, d=1.0 / sample_rate))
    physical_frequency = frequency_offset + baseband_frequency
    source_spectrum = np.fft.fftshift(np.fft.fft(signal, n=N_FFT))

    ka_signed = 2 * np.pi * physical_frequency * SPHERE_RADIUS_M / SOUND_SPEED_M_S
    ka = np.abs(ka_signed)
    # The modal function remains in Hickling's exp(-i*omega*t) convention.
    ff_hickling = form_function_hickling(ka)
    # NumPy IFFT uses exp(+i*omega*t). Conjugation converts the complex response
    # to that convention; it reverses phase, preserves magnitude, and avoids the
    # reversed packet ordering obtained by multiplying by ff_hickling directly.
    ff_openstb = np.conjugate(ff_hickling)
    active = physical_frequency > 0
    if SOURCE_CASE.startswith("hickling_"):
        lower, upper = SOURCE_CASES[SOURCE_CASE]["integration_ka_bounds"]
        active &= (ka_signed >= lower) & (ka_signed <= upper)
    # The positive-physical-frequency analytic-source domain is explicit.
    # No Hermitian spectrum reconstruction or frequency-array reversal is used.
    response = np.where(active, ff_openstb, 0.0)
    echo_spectrum = source_spectrum * response
    echo_baseband = np.fft.ifft(np.fft.ifftshift(echo_spectrum))

    time_s = np.arange(N_FFT) / sample_rate
    echo_passband = np.real(
        echo_baseband * np.exp(1j * 2 * np.pi * baseband_frequency * time_s)
    )

    # Centered coordinates are useful because this standalone applies no propagation
    # delay; the target impulse response wraps around the FFT record origin.
    centered_time_s = (np.arange(N_FFT) - N_FFT // 2) / sample_rate

    return {
        "frequency_hz": physical_frequency,
        "frequency_offset_hz": frequency_offset,
        "source_spectrum": source_spectrum,
        "ka": ka,
        "form_function": ff_hickling,
        "ff_openstb": ff_openstb,
        "response_mask": active,
        "echo_spectrum": echo_spectrum,
        "echo_baseband": echo_baseband,
        "echo_passband": echo_passband,
        "centered_time_s": centered_time_s,
        "echo_baseband_centered": np.fft.fftshift(echo_baseband),
        "echo_passband_centered": np.fft.fftshift(echo_passband),
    }


def normalized(values: np.ndarray) -> np.ndarray:
    """Normalize an array by its maximum absolute value without dividing by zero."""

    values = np.asarray(values)
    return values / (np.max(np.abs(values)) + np.finfo(float).eps)


def plot_standard_results(
    source: dict[str, object], result: dict[str, np.ndarray]
) -> None:
    """Plot the three concise figures produced in normal standalone use."""

    source_time = np.asarray(source["time_s"])
    source_signal = np.asarray(source["signal"])
    frequency = result["frequency_hz"]
    source_spectrum = result["source_spectrum"]
    ka = result["ka"]
    ff = result["form_function"]

    significant = np.abs(source_spectrum) > np.max(np.abs(source_spectrum)) * 1e-4
    positive_physical = frequency >= 0.0

    is_lfm = SOURCE_CASE == "lfm_chirp"
    row_count = 3 if is_lfm else 2
    fig, axes = plt.subplots(row_count, 1, figsize=(11, 9 if is_lfm else 7))
    ax_baseband = axes[0]
    ax_spectrum = axes[-1]

    baseband_label = "complex baseband real part" if is_lfm else "physical real part"
    ax_baseband.plot(source_time * 1e3, np.real(source_signal), label=baseband_label)
    ax_baseband.plot(
        source_time * 1e3,
        np.abs(source_signal),
        "k--",
        alpha=0.65,
        label="analytic envelope",
    )
    ax_baseband.set(
        xlabel="Time [ms]",
        ylabel="Amplitude",
        title="Standalone incident complex signal used for echo synthesis\n"
        + str(source["label"]),
    )
    ax_baseband.grid(True, alpha=0.3)
    ax_baseband.legend()

    if is_lfm:
        ax_passband = axes[1]
        baseband_frequency = float(source["baseband_frequency_hz"])
        physical_signal = np.real(
            source_signal * np.exp(1j * 2 * np.pi * baseband_frequency * source_time)
        )
        ax_passband.plot(
            source_time * 1e3, physical_signal, label="reconstructed physical chirp"
        )
        ax_passband.plot(
            source_time * 1e3,
            np.abs(source_signal),
            "k--",
            alpha=0.65,
            label="analytic envelope",
        )
        ax_passband.set(
            xlabel="Time [ms]",
            ylabel="Amplitude",
            title=(
                "Physical passband view: monotonically increasing frequency "
                f"({CHIRP_START_HZ / 1e3:g}-{CHIRP_STOP_HZ / 1e3:g} kHz)"
            ),
        )
        ax_passband.grid(True, alpha=0.3)
        ax_passband.legend()

        active_xlim_ms = (
            -0.02 * float(source["duration_s"]) * 1e3,
            1.02 * float(source["duration_s"]) * 1e3,
        )
        ax_baseband.set_xlim(*active_xlim_ms)
        ax_passband.set_xlim(*active_xlim_ms)

    plot_mask = significant & positive_physical
    ax_spectrum.plot(
        frequency[plot_mask] / 1e3, normalized(np.abs(source_spectrum[plot_mask]))
    )
    ax_spectrum.set(
        xlabel="Physical frequency [kHz]",
        ylabel="Normalized magnitude",
        title="Standalone incident complex-signal spectrum (FFT)",
    )
    ax_spectrum.grid(True, alpha=0.3)
    fig.tight_layout()

    order = np.argsort(ka)
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(11, 7))
    ax_mag.plot(ka[order], np.abs(ff[order]))
    ax_mag.set(
        xlabel="ka",
        ylabel="|f(ka)|",
        title=(
            f"{SELECTED_MATERIAL}: form function calculated by the standalone\n"
            "Hickling convention: modal response for exp(-iωt), before conjugation"
        ),
        xlim=(0, FORM_FUNCTION_PLOT_KA_MAX),
    )
    ax_mag.grid(True, alpha=0.3)
    ax_phase.plot(ka[order], np.mod(np.angle(ff[order]), 2 * np.pi))
    ax_phase.set(
        xlabel="ka",
        ylabel="arg[f(ka)] [rad]",
        title="Standalone modal phase, wrapped to [0, 2π)",
        xlim=(0, FORM_FUNCTION_PLOT_KA_MAX),
        ylim=(0, 2 * np.pi),
    )
    ax_phase.grid(True, alpha=0.3)
    fig.tight_layout()

    echo = normalized(result["echo_passband_centered"])
    echo_spectrum = result["echo_spectrum"]
    fig, (ax_echo, ax_echo_spectrum) = plt.subplots(2, 1, figsize=(11, 7))
    ax_echo.plot(result["centered_time_s"] * 1e3, echo)
    ax_echo.set(
        xlabel="Time relative to FFT origin [ms]",
        ylabel="Normalized pressure",
        title=(
            "Solid-sphere echo calculated by the analytic standalone (target only)\n"
            "Analytic: complex-source FFT/IFFT synthesis, not a closed-form time solution"
        ),
    )
    ax_echo.grid(True, alpha=0.3)
    visible_echo = np.abs(echo) > 1e-3
    if np.any(visible_echo):
        visible_time = result["centered_time_s"][visible_echo] * 1e3
        padding = max(0.05, 0.1 * (visible_time.max() - visible_time.min()))
        ax_echo.set_xlim(visible_time.min() - padding, visible_time.max() + padding)
    ax_echo_spectrum.plot(
        frequency[plot_mask] / 1e3,
        normalized(np.abs(echo_spectrum[plot_mask])),
    )
    ax_echo_spectrum.set(
        xlabel="Physical frequency [kHz]",
        ylabel="Normalized magnitude",
        title="Scattered spectrum calculated by the standalone",
    )
    ax_echo_spectrum.grid(True, alpha=0.3)
    fig.tight_layout()


def hickling_spectrum(ka, source):
    """Eq. (16), with the correct 2/pi prefactor and centered time origin."""
    half_duration_tau = float(source["n_cycles"]) * np.pi / float(source["k0a"])
    return (
        (2 / np.pi)
        * half_duration_tau
        * np.sinc((ka - float(source["k0a"])) * half_duration_tau / np.pi)
    )


def plot_hickling_validation(source, result):
    """Independent centered Eq. (14) synthesis, not the normal FFT pipeline."""
    if not SOURCE_CASE.startswith("hickling_"):
        print("Hickling-specific diagnostics skipped for the general source")
        return
    if SELECTED_MATERIAL != "Armco iron":
        warnings.warn("Hickling Figs. 16-18 use Armco iron; selected material differs.")
    lower, upper = SOURCE_CASES[SOURCE_CASE]["integration_ka_bounds"]
    ka = np.linspace(lower, upper, 4096)
    ff = form_function_hickling(ka)
    g = hickling_spectrum(ka, source)
    positive = (result["frequency_hz"] > 0) & (result["ka"] <= upper + 5)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        result["ka"][positive],
        normalized(abs(result["source_spectrum"][positive])),
        label="Standalone complex-source FFT",
    )
    ax.plot(ka, normalized(abs(g)), "--", label="Hickling Eq. (16)")
    ax.set(
        xlabel="ka",
        ylabel="Normalized source magnitude",
        title=(
            "Incident-signal spectrum: standalone FFT vs Hickling Eq. (16)\n"
            "Magnitudes independently normalized to their maxima"
        ),
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    phase_ka = np.linspace(0.001, FORM_FUNCTION_PLOT_KA_MAX, 4096)
    phase_ff = form_function_hickling(phase_ka)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(phase_ka, -np.unwrap(np.angle(phase_ff)) / phase_ka)
    ax.set(
        xlabel="ka",
        ylabel="-unwrapped arg(f)/ka",
        title=(
            "Standalone form function — Hickling phase parameter\n"
            "−unwrap(arg f) / ka: continuous phase divided by ka, not the wrapped phase above"
        ),
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if COMPARE_DIRECT_HICKLING:
        half = float(source["duration_s"]) * SOUND_SPEED_M_S / (2 * SPHERE_RADIUS_M)
        u = np.linspace(-half - 2, half + 12, 1601)
        direct = np.array(
            [np.trapezoid(g * ff * np.exp(-1j * ka * value), ka) for value in u]
        )
        fft_u = (
            (result["centered_time_s"] - float(source["duration_s"]) / 2)
            * SOUND_SPEED_M_S
            / SPHERE_RADIUS_M
        )
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(
            fft_u,
            normalized(result["echo_passband_centered"]),
            label="Standalone FFT/IFFT (conjugated modal response)",
        )
        ax.plot(
            u,
            normalized(direct.real),
            "--",
            label="Standalone numerical integration of Hickling Eq. (14)",
        )
        ax.set(
            xlabel="u = tau - 2R (pulse-centered, no propagation delay)",
            ylabel="Normalized pressure",
            title=(
                "Both echoes calculated by the standalone: FFT/IFFT vs Hickling Eq. (14) integration\n"
                "Pulse-centered dimensionless time; no propagation delay; each curve normalized"
            ),
            xlim=(u[0], u[-1]),
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()


def compare_openstb_dump(path=None):
    """Recompute raw modal responses with dump parameters, not source settings.

    This is a form-function regression comparison, not a final-echo comparison.
    The raw modal response is unscaled and unmasked in both conventions.
    """
    path = Path(OPENSTB_DUMP if path is None else path)
    if not path.exists():
        print(f"OpenSTB form-function comparison skipped; dump not found: {path}")
        return None
    scalar_names = (
        "sound_speed_m_s",
        "radius_m",
        "solid_density_kg_m3",
        "compressional_speed_m_s",
        "shear_speed_m_s",
        "fluid_density_kg_m3",
        "n_terms",
        "ka_eps",
    )
    array_names = ("ka", "f_hz", "ff_hickling_complex", "ff_openstb_complex")
    try:
        with np.load(path, allow_pickle=False) as stored:
            required = set(scalar_names + array_names) | {
                "target_model",
                "response_geometry",
            }
            missing = required - set(stored.files)
            if missing:
                raise ValueError(f"missing keys: {sorted(missing)}")
            if stored["target_model"].item() != "solid_sphere":
                raise ValueError("dump is not for a solid sphere")
            if (
                stored["response_geometry"].item()
                != "far_field_backscatter_approximation"
            ):
                raise ValueError("unsupported response geometry")
            parameters = {name: float(stored[name].item()) for name in scalar_names}
            arrays = {name: np.asarray(stored[name]).copy() for name in array_names}
        if any(not np.isfinite(value) or value <= 0 for value in parameters.values()):
            raise ValueError("invalid nonpositive/nonfinite modal parameters")
        if not parameters["n_terms"].is_integer():
            raise ValueError("n_terms must be an integer")
        ka = arrays["ka"].astype(float)
        if ka.ndim != 1 or not ka.size or np.any(ka < 0):
            raise ValueError("ka must be a nonempty nonnegative 1D grid")
        if any(
            value.shape != ka.shape or not np.isfinite(value).all()
            for value in arrays.values()
        ):
            raise ValueError("incompatible or nonfinite dump arrays")
        c = parameters["sound_speed_m_s"]
        a = parameters["radius_m"]
        if not np.allclose(
            ka, 2 * np.pi * abs(arrays["f_hz"]) * a / c, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                "ka disagrees with physical frequency, radius or sound speed"
            )
    except (OSError, ValueError, KeyError) as error:
        warnings.warn(f"OpenSTB comparison skipped: {error}", stacklevel=2)
        return None

    reference_hickling, _ = compute_hickling_form_function(
        arrays["f_hz"],
        c,
        a,
        parameters["compressional_speed_m_s"],
        parameters["shear_speed_m_s"],
        parameters["fluid_density_kg_m3"],
        parameters["solid_density_kg_m3"],
        1.0,
        int(parameters["n_terms"]),
        parameters["ka_eps"],
    )
    print(f"\nOpenSTB solid-sphere form-function comparison: {path}")
    print(
        "  Reference uses dump parameters, independently of the selected source/material:"
    )
    print(f"  {parameters}")
    print("  Raw form functions before scale/frequency mask; no final echo compared.")
    metrics = {}
    positive = arrays["f_hz"] > 0
    order = np.argsort(ka[positive])
    for convention, reference, key in (
        ("Hickling", reference_hickling, "ff_hickling_complex"),
        (
            "OpenSTB (conjugated)",
            np.conjugate(reference_hickling),
            "ff_openstb_complex",
        ),
    ):
        dumped = arrays[key]
        relative = np.linalg.norm(reference - dumped) / max(
            np.linalg.norm(dumped), np.finfo(float).eps
        )
        magnitude = np.linalg.norm(abs(reference) - abs(dumped)) / max(
            np.linalg.norm(dumped), np.finfo(float).eps
        )
        significant = (ka > parameters["ka_eps"]) & (
            abs(dumped) > abs(dumped).max() * 1e-10
        )
        if significant.any():
            phase = np.angle(reference[significant] * np.conjugate(dumped[significant]))
            phase_rms = float(np.sqrt(np.mean(phase**2)))
        else:
            phase_rms = float("nan")
        metrics[key] = dict(
            complex_relative_l2=float(relative),
            magnitude_relative_l2=float(magnitude),
            phase_rms_rad=phase_rms,
        )
        print(
            f"  {convention}: complex L2={relative:.3e}, magnitude L2={magnitude:.3e}, "
            f"wrapped phase RMS={phase_rms:.3e} rad"
        )
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(11, 7))
        x = ka[positive][order]
        for values, label, style in (
            (dumped, "OpenSTB plugin dump", "-"),
            (reference, "Standalone recomputation", "--"),
        ):
            y = values[positive][order]
            ax_mag.plot(x, abs(y), style, label=label)
            ax_phase.plot(x, np.mod(np.angle(y), 2 * np.pi), style, label=label)
        ax_mag.set(
            xlabel="ka",
            ylabel="|f(ka)|",
            title=(
                f"Solid-sphere form function — {convention} convention\n"
                "OpenSTB plugin dump vs standalone recomputation (before scale/frequency mask)"
            ),
        )
        ax_phase.set(xlabel="ka", ylabel="Wrapped phase [rad]", ylim=(0, 2 * np.pi))
        for ax in (ax_mag, ax_phase):
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
    return metrics


def compare_openstb_echo(source, dump_path=None, results_path=None):
    """Compare independent synthesis with one controlled OpenSTB trace.

    No peak alignment, fitted gain, or reuse of the plugin's output spectrum.
    Propagation is included analytically, then the plotting time axis subtracts
    that known delay. Metrics use independently peak-normalized complex samples; no phase fitting.
    """
    dump_path = Path(OPENSTB_DUMP if dump_path is None else dump_path)
    results_path = Path(OPENSTB_RESULTS if results_path is None else results_path)
    try:
        with np.load(dump_path, allow_pickle=False) as stored:
            dump = {name: stored[name].copy() for name in stored.files}
        with np.load(results_path, allow_pickle=False) as stored:
            t = stored["sample_time"].copy()
            pressure = stored["pressure"].copy()
            pings = stored["ping_start_time"].copy()
            carrier = float(stored["baseband_frequency"])
            dimensions = list(stored["pressure_dimensions"])
            if "echo_travel_time_s" not in stored:
                raise ValueError(
                    "result lacks travel times; rerun simulate.py with save_echo_metadata=True"
                )
            travel_times = stored["echo_travel_time_s"].copy()
            reflectivity = float(stored["echo_target_reflectivity"])
            if int(stored["echo_metadata_schema"]) != 1:
                raise ValueError("unsupported echo metadata schema")
            if list(stored["echo_travel_time_dimensions"]) != ["ping", "channel"]:
                raise ValueError("unsupported travel-time dimensions")
            if list(stored["echo_distortion_classes"]) != ["SolidSphereFormFunction"]:
                raise ValueError("comparison requires only the solid-sphere distortion")
        if dump["target_model"].item() != "solid_sphere":
            raise ValueError("not a solid-sphere dump")
        if dump["response_geometry"].item() != "far_field_backscatter_approximation":
            raise ValueError("unsupported response geometry")
        if ECHO_FIELD != "far-field":
            raise ValueError("only far-field comparison is supported")
        if dimensions != ["ping", "channel", "sample_time"]:
            raise ValueError("unsupported result dimensions")
        if (
            pressure.ndim != 3
            or pressure.shape[0] != len(pings)
            or pressure.shape[2] != len(t)
            or t.size < 2
        ):
            raise ValueError("incompatible pressure dimensions/time grid")
        if (
            travel_times.shape != pressure.shape[:2]
            or not np.isfinite(travel_times).all()
        ):
            raise ValueError("travel-time table does not match pressure")
        if not np.isfinite(reflectivity):
            raise ValueError("nonfinite target reflectivity")
        if not np.array_equal(dump["spectrum_shape"], [1, len(dump["f_hz"]), 1]):
            raise ValueError("comparison requires a single-target representative dump")
        material = HICKLING_MATERIALS[SELECTED_MATERIAL]
        expected_parameters = {
            "sound_speed_m_s": SOUND_SPEED_M_S,
            "radius_m": SPHERE_RADIUS_M,
            "fluid_density_kg_m3": WATER_DENSITY_KG_M3,
            "solid_density_kg_m3": material["rho2"],
            "compressional_speed_m_s": material["cd2"],
            "shear_speed_m_s": material["cs2"],
            "n_terms": N_TERMS,
            "ka_eps": KA_EPS,
        }
        for key, value in expected_parameters.items():
            if not np.isclose(float(dump[key]), value, rtol=1e-12, atol=0):
                raise ValueError(f"standalone/dump mismatch: {key}")
        fs = float(source["sample_rate_hz"])
        if not np.allclose(t, np.arange(t.size) / fs, rtol=1e-12, atol=1e-14):
            raise ValueError("standalone source and result sampling disagree")
        if not np.isclose(carrier, float(source["baseband_frequency_hz"]), atol=1e-12):
            raise ValueError("source and result carrier disagree")
        if not np.isclose(carrier, float(dump["baseband_frequency_hz"]), atol=1e-12):
            raise ValueError("dump and result carrier disagree")
        nfft = t.size + int(np.ceil(float(source["duration_s"]) * 1.1 * fs))
        frequency = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs)) + carrier
        if frequency.shape != dump["f_hz"].shape or not np.allclose(
            frequency, dump["f_hz"], rtol=1e-12, atol=1e-9
        ):
            raise ValueError("dump/result FFT grids disagree")
        if len(source["signal"]) > nfft:
            raise ValueError("result record is shorter than the standalone source")
        incident = np.fft.fftshift(np.fft.fft(source["signal"], nfft))
        # Compare unit-L2 COMPLEX spectra: only a real positive amplitude ratio
        # is ignored. No phase rotation, delay, or complex-gain fit is allowed.
        dumped_incident = dump["S_in_complex"]
        if dumped_incident.shape != incident.shape or not np.isfinite(dumped_incident).all():
            raise ValueError("invalid incident spectrum in dump")
        incident_norm = np.linalg.norm(incident)
        dumped_norm = np.linalg.norm(dumped_incident)
        if not np.isfinite(incident_norm) or not np.isfinite(dumped_norm) or min(incident_norm, dumped_norm) <= 0:
            raise ValueError("incident spectra must be finite and nonzero")
        source_scale = float(dumped_norm / incident_norm)
        source_error = np.linalg.norm(incident/incident_norm - dumped_incident/dumped_norm)
        if source_error > 1e-10:
            raise ValueError(
                f"incident spectra differ in shape or phase (normalized L2={source_error:.3e}); "
                "only a positive amplitude scale is allowed"
            )
        bounds = dump["validation_ka_bounds"]
        expected_bounds = (
            SOURCE_CASES[SOURCE_CASE]["integration_ka_bounds"]
            if SOURCE_CASE.startswith("hickling_")
            else ()
        )
        if not np.array_equal(bounds, expected_bounds):
            raise ValueError("standalone and dump validation bands disagree")
        ping_index = int(OPENSTB_ECHO_PING_INDEX)
        receiver_index = int(OPENSTB_ECHO_RECEIVER_INDEX)
        if not (
            0 <= ping_index < pressure.shape[0]
            and 0 <= receiver_index < pressure.shape[1]
        ):
            raise ValueError("selected ping/receiver index is out of range")
        delay = float(travel_times[ping_index, receiver_index])
        delay_origin = "per-ping/receiver travel-time table in results"
        if not np.isfinite(delay) or not 0 <= delay < t[-1]:
            raise ValueError("travel time is outside the recording window")
        actual = pressure[ping_index, receiver_index]
        if not np.isfinite(actual).all() or not np.linalg.norm(actual):
            raise ValueError("selected echo is nonfinite or zero")
    except (OSError, ValueError, KeyError) as error:
        warnings.warn(f"OpenSTB echo comparison skipped: {error}", stacklevel=2)
        return None

    ka = 2 * np.pi * abs(frequency) * SPHERE_RADIUS_M / SOUND_SPEED_M_S
    ff = np.conjugate(form_function_hickling(ka))
    mask = frequency > 0
    if len(expected_bounds):
        mask &= (ka >= expected_bounds[0]) & (ka <= expected_bounds[1])
    # Compute the reference independently of stored FF and S_out_complex.
    spectrum = incident * np.where(mask, float(dump["scale"]) * ff, 0)
    spectrum *= np.exp(-2j * np.pi * frequency * delay)
    spectrum *= reflectivity
    predicted = np.fft.ifft(np.fft.ifftshift(spectrum))[: t.size]
    predicted_peak = np.max(abs(predicted))
    actual_peak = np.max(abs(actual))
    if not np.isfinite(predicted).all() or predicted_peak <= 0:
        warnings.warn("OpenSTB echo comparison skipped: reference echo is nonfinite or zero",
                      stacklevel=2)
        return None
    # Positive real peak normalization preserves the phase of every complex sample.
    # Do not scale the incident signal or alter the independently synthesized echo.
    predicted_shape = predicted / predicted_peak
    actual_shape = actual / actual_peak
    relative = np.linalg.norm(predicted_shape - actual_shape) / np.linalg.norm(actual_shape)
    max_error = np.max(abs(predicted_shape - actual_shape))
    print(f"\nOpenSTB echo comparison: {results_path}")
    print(f"  Ping {ping_index} (t={pings[ping_index]:g} s), receiver {receiver_index}")
    print(f"  Delay={delay:.12g} s from {delay_origin}; no peak/gain fitting")
    print("  Uses stored single-target reflectivity; assumes no extra distortions.")
    print(
        f"  Source normalized L2={source_error:.3e}; normalized echo complex relative L2={relative:.3e}"
    )
    print(f"  Source positive amplitude ratio (OpenSTB/standalone)={source_scale:.9g}")
    print(f"  Echo maxima before normalization: OpenSTB={actual_peak:.9g}, standalone={predicted_peak:.9g}")
    print(f"  Normalized echo maximum complex-sample error={max_error:.3e}")
    print("  Shape/phase comparison only: absolute pressure agreement is NOT tested.")
    time_relative = (t - delay) * 1e3
    modulation = np.exp(2j * np.pi * carrier * t)
    fig, (ax_time, ax_error) = plt.subplots(2, 1, figsize=(11, 7))
    ax_time.plot(time_relative, (actual_shape * modulation).real, label="OpenSTB echo / its complex peak magnitude")
    ax_time.plot(
        time_relative,
        (predicted_shape * modulation).real,
        "--",
        label="Standalone echo / its complex peak magnitude",
    )
    ax_time.set(
        title=(
            f"Solid sphere — OpenSTB vs standalone echo, ping {ping_index}, receiver {receiver_index}\n"
            "Each echo normalized by its complex peak magnitude; time minus known travel delay"
        ),
        xlabel="Time minus calculated travel time [ms]",
        ylabel="Normalized pressure",
    )
    ax_error.plot(time_relative, abs(predicted_shape - actual_shape))
    ax_error.set(
        title="Difference between normalized complex echoes (phase preserved)",
        xlabel="Time minus calculated travel time [ms]",
        ylabel="Normalized absolute error",
    )
    # A band-limited response has long tails: do not let tiny ringing force a
    # full-record view that hides the principal packets. Metrics still use ALL samples.
    window = OPENSTB_ECHO_PLOT_WINDOW_MS
    if window is None:
        target_time = SPHERE_RADIUS_M / SOUND_SPEED_M_S
        window = (
            -4 * target_time * 1e3,
            (float(source["duration_s"]) + 20 * target_time) * 1e3,
        )
    for ax in (ax_time, ax_error):
        ax.set_xlim(*window)
    ax_time.legend()
    for ax in (ax_time, ax_error):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return dict(
        complex_relative_l2=float(relative),
        max_absolute_error=float(max_error),
        source_relative_l2=float(source_error),
        source_positive_scale=source_scale,
        normalization="complex_peak_magnitude",
        delay_s=delay,
        ping_index=ping_index,
        receiver_index=receiver_index,
    )


def main():
    source = build_source()
    result = synthesize_echo(source)
    print(f"Solid sphere: {SELECTED_MATERIAL}, {SOURCE_CASE}, {ECHO_FIELD}")
    print(
        f"FFT points: {N_FFT}; reference frequency: {source['baseband_frequency_hz']} Hz"
    )
    print("Response convention: ff_openstb = conjugate(ff_hickling)")
    print(f"Finite echo: {np.isfinite(result['echo_baseband']).all()}")
    plot_standard_results(source, result)
    if SHOW_VALIDATION_PLOTS:
        plot_hickling_validation(source, result)
    if COMPARE_OPENSTB_DUMP:
        compare_openstb_dump()
    if COMPARE_OPENSTB_ECHO:
        compare_openstb_echo(source)
    plt.show()


if __name__ == "__main__":
    main()
