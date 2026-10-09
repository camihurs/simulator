"""Solid elastic sphere in free field: analytic FFT/IFFT standalone.

Historical SolidSphereEcho.py is preserved. Modal equations retain Hickling's
convention; explicit conjugation adapts the response to NumPy/OpenSTB synthesis.
Only far-field monostatic backscattering is supported here.
"""

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
    ax_baseband.set(xlabel="Time [ms]", ylabel="Amplitude", title=str(source["label"]))
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
        title="Incident analytic-signal spectrum",
    )
    ax_spectrum.grid(True, alpha=0.3)
    fig.tight_layout()

    order = np.argsort(ka)
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(11, 7))
    ax_mag.plot(ka[order], np.abs(ff[order]))
    ax_mag.set(
        xlabel="ka",
        ylabel="|f(ka)|",
        title=f"{SELECTED_MATERIAL}: form function (Hickling convention)",
        xlim=(0, FORM_FUNCTION_PLOT_KA_MAX),
    )
    ax_mag.grid(True, alpha=0.3)
    ax_phase.plot(ka[order], np.mod(np.angle(ff[order]), 2 * np.pi))
    ax_phase.set(
        xlabel="ka",
        ylabel="arg[f(ka)] [rad]",
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
        title="Solid-sphere echo (target response only)",
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
        title="Scattered spectrum",
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
        label="analytic FFT",
    )
    ax.plot(ka, normalized(abs(g)), "--", label="Hickling Eq. (16)")
    ax.set(xlabel="ka", ylabel="Normalized source magnitude")
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
        title="Hickling phase parameter (modal convention)",
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
            label="FFT/IFFT with conjugated response",
        )
        ax.plot(u, normalized(direct.real), "--", label="Direct Hickling Eq. (14)")
        ax.set(
            xlabel="u = tau - 2R (pulse-centered, no propagation delay)",
            ylabel="Normalized pressure",
            xlim=(u[0], u[-1]),
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()


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
    plt.show()


if __name__ == "__main__":
    main()
