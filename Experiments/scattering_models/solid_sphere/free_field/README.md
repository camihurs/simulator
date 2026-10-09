# Solid Elastic Sphere Scattering in Free Field

This directory contains scientific standalones for the complex form function and
target-only echo of a homogeneous solid elastic sphere immersed in water. The
reference is Hickling's *Analysis of echoes from a solid elastic sphere in water*.
The scripts exclude the propagation and platform effects of the complete OpenSTB
simulation.

## 1. Two standalones, two purposes

| File | Purpose |
| --- | --- |
| `SolidSphereEcho.py` | Preserved historical exploratory and validation script, including the original synthesis and convention diagnostics. |
| `SolidSphereEchoAnalytic.py` | Recommended standalone for new experiments: one complex-source FFT/IFFT pipeline, with optional Hickling diagnostics. |

The second script was created to simplify the source-to-echo workflow and align its
Fourier organization with OpenSTB, not to replace or modify the validated modal
physics. It retains the historical modal equations and material presets without
importing either the historical script or OpenSTB at runtime.

The historical implementation was adapted from
`../../AcousticScattering_Menu_FF_spheres.py`. It remains available as a record of
the earlier scientific work; it is not the recommended template for new synthesis
paths, particularly its inherited frequency-array reversal.

## 2. Physical model and configuration

The new standalone supports free-field, far-field monostatic backscattering only.
The sphere is homogeneous and isotropic, with density, compressional-wave speed,
and shear-wave speed. The surrounding fluid has configurable density and sound
speed. Elastic modes produce resonances and multiple scattered wave packets.
No explicit material damping is included.

The dimensionless frequency is

$$ka = \frac{2\pi f a}{c},$$

where $f$ is physical frequency, $a$ is sphere radius, and $c$ is fluid sound speed.
The modal solution uses spherical Bessel/Hankel functions and the historical
boundary-condition determinant ratio. The far-field scaling is

$$f_\infty(ka)=\frac{2}{i\,ka}\sum_n f_n(ka).$$

Settings near the top of `SolidSphereEchoAnalytic.py` include:

```python
SELECTED_MATERIAL = "Armco iron"
ECHO_FIELD = "far-field"
SOUND_SPEED_M_S = 1410.0
WATER_DENSITY_KG_M3 = 1000.0
SPHERE_RADIUS_M = 0.25
N_TERMS = 80
KA_EPS = 1e-8
N_FFT = 16384
```

The sum includes orders 0 through `N_TERMS - 1`. The implementation assigns zero
for `ka <= KA_EPS`; phase at zero magnitude has no physical meaning.
Although historical low-level functions contain a near-field branch, the new
standalone explicitly rejects a near-field configuration. Near-field echoes are
not validated here.

### Material presets

Use the exact material name in `SELECTED_MATERIAL`. These are the retained presets,
not a claim of exhaustive validation of every material in the new standalone.

| Material | Density [kg/m³] | Compressional speed [m/s] | Shear speed [m/s] |
| --- | ---: | ---: | ---: |
| Beryllium | 1870 | 12890 | 8880 |
| Fused silica | 2200 | 5968 | 3764 |
| Heavy silicate, flint glass | 3880 | 3980 | 2380 |
| Armco iron | 7700 | 5960 | 3240 |
| Monel metal | 8900 | 5350 | 2720 |
| Aluminum | 2700 | 6420 | 3040 |
| Yellow brass | 8600 | 4700 | 2110 |
| Lucite | 1180 | 2680 | 1100 |
| Lead | 11340 | 1960 | 690 |
| Ice | 917 | 2743 | 1433 |

## 3. Incident sources

Choose `SOURCE_CASE` near the beginning of the script. Reference and general sources
share the same FFT/IFFT synthesis; only source parameters, the reference frequency
band, and the optional diagnostics differ.

### Hickling reference presets

| `SOURCE_CASE` | Cycles | $k_0a$ | Echo integration band in $ka$ |
| --- | ---: | ---: | --- |
| `hickling_fig16_max` | 5 | 24.5 | 10–40 |
| `hickling_fig16_min` | 5 | 25.5 | 10–40 |
| `hickling_fig17_max` | 25 | 24.5 | 15–35 |
| `hickling_fig17_min` | 25 | 25.5 | 15–35 |
| `hickling_fig18_max` | 50 | 24.5 | 15–35 |
| `hickling_fig18_min` | 50 | 25.5 | 15–35 |

The published echo presets refer to Armco iron. Selecting another material is an
experiment, not reproduction of those paper figures. The source frequency is
$f_0=(k_0a)c/(2\pi a)$, duration is $T=N/f_0$, and sample rate is $10f_0$.
The complex source is $\exp[i2\pi f_0(t-T/2)]$ for $0\le t<T$: a causal,
positive-exponential counterpart of Hickling's centered source.

### `tone_burst`

A general rectangular four-cycle burst at 14 kHz by default:

```python
TONE_FREQUENCY_HZ = 14_000.0
TONE_CYCLES = 4.0
```

It uses $\exp[i(2\pi f_0t-\pi/2)]$, whose real part is a sine. It differs from
the Hickling preset in frequency, duration, initial phase, and absence of the
paper-specific integration-band restriction. It is not a published echo preset.

### `lfm_chirp`

The defaults are 0.5–10 kHz over 20 ms, sampled at 30 kHz, with a Hann envelope.
The complex baseband representation uses a fixed reference frequency
$f_\mathrm{reference}=5.25$ kHz (named `baseband_frequency` in the code).
Thus the instantaneous baseband frequency runs from −4.75 to +4.75 kHz.

The physical-frequency grid and real passband view are obtained through

$$f_\mathrm{physical}=f_\mathrm{offset}+f_\mathrm{reference},$$

$$s_\mathrm{physical}(t)=\operatorname{Re}\{s_\mathrm{bb}(t)
e^{i2\pi f_\mathrm{reference}t}\}.$$

The real baseband trace appears to slow down near the midpoint because its
frequency crosses zero. The physical chirp increases monotonically from 0.5 to
10 kHz; the source figure shows both representations separately.

Other incident signals can be added through `build_source()` without changing the
modal solution or synthesis convention. Arbitrary-file loading is not currently
implemented. Finite rectangular complex bursts are not strictly one-sided
Hilbert analytic signals: their spectra have tails outside the nominal band.

## 4. FFT/IFFT pipeline and the required conjugation

The canonical path is:

```text
complex incident source
-> fftshift(fft(source))
-> physical-frequency grid
-> Hickling modal form function
-> explicit conjugation to the NumPy/OpenSTB convention
-> positive-physical-frequency mask (plus Hickling band for reference presets)
-> spectral multiplication
-> ifft(ifftshift(echo spectrum))
-> complex baseband echo and real passband view
```

The FFT grid is bilateral, but the applied target response is explicitly zero at
nonpositive physical frequencies. No Hermitian reconstruction, frequency-array
reversal, or extra factor of two is used.

### Why conjugate the form function?

Hickling's response is expressed with the time-harmonic convention
$e^{-i\omega t}$. His echo integral also synthesizes with a negative exponential.
NumPy uses the opposite synthesis sign:

| Convention | Forward transform | Inverse/synthesis |
| --- | --- | --- |
| Hickling | $e^{+i\omega t}$ | $e^{-i\omega t}$ |
| NumPy/OpenSTB | $e^{-i\omega t}$ | $e^{+i\omega t}$ |

For the same real physical response, changing this convention requires

$$f_\mathrm{OpenSTB}(\omega)=f_\mathrm{Hickling}(\omega)^*.$$

Conjugation preserves the magnitude and changes the sign of the phase. In the
controlled reference test, using the Hickling response directly with NumPy IFFT
reverses the target-response wave-packet ordering; the conjugated response restores
agreement with the direct Hickling integral. This is a convention conversion, not
a change to elastic physics or an empirical correction specific to Armco iron.

The conversion remains explicit at the synthesis boundary:

```python
ff_hickling = form_function_hickling(ka)
ff_openstb = np.conjugate(ff_hickling)
echo_spectrum = source_spectrum * response
```

Here `response` is the masked `ff_openstb`. The modal function and the standard
form-function plot retain Hickling's convention for paper comparison. Do not
conjugate the modal equations as well: that would cause double conversion.
Do not automatically apply this rule to the rigid sphere, whose modal convention
has been independently checked.

Pulse centering is a separate operation. Conjugation converts phase convention;
subtracting $T/2$ from the reference time coordinate aligns the causal source with
Hickling's centered pulse. Neither operation adds propagation travel time.

## 5. Figures and Hickling diagnostics

Normal execution generates three figures:

1. Incident waveform/envelope and normalized spectrum. For LFM, three stacked
   panels show baseband, physical passband, and spectrum.
2. Form-function magnitude and wrapped phase in $[0,2\pi)$, in Hickling convention.
3. Normalized real target-only echo and normalized scattered spectrum.

The form-function grid comes from the FFT settings. `FORM_FUNCTION_PLOT_KA_MAX`
sets a display limit, not additional computed frequencies; a curve can end before
that limit. The echo is plotted relative to the centered FFT origin, without a
range-dependent arrival time. Negative relative times are not negative physical
travel times. Long records can make the main packet look compressed; zooming is
useful for inspecting its structure.

For reference diagnostics set:

```python
SOURCE_CASE = "hickling_fig16_max"
SHOW_VALIDATION_PLOTS = True
COMPARE_DIRECT_HICKLING = True
```

`SHOW_VALIDATION_PLOTS` adds two Hickling-specific figures: incident spectral
magnitude versus Eq. (16), and the phase parameter used for paper comparison.
`COMPARE_DIRECT_HICKLING` additionally enables the echo comparison against direct
Eq. (14), only when `SHOW_VALIDATION_PLOTS` is enabled. These diagnostics are
skipped for `tone_burst` and `lfm_chirp`; there is no matching Hickling reference
echo for those general settings. Set `SHOW_VALIDATION_PLOTS = False` for just the
three normal figures.

### Reference equations

Hickling Eq. (15) is a centered pulse $e^{-i\omega_0t}$ for
$-\Delta t<t<\Delta t$. With $x=ka$, $x_0=k_0a$, and $N$ total cycles,
$\Delta\tau=N\pi/x_0$. Eq. (16) is evaluated as

$$g(x)=\frac{2}{\pi}\frac{\sin[(x-x_0)\Delta\tau]}{x-x_0}.$$

The prefactor is **$2/\pi$, not its square root**. `np.sinc` evaluates the removable
singularity at $x=x_0$ robustly. The spectral comparison uses normalized magnitudes;
it is not a full complex-spectrum comparison without time-origin conversion.

For Eq. (14), the direct diagnostic evaluates

$$P_e(u)\propto\int g(x)f_\infty(x)e^{-ixu}\,dx,
\qquad u=\tau-2R,$$

on the selected reference band using numerical quadrature. The FFT echo is aligned
to the pulse-centered coordinate; both real echo curves are independently
normalized, so this checks waveform shape and timing, not absolute received level.

### Wrapped phase versus Hickling's phase parameter

Hickling's phase comparison uses

$$-\frac{\operatorname{unwrap}(\arg f_\infty)}{ka},$$

not the wrapped argument shown in the normal figure. Wrapping produces jumps at
$2\pi$ boundaries; unwrapping and dividing by $ka$ produce a different-looking
plot of the same complex response. The optional phase figure is the appropriate
one for comparison with Hickling's phase plot.

## 6. Validation status and limitations

The October 2026 review covered Armco iron in far-field backscattering:

- Sampled complex form-function values from the new and historical implementations
  agreed exactly on the tested grid.
- The `hickling_fig16_max` magnitude and paper-specific phase plots were visually
  checked against Hickling; the conjugated FFT/IFFT echo closely overlapped the
  direct Eq. (14) echo, including packet ordering and the later tail.
- General four-cycle `tone_burst` execution and figures were reviewed successfully.
- The LFM source, echo, and spectrum were reviewed successfully. The user reported
  visual agreement of echo and spectrum with the historical menu implementation.
  This is historical concordance, not independent published chirp validation.

This does not establish exhaustive validation of every material, all six reference
presets, near-field scattering, or numerical convergence at arbitrary settings.
Larger frequency ranges or narrower resonances may require increased modal count,
sampling rate, FFT length, and convergence checks. `N_FFT` must cover the complete
source record; sufficient record duration also matters for avoiding circular
overlap of long response tails. Nonfinite modal values are replaced by zero in the
inherited implementation; finite output alone does not prove numerical accuracy.

The standalone has OpenSTB-compatible Fourier organization, but does not yet
validate a solid-sphere OpenSTB plugin. That integration is the next milestone.
Its controlled comparison must match source phase, sampling, physical-frequency
grid, material, geometry, and any frequency mask. A reference-band mask must not
be mistaken for an intrinsic band limit of the sphere.

Sources have unit-scale envelopes rather than calibrated source levels. Plotted
spectra and echoes are normalized. No spreading, absorption, beampattern, Doppler,
noise, seabed, motion, or propagation delay is included. Absolute received pressure
and arrival times require the corresponding OpenSTB geometry and effects.

## 7. Running the scripts

From the repository root, with NumPy, SciPy, and Matplotlib installed:

```powershell
.venv\Scripts\python.exe Experiments\scattering_models\solid_sphere\free_field\SolidSphereEchoAnalytic.py
```

For the historical workflow, run `SolidSphereEcho.py` instead. Both can be run
interactively with the VS Code play button. Configuration is edited at the top of
the script; there is no command-line case selector.

## Reference

Hickling, R. (1962). *Analysis of echoes from a solid elastic sphere in water*.
The Journal of the Acoustical Society of America.
[Tracked reference PDF](../../references/6.%20Analysis%20of%20echoes%20from%20a%20solid%20slastic%20sphere%20in%20water.pdf).
The PDF filename retains the historical `slastic` typo.
