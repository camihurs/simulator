# Rigid-Sphere Scattering in Free Field

This directory contains standalone implementations of the far-field acoustic response
of an ideally rigid sphere immersed in a fluid. The scripts calculate the complex form
function and synthesize the echo produced by an incident pulse without the additional
propagation and platform effects present in OpenSTB.

The reference model is A. J. Rudgers' classical solution for acoustic pulses scattered
by a rigid sphere. The implementation is restricted to free-field scattering.

## 1. Available standalones

| File | Purpose |
| --- | --- |
| `RigidSphereEcho.py` | Historical exploratory script used for the original reproduction of Rudgers and the validation reported in the elastic-scattering plugins paper. |
| `RigidSphereEchoAnalytic.py` | Clean general-purpose standalone using the same analytic-signal and FFT/IFFT convention as OpenSTB. This is the recommended script for new experiments. |

The historical script is preserved because it records the original scientific
validation path. The analytic script was not created to correct the rigid-sphere modal
physics: both scripts and the OpenSTB plugin calculate the same complex form function.
It was created to provide a clearer and reusable source-to-echo pipeline for arbitrary
incident signals and to serve as an architectural reference for other target models.

## 2. Physical model

### 2.1 Assumptions

The current standalone model assumes:

- an ideally rigid sphere;
- a homogeneous fluid with sound speed `c`;
- free-field conditions;
- far-field observation;
- a selectable scattering angle, with `theta = pi` representing monostatic
  backscattering;
- no propagation loss, absorption, Doppler, transducer beampattern, noise, or platform
  motion.

The dimensionless frequency is

$$
ka = \frac{2\pi f a}{c},
$$

where $f$ is physical frequency, $a$ is sphere radius, and $c$ is sound speed.

### 2.2 Form function

The far-field form function corresponds to Eq. (11) of Rudgers:

$$
f(ka) = -\frac{2}{ka}
\sum_{n=0}^{\infty}
(2n+1)P_n(\cos\theta)\sin\eta_n(ka)e^{i\eta_n(ka)},
$$

where $P_n$ is the Legendre polynomial of order $n$ and the rigid-sphere modal phase
shift is evaluated numerically from

$$
\eta_n = \operatorname{atan2}\left(j_n'(ka),-y_n'(ka)\right).
$$

The numerical implementation truncates the series after a configurable number of
modes (`N_TERMS = 80` by default). The `atan2` expression avoids unstable direct
division when the denominator approaches zero.

### 2.3 Behaviour at `ka = 0`

The analytical expression contains the factor $2/(ka)$, but the rigid-sphere form
function tends to zero in the Rayleigh limit. The standalone therefore evaluates the
modal expression using a small positive threshold and explicitly assigns

$$
f(0)=0.
$$

The phase at exactly `ka = 0` is undefined because the magnitude is zero. NumPy reports
the argument of complex zero as zero; this convention has no physical effect on the
echo.

## 3. Historical Rudgers standalone

`RigidSphereEcho.py` reproduces the reference case used during the original model
development:

- $k_0a=15$;
- a two-cycle real truncated sine;
- monostatic far-field backscattering;
- sphere radius $a=0.25$ m;
- sound speed $c=1480$ m/s.

The corresponding centre frequency is

$$
f_0 = \frac{(k_0a)c}{2\pi a} \approx 14.13\ \text{kHz},
$$

and the pulse duration is approximately 0.142 ms.

This script follows the original validation workflow: it constructs a real causal
sine, retains the positive-frequency part of its FFT, applies the historical factor of
two, multiplies by the form function, and uses an IFFT-based synthesis. It also contains
diagnostics accumulated during comparison with the OpenSTB plugin dump.

That procedure remains useful for reproducing the historical figures, but it should
not be interpreted as the canonical pipeline for arbitrary analytic sources. In
particular, a one-sided real-source synthesis and a full-grid analytic-signal synthesis
need not produce point-for-point identical secondary features.

## 4. Analytic standalone

`RigidSphereEchoAnalytic.py` uses one canonical pipeline for both reference and general
sources:

```text
analytic complex source
-> bilateral fftshift(fft(...))
-> physical frequency grid
-> complex rigid-sphere form function
-> direct multiplication S(f) * f(ka)
-> ifft(ifftshift(...))
-> complex baseband echo
-> optional real passband view
```

This is the same Fourier organization used by OpenSTB. The standalone does not reverse
frequency arrays, reconstruct a Hermitian spectrum manually, or introduce a separate
echo-synthesis convention for the reference case.

### 4.1 Source cases

Select the source near the beginning of the file:

```python
SOURCE_CASE = "rudgers_fig6"
```

Available cases are:

#### `rudgers_fig6`

A fixed, reproducible preset with $k_0a=15$ and two cycles. The analytic source is the
complex equivalent of Rudgers' physical sine:

$$
s_a(t)=e^{i(2\pi f_0t-\pi/2)},
$$

whose real part is $\sin(2\pi f_0t)$ inside the pulse duration.

#### `tone_burst`

A configurable analytic fixed-frequency burst. Set:

```python
TONE_FREQUENCY_HZ = 14_000.0
TONE_CYCLES = 4.0
```

This case is intended for general experiments and is not automatically a Rudgers
reference case.

#### `lfm_chirp`

A configurable analytic linear-frequency-modulated chirp. The defaults match the
OpenSTB experiment source:

```python
CHIRP_START_HZ = 500.0
CHIRP_STOP_HZ = 10_000.0
CHIRP_DURATION_S = 0.020
CHIRP_SAMPLE_RATE_HZ = 30_000.0
CHIRP_USE_HANN_WINDOW = True
```

The chirp is represented in complex baseband around 5.25 kHz. Its instantaneous
baseband frequency varies from -4.75 to +4.75 kHz and crosses zero at the pulse
midpoint. This can make the real part of the baseband signal appear to decrease to zero
frequency and increase again. The physical passband frequency does not reverse: it
increases monotonically from 0.5 to 10 kHz.

For this reason, the source figure for `lfm_chirp` contains three panels:

1. the real part of the complex baseband signal and its envelope;
2. the reconstructed physical passband chirp and its envelope;
3. the spectrum plotted against physical frequency.

### 4.2 Normal figures

With all optional comparisons disabled, the analytic standalone produces three
figures:

1. incident source in time and frequency;
2. form-function magnitude and wrapped phase;
3. target-only echo and scattered spectrum.

The echo time is relative to the FFT origin because the standalone does not add a
propagation range or travel-time delay.

### 4.3 Optional validation figures

The validation controls are:

```python
SHOW_VALIDATION_PLOTS = True
COMPARE_DIRECT_RUDGERS = True
COMPARE_OPENSTB_DUMP = True
```

For `rudgers_fig6`, the first option adds a comparison between:

- the spectrum obtained from the analytic-source FFT;
- the spectrum obtained from Rudgers' defining integral for the real truncated sine.

`COMPARE_DIRECT_RUDGERS` additionally compares the canonical FFT/IFFT echo with a
direct numerical evaluation of Rudgers Eq. (8). These Rudgers-specific figures are not
shown for `tone_burst` or `lfm_chirp`, because different source parameters would not be
a valid comparison with the published reference case.

`COMPARE_OPENSTB_DUMP` compares the standalone form function with the values generated
by the OpenSTB `RigidSphereFormFunction` plugin. This comparison is independent of the
selected source because it evaluates only the target response on the dump's `ka` grid
and at its stored scattering angle.

## 5. Validation results and interpretation

### 5.1 Form function

The standalone form function reproduces the magnitude and phase of Rudgers' Fig. 2.
When evaluated on the controlled OpenSTB dump grid, the analytic standalone and the
OpenSTB plugin agree numerically in both magnitude and wrapped phase. The comparison
excludes only the DC convention: the standalone assigns the physical limit `f(0)=0`,
whereas the plugin evaluates the expression at `ka_eps`.

### 5.2 Rudgers echo

The principal echo packet from the analytic FFT/IFFT synthesis agrees closely with the
direct evaluation of Rudgers Eq. (8), including its temporal orientation and dominant
amplitudes. This reproduces the essential features of Rudgers' Fig. 6.

The analytic echo contains secondary peaks that are absent from Rudgers' real-source
echo. The same peaks appear in the OpenSTB result reported in the elastic-scattering
plugins paper and disappear when the historical real incident signal is used. Since
the form function is identical between the standalone and OpenSTB, the working
interpretation is that these features arise from the finite-duration analytic complex
source and its full FFT-grid spectral construction, not from an error in the rigid-
sphere modal response.

The analytic/OpenSTB-compatible echo is therefore not expected to be point-for-point
identical to Rudgers' real-source, positive-frequency synthesis, even though the form
function and the principal physical echo components agree.

## 6. Controlled comparison with OpenSTB

The optional comparison reads:

```text
Experiments/simple_points_study/rigid_sphere_ff_debug.npz
```

This file is generated by running `Experiments/simple_points_study/simulate.py`. It is
a valid controlled rigid-sphere reference dump only when the experiment is configured
as follows:

1. Select the `sine` source in `shared_params.py`.
2. Keep only `RigidSphereFormFunction` enabled.
3. Disable the other propagation/distortion plugins.
4. Use stop-and-hop travel time.
5. Use the identity quaternion for transducer orientation.
6. Disable/comment both transmit and receive beampatterns.

The dump must contain compatible one-dimensional arrays for `f_hz`, `ka`,
`ff_complex`, and `S_in_complex`. A dump produced under another experiment
configuration should not be interpreted as the controlled Rudgers comparison.

The standalone reads the scattering angle saved in the dump. It can differ slightly
from exactly $\pi$ because it comes from the numerical experiment geometry. Recomputing
the standalone form function at that stored angle permits a direct point-by-point
comparison with the plugin.

## 7. Running the scripts

From the repository root, run either script with the project virtual environment:

```powershell
.venv\Scripts\python.exe Experiments\scattering_models\rigid_sphere\free_field\RigidSphereEchoAnalytic.py
```

or, for the historical validation:

```powershell
.venv\Scripts\python.exe Experiments\scattering_models\rigid_sphere\free_field\RigidSphereEcho.py
```

The scripts are also designed to be run with the VS Code play button, which is the
normal interactive plotting workflow for this project.

## 8. Scope and limitations

- Only the free-field far-field rigid-sphere response is validated here.
- The standalone computes the target response only; it is not a replacement for the
  complete OpenSTB pipeline.
- Absolute received levels and arrival times require propagation geometry and the
  relevant OpenSTB distortions.
- The default modal truncation and FFT settings are suitable for the documented cases,
  but substantially larger `ka` ranges may require a convergence review.
- The optional OpenSTB comparison is meaningful only when the dump was generated with
  the controlled configuration listed above.

## References

1. Rudgers, A. J. (1969). "Acoustic Pulses Scattered by a Rigid Sphere Immersed in a
   Fluid." *The Journal of the Acoustical Society of America*, 45(4), 900-910.
   <https://doi.org/10.1121/1.1911567>
2. Hurtado Erasso, C. A., Bonnett, B., Lopera Tellez, O., Lambot, S., and Neyt, X.
   (2026). "Elastic Scattering Plugins for the OpenSTB Sonar Simulator." *Proceedings
   of the Institute of Acoustics*, 48(1).
