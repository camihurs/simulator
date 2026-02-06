# Rigid Sphere Scattering in Free Field

## 1. Parameter Selection

This implementation reproduces Figure 6 from Rudgers (1968): "Acoustic Pulses Scattered by a Rigid Sphere Immersed in a Fluid".

### 1.1. Paper Parameters (Normalized)

The paper uses normalized variables:
- **k₀a = 15.0** (size parameter for the amplitude of the incident signal)
- **b = 2** (number of cycles of the incident sine signal)
- **Geometry**: Monostatic backscattering in the far field (θ = π)

### 1.2. Physical Parameters

To convert from normalized to physical parameters:

**Given:**
- Sphere radius: **a = 0.25 m**
- Sound speed: **c = 1480 m/s**
- Size parameter from paper: **k₀a = 15.0**

**Calculate center frequency:**

$$k_0 = \frac{2\pi f_0}{c}$$

$$k_0 a = \frac{2\pi f_0}{c} \times a = 15.0$$

$$f_0 = \frac{15.0 \times c}{2\pi \times a} = \frac{15.0 \times 1480}{2\pi \times 0.25} \approx 14{,}133 \text{ Hz}$$

**Pulse parameters:**
- Center frequency: **f₀ ≈ 14.1 kHz**
- Number of cycles: **2**
- Pulse duration: **T = 2/f₀ ≈ 0.141 ms**
- Wavelength: **λ = c/f₀ ≈ 0.105 m**

### 1.3. Validation

The normalized pulse length from the paper is:
$$m = \frac{2b\pi}{k_0a} = \frac{4\pi}{15} \approx 0.838$$

This represents the pulse duration in units of a/c (time for sound to travel one radius).

Physical pulse length: **L = cT ≈ 0.209 m ≈ 0.838a** ✓


## 2. Implementation Details


### 2.1. Form function definition

The form function implemented in this script corresponds **explicitly to Equation (11)** in Rudgers (1968).

For monostatic backscattering in the far field (θ = π), the form function is computed as:

$$
f(ka) = -\frac{2}{ka} \sum_{n=0}^{\infty} (2n+1)\, P_n(\cos\theta)\, \sin\eta_n(ka)\, e^{i\eta_n(ka)}
$$

where:
- $P_n(\cdot)$ are Legendre polynomials,
- $\eta_n(ka)$ are the modal phase shifts for a rigid sphere,
- The summation is truncated to a finite number of modes in the numerical implementation.

This expression is evaluated in the frequency domain to obtain the complex-valued form function \( f(ka) \), which is then used in the synthesis of the scattered pulse following Equation (8) of the paper.


### 2.2. Pulse Synthesis Method: FFT vs. Trapezoidal Integration

The scattered pulse is synthesized using Equation (8) from the paper:

$$\Psi(\tau) = \frac{1}{2\pi} \text{Re} \int_0^{\infty} g(ka) f(ka) e^{ika\tau} d(ka)$$

**Paper's approach (1968):**
- Direct numerical integration using the **trapezoidal rule**
- Computed for discrete values of τ
- FFT algorithms were not yet widely available

**Our implementation:**
- Uses **Inverse Fast Fourier Transform (IFFT)**
- Mathematically equivalent to Eq. 8
- Computationally more efficient

**Why FFT is equivalent:**

The IFFT computes:
$$\text{IFFT}\{G[k]\} = \frac{1}{N}\sum_{k=0}^{N-1} G[k] e^{i 2\pi kn/N}$$

When we multiply the spectra `g(ka) × f(ka)` in the frequency domain and apply IFFT, we're computing the same integral as Eq. 8, but using the **convolution theorem** and efficient FFT algorithms instead of direct numerical integration.

**Result:** Both methods produce identical scattered pulseforms (within numerical precision), but FFT is significantly faster for large datasets.


### 2.3. Spectral Analysis

#### 2.3.1. Factor of 2 in FFT
The incident pulse spectrum is computed using:
```python
incident_fft = np.fft.fft(incident_signal, n_fft) * dt
incident_fft *= 2  # Factor of 2 for positive frequencies only
```

**Why the factor of 2?**

In the paper, the spectrum is defined as a Fourier integral over all time (Eq. 4):
$$g(ka) = \frac{2}{p_0} \int_{-\infty}^{\infty} p_i(\tau') e^{-ika\tau'} d\tau'$$

However, when synthesizing the scattered pulse (Eq. 8), the integration is only over **positive ka values** (positive frequencies):
$$\Psi(\tau) = \frac{1}{2\pi} \text{Re} \int_0^{\infty} g(ka) f(ka) e^{ika\tau} d(ka)$$

When we take only the positive frequencies from a two-sided FFT, we must multiply by 2 to conserve energy. This is because the negative frequencies (which are complex conjugates for real signals) contain the same energy as the positive frequencies.

**Physical interpretation**: The real signal contains energy at both +f and -f frequencies. By keeping only positive frequencies and doubling, we account for the total energy.

#### 2.3.2. Form Function at ka = 0

The form function is computed for ka > 0, and we prepend a zero value:
```python
f_ka = compute_form_function(ka_positive[1:], theta=np.pi)
f_ka = np.concatenate([[0], f_ka])
```

**Why exclude ka = 0 from the computation?**

From Eq. 9 in the paper, the form function includes the term:
$$f(ka) = -\frac{2}{ka} \sum_{n=0}^{\infty} (\ldots)$$

This creates a **division by zero** at ka = 0.

**Why set f(0) = 0?**

Physically, when ka → 0 (sphere radius ≪ wavelength), the scattering cross-section vanishes. The Rayleigh scattering limit shows that scattering efficiency scales as (ka)⁴ for very small spheres, confirming that f(ka) → 0 as ka → 0.

Mathematically, examining Eq. 9 term by term, each term in the sum behaves as:
$$(2n+1) P_n(\cos\theta) \sin\eta_n e^{i\eta_n}$$

As ka → 0, the phase shifts $η_n$ → 0, making $sin(η_n) → 0$, and thus the entire sum vanishes.

**Why do the phase shifts $η_n → 0$?**

From Eq. 10, the phase shifts are defined as:
$$\tan(\eta_n) = -\frac{(n+1)j_n(ka) - (ka)j_{n-1}(ka)}{(n+1)y_n(ka) - (ka)y_{n-1}(ka)}$$

Using the recurrence relation for spherical Bessel functions:
$$j_n'(x) = j_{n-1}(x) - \frac{n+1}{x}j_n(x)$$

We can show that:
$$(n+1)j_n(ka) - (ka)j_{n-1}(ka) = -(ka)j_n'(ka)$$

And similarly for $y_n$. This is why the code uses derivatives directly.

**Asymptotic behavior as ka → 0:**

The spherical Bessel functions behave as:
- **$j_n$ (regular at origin)**: j₀(ka) → 1, j₁(ka) → $\frac{ka}{3}$, $j_n$(ka) ~ (ka)ⁿ
- **$y_n$ (singular at origin)**: y₀(ka) → $\frac{-1}{ka}$, y₁(ka) → $\frac{-1}{ka²}$, $y_n$(ka) ~ $\frac{-1}{(ka)^{n+1}}$

For the derivatives:
- $j_n$'(ka) → 0 (or small values proportional to ka)
- $y_n$'(ka) → -∞ (diverges)

Therefore:
$$\tan(\eta_n) = \frac{j_n'(ka)}{-y_n'(ka)} \approx \frac{\text{small}}{-\infty} \rightarrow 0$$

Since tan($η_n$) → 0, then **$η_n$ → 0**, making $sin(η_n)$ → 0, and thus f(ka) → 0 as ka → 0.


### 2.4. Form Function Phase at ka = 0

The phase plot of the form function f(ka) shows a minor discrepancy at ka = 0 compared to Figure 2 in the paper:
- **Our implementation**: Phase starts at 0 radians
- **Paper Figure 2**: Phase starts at π radians

**Why this difference occurs:**

In our implementation, we set f(0) = 0 (complex zero) to avoid division by zero in Eq. 9. When computing the phase:
```python
phase_f = np.angle(f_ka)  # For f_ka[0] = 0+0j
```

NumPy's `np.angle(0+0j)` returns 0 by convention.

**Why this difference doesn't matter:**

1. **Magnitude is zero**: Since |f(0)| = 0, the value at ka = 0 makes **no contribution** to the scattered field, regardless of its phase.

2. **Phase is undefined**: Mathematically, the phase (argument) of a complex number with zero magnitude is undefined. Any phase value would be equally valid.

3. **Physical interpretation**: At ka = 0 (wavelength >> sphere radius), the scattering cross-section vanishes. The sphere is acoustically invisible, so the phase of the (non-existent) scattered wave is meaningless.

4. **For ka > 0**: Our phase values match the paper's Figure 2 perfectly, showing the characteristic sawtooth pattern with jumps at specific ka values.

**Conclusion**: The difference at ka = 0 is purely cosmetic and has no effect on the computed scattered pulse.


## References

Rudgers, A. J. (1968). "Acoustic Pulses Scattered by a Rigid Sphere Immersed in a Fluid." *The Journal of the Acoustical Society of America*, 45(4), 900-910.