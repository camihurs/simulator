import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn
import matplotlib.pyplot as plt
from tqdm import tqdm

#Esta fue la versión que me dio Claude y yo modifiqué.
#Da el mismo eco del paper. No calcula directamente g(ka),
#sino que la interpola a partir de la FFT del pulso incidente,
#Y además compara el eco calculado usando FFT con el método del trapecio.

# Función rect
def rect(x):
   return np.where(np.abs(x) <= 0.5, 1, 0)

# Función para calcular f(ka)
def f_ka(ka_range, theta=np.pi):
   """
   Calcula la función de forma f(ka) vectorizada
   """
   N_terms = 80
   f = np.zeros_like(ka_range, dtype=complex)
   cos_theta = np.cos(theta)

   for n in range(N_terms):
       j_n = spherical_jn(n, ka_range, True)
       y_n = spherical_yn(n, ka_range, True)
       eta_n = np.arctan(j_n / -y_n)
       f += (2*n + 1) * eval_legendre(n, cos_theta) * np.sin(eta_n) * np.exp(1j*eta_n)

   f = f * (2/ka_range)
   return -f

# Parámetros principales--------------------------------------------------------------
k0a = 15.0  # tamaño del pulso en ka
b = 2       # número de ciclos
m = 2 * b * np.pi / k0a  # longitud normalizada del pulso
duration = 2.5  # duración del pulso en tiempo normalizado
sample_freq = 102.4
dt = 1/sample_freq
fftpoints = 8192



# Generar el pulso temporal y su espectro---------------------------------------------
tau = np.linspace(0, duration, int(duration*sample_freq))
pi = rect(tau/m - 1/2) * np.sin(k0a*tau)

# Transformada de Fourier para obtener el espectro
pi_fft = 2 * np.fft.fft(pi, fftpoints) * dt
freqs = np.fft.fftfreq(fftpoints, dt)
k = 2 * np.pi * freqs

# Obtener g(ka) solo para ka > 0
k_positive = k[k >= 0]
g_k = np.abs(pi_fft[k >= 0])
phase_g_k = np.angle(pi_fft[k >= 0])
phase_g_k = np.mod(phase_g_k + 2*np.pi, 2*np.pi)

# Graficar el pulso temporal
plt.figure(figsize=(12, 6))
plt.plot(tau, pi, 'b-', linewidth=1.5)
plt.grid(True, alpha=0.3)
plt.xlabel("τ'")
plt.ylabel("$p_i(τ')$")
plt.title("Ideal Incident Pulseform")
plt.ylim(-1.1, 1.1)
plt.xlim(-0.5, 2.5)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.show()

# Graficar el espectro
plt.figure(figsize=(12, 8))

# Gráfica de la magnitud
plt.subplot(2, 1, 1)
plt.plot(k_positive, g_k, 'b-', linewidth=1.5)
plt.xlabel("$ka$")
plt.ylabel("$|g(ka)|$")
plt.title("Spectrum $g(ka)$ of a 2-cycle pulse with $k_0a = 15.0$")
plt.grid(True, alpha=0.3)
plt.xlim(0, 30)

# Gráfica de la fase
plt.subplot(2, 1, 2)
plt.plot(k_positive, phase_g_k, 'r-', linewidth=1.5)
plt.xlabel("$ka$")
plt.ylabel("Phase of $g(ka)$ (radians)")
plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
         ['0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])
plt.grid(True, alpha=0.3)
plt.xlim(0, 30)
plt.ylim(0, 2*np.pi)
plt.tight_layout()
plt.show()

# Parámetros para el cálculo del eco---------------------------------------------------
ka_max = 32*np.pi
n_points_tau = 500
tau_range = np.linspace(-3, 8, n_points_tau)
dka = 0.001 * np.pi
n_points_ka = int(ka_max/dka)
ka_range = np.linspace(0.01, ka_max, n_points_ka)

# Calcular f(ka)
print("Calculando función de forma f(ka)...")
f_ka_values = f_ka(ka_range)

# Interpolar g(ka) a los puntos necesarios
g_ka_values = np.interp(ka_range, k_positive, pi_fft[k >= 0].real) + \
             1j * np.interp(ka_range, k_positive, pi_fft[k >= 0].imag)


#Calcular el eco dispersado-------------------------------------------------------------
# Método 1: Trapecio (original)----------------------------------------------------------------
print("Calculando pulso dispersado (método trapecio)...")
scattered_pulse_trap = np.zeros(len(tau_range), dtype=complex)

for i, tau in enumerate(tqdm(tau_range)):
    integrand = g_ka_values * f_ka_values * np.exp(1j*ka_range*tau)
    scattered_pulse_trap[i] = np.sum(integrand) * dka

scattered_pulse_trap = np.real(scattered_pulse_trap)
scattered_pulse_trap = scattered_pulse_trap / np.max(np.abs(scattered_pulse_trap))


# Método 2: FFT
print("Calculando pulso dispersado (método FFT)...")
product = g_ka_values * f_ka_values  # Producto en el dominio de la frecuencia
fft_length = len(product)

# IFFT y normalización
scattered_pulse_fft = np.real(np.fft.ifft(product))
scattered_pulse_fft = scattered_pulse_fft / np.max(np.abs(scattered_pulse_fft))

# Crear vector de tiempo para la FFT
dt_fft = 2*np.pi/(fft_length*dka)
tau_fft_full = np.arange(-fft_length//2, fft_length//2)*dt_fft

# Reorganizar la señal
scattered_pulse_fft = np.roll(scattered_pulse_fft, fft_length//2)

# Interpolar al mismo espaciado que el método del trapecio
from scipy.interpolate import interp1d
f_interp = interp1d(tau_fft_full, scattered_pulse_fft, kind='cubic')
scattered_pulse_fft = f_interp(tau_range)

scattered_pulse_fft = f_interp(tau_range)
tau_fft = tau_range


# # Método 2: FFT
# print("Calculando pulso dispersado (método FFT)...")
# product = g_ka_values * f_ka_values  # Producto en el dominio de la frecuencia
# fft_length = len(product)  # Usamos el mismo número de puntos que tenemos en product

# # IFFT y normalización
# scattered_pulse_fft = np.real(np.fft.ifft(product))
# scattered_pulse_fft = scattered_pulse_fft / np.max(np.abs(scattered_pulse_fft))

# # Crear vector de tiempo para el resultado de la FFT
# dt_fft = 2*np.pi/(fft_length*dka)  # Paso temporal para la FFT
# tau_fft = np.arange(-fft_length//2, fft_length//2)*dt_fft

# # Reorganizar la señal para centrarla
# scattered_pulse_fft = np.roll(scattered_pulse_fft, fft_length//2)



# # Imprimir espaciado temporal del método del trapecio
# dt_trap = (tau_range[-1] - tau_range[0])/n_points_tau
# print(f"\nEspaciado temporal (Trapecio): {dt_trap:.6f}")

# # Imprimir espaciado temporal de la FFT
# print(f"Espaciado temporal (FFT): {dt_fft:.6f}")
# print(f"Número de puntos FFT: {len(scattered_pulse_fft)}")
# print(f"Número de puntos Trapecio: {len(scattered_pulse_trap)}")



# Graficar comparación
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(tau_range, scattered_pulse_trap, 'b-', label='Trapezoid', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('$\\tau$ VALUES', fontsize=12)
plt.ylabel('$\\Psi(\\tau)$', fontsize=12)
plt.ylim(-1.1, 1.1)
plt.xlim(-3, 8)
plt.title('Scattered Pulse (Trapezoid Method)', fontsize=14)
plt.legend()

plt.subplot(212)
plt.plot(tau_fft, scattered_pulse_fft, 'r-', label='FFT', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('$\\tau$ VALUES', fontsize=12)
plt.ylabel('$\\Psi(\\tau)$', fontsize=12)
plt.ylim(-1.1, 1.1)
plt.xlim(-3, 8)
plt.title('Scattered Pulse (FFT Method)', fontsize=14)
plt.legend()

plt.tight_layout()
plt.show()