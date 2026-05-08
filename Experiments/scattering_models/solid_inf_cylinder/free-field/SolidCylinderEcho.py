#Script iniciado el 1 de abril de 2025
#Vamos a calcular el eco de un cilindro rígido, usando tres tipos de señales incidentes:
#1. Señal seno
#2. Señal Ricker
#3. Señal chirp

#Voy a basarme en el script 25.AcousticScattering_Parallel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import chirp
from scipy.special import jv, yv, jvp, yvp
import re
from scipy.signal.windows import hann
from tqdm import tqdm



def hankel1(n, x):
    """
    Función de Hankel de primer tipo: H_n^(1)(x) = J_n(x) + i*Y_n(x)
    """
    return jv(n, x) + 1j * yv(n, x)



def hankel1p(n, x):
    """
    Derivada de la función de Hankel de primer tipo
    """
    return jvp(n, x, 1) + 1j * yvp(n, x, 1)



def stft(signal, fs, L=2, R=None):
    """
    Calcula la STFT de una señal
    Args:
        signal: señal de entrada
        fs: frecuencia de muestreo
        L: longitud de la ventana en milisegundos (window length)
        R: tamaño del paso en muestras (step size)
    """
    # Convertir L de ms a muestras
    window_length = int(L * 1e-3 * fs)

    # Si R no se especifica, usar 10% de L (90% overlap)
    if R is None:
        R = window_length // 10

    # Crear ventana Hanning
    window = hann(window_length)

    # Calcular número de frames
    num_frames = 1 + (len(signal) - window_length) // R

    # Preparar matriz para almacenar STFT
    nfft = 1024
    stft_matrix = np.zeros((nfft//2 + 1, num_frames), dtype=complex)

    # Calcular STFT
    for i in range(num_frames):
        # Extraer segmento
        start = i * R  # Usar R directamente como step size
        segment = signal[start:start + window_length]

        # Aplicar ventana
        windowed_segment = segment * window

        # Calcular FFT
        spectrum = np.fft.rfft(windowed_segment, n=nfft)

        # Almacenar en matriz
        stft_matrix[:, i] = spectrum

    # Calcular frecuencias y tiempos
    frequencies = np.fft.rfftfreq(nfft, d=1/fs)
    times = np.arange(num_frames) * R / fs

    return frequencies, times, stft_matrix



def ModesInfSolidCyl(n):

    epsilon_n = 1 if n == 0 else 2
    fn_array = np.zeros(freqs.size, dtype=complex)

    for i in range(len(freqs)):

        d11 = (rho1/rho2) * ks2[i] ** 2 * a ** 2 * hankel1(n, k1[i] * a)
        d12 = -2 * kd2[i] * a * jvp(n, kd2[i] * a) + (2 * n ** 2 - ks2[i] ** 2 * a ** 2) * jv(n, kd2[i] * a)
        d13 = 2 * n * (ks2[i] * a * jvp(n, ks2[i] * a) - jv(n, ks2[i] * a))

        d21 = -k1[i] * a * hankel1p(n, k1[i] * a)
        d22 = kd2[i] * a * jvp(n, kd2[i] * a)
        d23 = n * jv(n, ks2[i] * a)

        d31 = 0
        d32 = 2 * n * (jv(n, kd2[i] * a) - kd2[i] * a * jvp(n, kd2[i] * a))
        d33 = 2 * ks2[i] * a * jvp(n, ks2[i] * a) + (ks2[i] ** 2 * a ** 2 - 2 * n ** 2) * jv(n, ks2[i] * a)

        A1ast = -(rho1/rho2) * ks2[i] ** 2 * a ** 2 * jv(n, k1[i] * a)
        A2ast = k1[i] * a * jvp(n, k1[i] * a)


        Bn_matrix = np.array([
        [A1ast, d12, d13],
        [A2ast, d22, d23],
        [0, d32, d33]
        ])

        Dn_matrix = np.array([
        [d11, d12, d13],
        [d21, d22, d23],
        [0, d32, d33]
        ])

        Bn_det = np.linalg.det(Bn_matrix)
        Dn_det = np.linalg.det(Dn_matrix)

        bn = Bn_det/Dn_det
        #fn_array[i] = (2/(np.sqrt(1j * np.pi * k1[i] * a))) * ((-1) ** n) * epsilon_n * bn
        fn_array[i] = ((-1) ** n) * epsilon_n * bn

    return fn_array


#--------------------------------------------------------------------------


def SinusoidDefaultParameters():
    default_f0 = 8000.0
    user_input = input(f"\nEnter the central frequency of the Sine (in Hz - Press Enter to use default value = {default_f0}): ")
    f0 = float(user_input) if user_input.strip() else default_f0

    default_duration = 0.25e-3
    duration = input(f"Enter the duration of the Sine (in seconds - Press Enter to use default value = {default_duration}): ")
    duration = float(duration) if duration.strip() else default_duration

    default_sampling_rate = 128e3
    sampling_rate = input(f"Enter the sampling rate of the Sine (in Hz - Press Enter to use default value = {default_sampling_rate}): ")
    sampling_rate = float(sampling_rate) if sampling_rate.strip() else default_sampling_rate
    return f0, duration, sampling_rate



def sinusoid_source(f0, duration, sample_freq):
    """Generates a 40 kHz sinusoidal signal with silent periods before and after."""
    points_time = int(duration * sample_freq)  # Points for the active sinusoid
    t = np.linspace(0, duration, points_time, endpoint=False)
    signal = np.sin(2 * np.pi * f0 * t)
    return signal, t



def ChirpDefaultParameters():
    default_f_start = 500
    user_input = input(f"\nEnter the start frequency of the chirp (in Hz - Press Enter to use default value = {default_f_start}): ")
    f_start = float(user_input) if user_input.strip() else default_f_start

    default_f_end = 10000
    f_end = input(f"Enter the end frequency of the chirp (in Hz - Press Enter to use default value = {default_f_end}): ")
    f_end = float(f_end) if f_end.strip() else default_f_end

    default_duration = 0.02
    duration = input(f"Enter the duration of the chirp (in seconds - Press Enter to use default value = {default_duration}): ")
    duration = float(duration) if duration.strip() else default_duration

    default_sampling_rate = 128000
    sampling_rate = input(f"Enter the sampling rate (in Hz - Press Enter to use default value = {default_sampling_rate}): ")
    sampling_rate = float(sampling_rate) if sampling_rate.strip() else default_sampling_rate


    choice = input("\nDo you want to apply a Hanning window to the incident chirp signal? (1: Yes, 2: No): ")
    if choice == '1':
        Hanning = True
    else:
        Hanning = False

    choice = input("Do you want the chirp from Min to Max frequency or from Max to Min frequency? (1: Min to Max, 2: Max to Min): ")
    if choice == '1':
        MintoMax = True
    else:
        MintoMax = False

    return f_start, f_end, duration, sampling_rate, Hanning, MintoMax



def chirp_source(f_start, f_end, duration, sampling_rate, MintoMax, Hanning):

    f_central = (f_end+f_start)/2

    t = np.linspace(0, duration, int(sampling_rate * duration))

    if MintoMax == 'True' or MintoMax == True:
        chirp_signal = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    else:
        chirp_signal = chirp(t, f0=f_end, f1=f_start, t1=duration, method='linear')

    if Hanning == 'True' or Hanning == True:
        window = hann(len(chirp_signal))
        chirp_signal = chirp_signal * window

    return chirp_signal, t, f_central,



def RickerDefaultParameters():
    default_f0 = 8000.0
    user_input = input(f"\nEnter the central frequency of the Ricker (in Hz - Press Enter to use default value = {default_f0}): ")
    f0 = float(user_input) if user_input.strip() else default_f0

    default_duration = 0.0007
    duration = input(f"Enter the duration of the Ricker (in seconds - Press Enter to use default value = {default_duration}): ")
    duration = float(duration) if duration.strip() else default_duration

    default_sampling_rate = 128e3
    sampling_rate = input(f"Enter the sampling rate of the Ricker (in Hz - Press Enter to use default value = {default_sampling_rate}): ")
    sampling_rate = float(sampling_rate) if sampling_rate.strip() else default_sampling_rate

    return f0, duration, sampling_rate



def ricker_source(f0, duration, sample_freq):

    points_time = int(duration * sample_freq)

    t = np.linspace(0, duration, points_time)

    term1 = 1 - 2 * (np.pi ** 2) * (f0 ** 2) * (t - duration / 2) ** 2
    term2 = np.exp(- (np.pi ** 2) * (f0 ** 2) * (t - duration / 2) ** 2)
    ricker = term1 * term2

    return ricker, t
#---------------------------------------------------------------------------

def PrintSummarySourceEcho(
        source, duration, sampling_rate, fft_points, f0, f_start = None, f_end = None, MintoMax = None, Hanning = None):
    print(f"\nSummary of the source signal and echo:")
    if 'Chirp' in source:
        print(f"\nDuration of the echo: {fft_points/sampling_rate:.5f} s")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Min to Max frequency: {MintoMax}")
        print(f"Duration of the {source}: {duration} s")
        print(f"Central frequency of the {source}: {f0} Hz")
        print(f"Bandwidth of the {source}: {f_end - f_start} Hz")
        print(f"Frequency range of the {source}: {f_start} to {f_end} Hz")
        print(f"Number of points in the FFT: {fft_points}")
        print(f"Hanning window applied: {Hanning}")
    else:
        if 'Ricker' in source:
            Source = 'Ricker'
        else:
            Source = 'Sine'
        print(f"\nDuration of the echo: {fft_points/sampling_rate:.5f} s")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Duration of the {Source}: {duration} s")
        print(f"Central frequency of the {Source}: {f0} Hz")
        print(f"Number of points in the FFT: {fft_points}")



def extract_upper_frequency(source):

    if 'Chirp' in source:
        match = re.search(r'to(\d+)kHz', source)
        if match:
            # Extraemos el número y lo convertimos a entero
            return int(match.group(1))*1000
        else:
            raise ValueError(f"Upper frequency not found in : {source}")

    elif 'Ricker' in source:
        pattern = r'(\d+)kHz$'
        match = re.search(pattern, source)
        if match:
            return 2.5 * int(match.group(1))*1000
        else:
            raise ValueError(f"Central frequency not found in : {source}")

    elif 'Sine' in source:
        duration_pattern = r'Sine(\d+\.?\d*)ms'
        duration_match = re.search(duration_pattern, source)

        f0_pattern = r'(\d+)kHz$'
        f0_match = re.search(f0_pattern, source)

        if duration_match and f0_match:
            duration = float(duration_match.group(1))/1000  # convertir a segundos
            f0 = int(f0_match.group(1)) * 1000  # convertir a Hz

            BW = 2/duration

            # Calcular frecuencias límite
            #f_lower = f0 - BW/2
            f_upper = f0 + BW/2
            return f_upper
        else:
            raise ValueError(f"Duration or central frequency not found in : {source}")



def GetFormFunctionEcho(f):

    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f_conj = np.conjugate(f[1:])
    f_conj_inv = f_conj[::-1]
    f_final = np.concatenate([f[:-1], f_conj_inv])

    return f_final



def GetEchoFF(S_f, f_final, fft_points, target):

    if 'Rigid' in target:
        echo_spectrum = S_f * f_final[::-1] # np.exp(1j * k1a/a * r) # No podemos multiplicar acá la exponencial debido a diferencias en el tamaño de k1a y los otros dos (f_final y S_f, los cuales tienen el doble de tamaño que k1a).
    else:
        echo_spectrum = S_f * f_final[::-1] #Eco al derecho

    echo = ifft(echo_spectrum, fft_points)
    echo_normalized = echo / np.max(np.abs(echo))

    # Normalización por raíz de la potencia
    #echo_power = np.mean(np.abs(echo)**2)
    #echo_normalized = echo / np.sqrt(echo_power)

    return echo, echo_normalized


#---------------------------------------------------------------------------------------
def PlotSource(source, source_signal, filtered_freqs, filtered_spectrum, f0, t, spectrum_dB, cutoff_freq, f_start = None, f_end = None):

    choice = input(f"\nDo you want to plot the source ({source}) in time and in frequency? (1: Yes, 2: No): ")
    if choice == '1':
        print("Plotting source signal in time domain...")
        # Graficamos los resultados sin normalizar la amplitud
        plt.figure(figsize=(10, 5))
        plt.plot(t * 1000, source_signal)  # Convertimos el tiempo a milisegundos para que coincida con el gráfico
        plt.title(f'{source}') # from {int(f_start)} to {int(f_end)} Hz')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        print("Plotting source signal in frequency domain...")
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_freqs, np.abs(filtered_spectrum))
        plt.title(f"FFT Spectrum of {source}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        if 'Chirp' in source:
            plt.xlim(int(f_start), int(f_end))
        elif 'Ricker' in source:
            plt.xlim(int(f0 - 20e3), int(f0 + 20e3))
        else:
            plt.xlim(int(f0 - 10e3), int(f0 + 10e3))
        #plt.ylim(0, 5)
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()

        print("Plotting source signal in frequency domain in dB...")
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_freqs/1000, spectrum_dB, label=f'Positive FFT (up to {cutoff_freq} Hz)')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Relative level (dB)')
        plt.title(f'Spectral response of {source} signal')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.clf()
        plt.close()
    else:
        print(f"{source} signal not plotted.")



def PlotFormFunction(f, ka, target, condition, params, fmin, fmax, frequency_points, xmin, xmax):
    print("\n...............................................................................")
    choice = input(f"Do you want to plot the form function for the {target} in {condition}? (1: Yes, 2: No): ")
    if choice == '1':
        # Graficar la magnitud de f en función de ka
        f_magnitude = np.abs(f)
        plt.figure(figsize=(12, 6))
        info_text = (f'fmin: {fmin:.1f} Hz\n'
                    f'fmax: {fmax:.2f} Hz\n'
                    f'Points in frequency: {frequency_points}\n'
                    #f'Slant range (r): {params["r"]:.2f} m\n'
                    #f'Thickness of the target: {params["h"]*100:.2f} % \n'
                    f'Sound speed in water (c1): {params["c1"]} m/s\n'
                    f'Surrounding fluid (Medium 1): {params["medium1"]}\n'
                    f'Number of modes used (n_max): {params["n_max"]}')

        plt.plot(ka, f_magnitude, label='Form function')
        plt.text(0.98, 0.02, info_text,
                transform=plt.gca().transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',  # Alinea el texto a la derecha
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('ka')
        plt.ylabel('Magnitude of form function')
        plt.title(f'Form function $(F_{{\\infty}})$ of a {target}\n (radius: {params["a"]:.3f} m) in {condition}', fontsize=18)
        plt.grid()
        plt.show()
        plt.clf()
        plt.close()


        #Phase of form function--------------------------------------------------------------------------------------------
        #f_phase = np.angle(f, deg=True)  # Convert to degrees for clarity
        #Phase of form function--------------------------------------------------------------------------------------------
        # Obtener la fase en radianes (entre -π y π)
        f_phase = np.angle(f)  # Por defecto está en radianes
        # Convertir a rango de 0 a 2π
        f_phase = f_phase + np.pi

        plt.figure(figsize=(12, 6))
        info_text = (f'fmin: {fmin:.1f} Hz\n'
            f'fmax: {fmax:.2f} Hz\n'
            f'Points in frequency: {frequency_points}\n'
            #f'Slant range (r): {params["r"]:.2f} m\n'
            #f'Thickness of the target: {params["h"]*100:.2f} % \n'
            f'Sound speed in water (c1): {params["c1"]} m/s\n'
            f'Surrounding fluid (Medium 1): {params["medium1"]}\n'
            f'Number of modes used (n_max): {params["n_max"]}')
        plt.plot(ka, f_phase)
        plt.text(0.98, 0.02, info_text,
            transform=plt.gca().transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('ka', fontsize=18)
        plt.ylabel('Phase (rad)', fontsize=18)
        if any(word in target for word in ['rigid', 'solid', 'vacuum']):
            plt.title(f'Phase of the form function of a {target}\n (radius: {params["a"]:.3f} m) in {condition}', fontsize=18)
        else:
            plt.title(f'Phase of the form function of a {target}\n (radius: {params["a"]} m, thickness (h) = {params["h"] * 100:.1f} %) in {condition}', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.xlim(xmin, xmax)
        plt.ylim(0, 2*np.pi)
        plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
                ['0', 'π/2', 'π', '3π/2', '2π'])
        plt.grid()
        plt.show()
        plt.clf()
        plt.close()
    else:
        print(f"You chose not to compute the form function for the {target} sphere in {condition}.")



def PlotEcho(echo_normalized, sampling_rate, source, field, params, target, condition, check = False):

    choice = input(f"\nDo you want to plot the echo for the {target} in {condition}? (1: Yes, 2: No): ")
    if choice == '1':
        n_max = params['n_max']

        t = np.linspace(0, len(echo_normalized) / sampling_rate, len(echo_normalized))
        specular_echo  = (r - params['a'])*1000/params['c1']
        #back_face_echo  = (r + 2*params['a'])*1000/cs2
        back_face_echo = (r - params['a'])*1000/params['c1'] + 2 * params['a']*1000/cd2
        plt.figure(figsize=(12, 6))
        info_text = (f'Source: {source}\n'
                    f'fft points: {fft_points}\n'
                    f'Sampling rate: {sampling_rate} Hz\n'
                    f'Slant range (r): {r:.2f} m\n'
                    f'Field: {field}\n'
                    f'Surrounding fluid (Medium 1): {params["medium1"]}\n'
                    f'Number of modes used (n_max): {n_max}')
        plt.plot(t*1000, np.real(echo_normalized))
        #plt.plot(t*1000, echo_normalized) #da igual que ponga np.real o no
        plt.text(0.98, 0.02, info_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='bottom',
                    horizontalalignment='right',  # Alinea el texto a la derecha
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel('One-way Travel Time (ms)', fontsize = 18)
        plt.ylabel('Normalized amplitude', fontsize = 18)
        plt.title(f'Acoustic backscattering of a {target} (radius: {params["a"]:.3f} m) in {condition}', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.grid()
        plt.axvline(specular_echo, color='red', linestyle='--', label=f'Specular echo = {specular_echo:5f} ms')
        plt.axvline(back_face_echo, color='green', linestyle='--', label=f'Back face echo = {back_face_echo:5f} ms')
        plt.legend()
        plt.show()
        plt.clf()
        plt.close()
    else:
        print(f"You chose not to plot the echo for the {target} in {condition}.")



#---------------------------------------------------------------------------------------
fft_points = 15000


choice = input("\nWhat type of source do you want to use? (1: Chirp, 2: Ricker, 3: Sinusoid): ")
if choice == '1':
    f_start, f_end, duration, sampling_rate, Hanning, MintoMax = ChirpDefaultParameters()
    source_signal, t, f0 = chirp_source(f_start, f_end, duration, sampling_rate, MintoMax, Hanning)
    source = f'Chirp{duration*1000:.2f}ms{f_start/1000}to{int(f_end/1000)}kHz{Hanning}Hanning{MintoMax}MintoMax'
    PrintSummarySourceEcho(source, duration, sampling_rate, fft_points, f0, f_start, f_end, MintoMax, Hanning)
elif choice == '2':
    f0, duration, sampling_rate = RickerDefaultParameters()
    source_signal, t = ricker_source(f0, duration, sampling_rate)
    source = f'Ricker{duration*1000:.2f}ms{int(f0/1000)}kHz'
    f_start, f_end = None, None
    PrintSummarySourceEcho(source, duration, sampling_rate, fft_points, f0)
else:
    f0, duration, sampling_rate = SinusoidDefaultParameters()
    source_signal, t = sinusoid_source(f0, duration, sampling_rate)
    source = f'Sine{duration*1000:.2f}ms{int(f0/1000)}kHz'
    f_start, f_end = None, None
    PrintSummarySourceEcho(source, duration, sampling_rate, fft_points, f0)


#1. COMPUTE THE SOURCE SPECTRUM--------------------------------------------------------------
S_f = fft(source_signal, fft_points)
freqs = fftfreq(fft_points, d=1/sampling_rate)

# Compute the Fourier transform, with shift, positive frequencies and then filter below 200 kHz
FT_p_source = np.fft.fftshift(S_f)
shifted_freqs = np.fft.fftshift(freqs)
print(f"\nThe minimum frequency is: {shifted_freqs[0]}")
print(f"The maximum frequency is: {shifted_freqs[-1]}")
# Filter only positive frequencies
positive_freqs = shifted_freqs[shifted_freqs >= 0]
FT_p_source_positive = FT_p_source[shifted_freqs >= 0]
# Create mask to cut frequencies
cutoff_freq = extract_upper_frequency(source)
#cutoff_freq = 25e3#200e3  # Cutoff frequency in Hz
mask = positive_freqs <= cutoff_freq
# Apply mask
filtered_freqs = positive_freqs[mask]
filtered_spectrum = FT_p_source_positive[mask]
spectrum_dB = 20 * np.log10(np.abs(filtered_spectrum))


PlotSource(source, source_signal, filtered_freqs, filtered_spectrum, f0, t, spectrum_dB, cutoff_freq, f_start, f_end)


freqs_positive = np.abs(freqs[:fft_points//2 + 1])
freqs_positive[0] += 1e-10  # Avoid division by zero in the next step
print(f"\nThe FREQUENCY STEP to get the form function with which the echo is computed is: {freqs_positive[2] - freqs_positive[1]} Hz")
#Imprimir las primeras 10 frecuencias
print("The first 10 frequencies are:")
print(freqs_positive[:15])

r = 20
a = 0.25

n_max_solid = 80
field = 'FarField'

rho2 = 2.79e3 #density of the shell
cd2 = 6380 #longitudinal wave speed of the shell
cs2 = 3100 #transverse wave speed of the shell

rho1 = 1000 #density of the surrounding fluid
c1 = 1476 #speed of sound in the surrounding fluid

k1a = 2 * np.pi * a * freqs_positive / c1
k1a_min = 2 * np.pi * a * freqs_positive[0] / c1
k1a_max = 2 * np.pi * a * freqs_positive[-1] / c1


fmin = c1 * k1a_min /(2 * np.pi * a)
fmax = c1 * k1a_max /(2 * np.pi * a)
freqs = np.linspace(fmin, fmax, k1a.size)
omega = 2 * np.pi * freqs

k1 = k1a/a #Comparar con otros códigos
ks2 = omega / cs2
kd2 = omega / cd2
phi = np.pi

print(f"\nThe minimum k1a is: {k1a[0]}")
print(f"The maximum k1a is: {k1a[-1]}")
print("\nStep size in ka:", k1a[1] - k1a[0])

print("\nComputing the echo for the infinite solid cylinder in free field...")

target = 'SolidCylinder'
condition = 'FF'

# Calcular la forma función total para el cilindro sólido
f = np.zeros(freqs.size, dtype=complex)


#f = ModesInfSolidCyl(5)

for n in tqdm(range(n_max_solid), total=n_max_solid, desc="Modes"):
    fn = ModesInfSolidCyl(n)
    f += fn

#f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
f = (2/np.sqrt(1j * np.pi * k1a)) * f * np.exp(1j * k1 * r)

params = {}
params["a"] = a
params["c1"] = c1
params["medium1"] = "water"
params["n_max"] = n_max_solid
PlotFormFunction(f, k1a, "solid cylinder", "free field (computed to get the echo)", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(k1a))


f_final_solid = GetFormFunctionEcho(f)
echo_solid, echo_solid_normalized = GetEchoFF(S_f, f_final_solid, fft_points, target)

PlotEcho(echo_solid_normalized, sampling_rate, source, field, params, target, condition)



#---------------------------Recortar el eco--------------------------------------------------
# Recortar el eco hasta 40 ms
max_time_ms = 45  # tiempo máximo en milisegundos
max_samples = int(max_time_ms * 1e-3 * sampling_rate)  # convertir ms a segundos y luego a muestras

# Verificar que no estemos intentando tomar más muestras de las disponibles
if max_samples > len(echo_solid_normalized):
    print(f"Advertencia: Solicitaste recortar a {max_time_ms} ms, pero la señal solo tiene {len(echo_solid_normalized)/sampling_rate*1000:.2f} ms")
    max_samples = len(echo_solid_normalized)

# Recortar el eco
echo_rigid_normalized_truncated = echo_solid_normalized[:max_samples]

# Crear vector de tiempo recortado
t_truncated = np.linspace(0, max_samples / sampling_rate, max_samples)

print(f"Echo recortado a {max_time_ms} ms ({max_samples} muestras)")
PlotEcho(echo_rigid_normalized_truncated, sampling_rate, source, field, params, target, condition)

#---------------------------------------------------------------------------------------
#------------Echo in frequency domain--------------------------------------------------

# Compute the Fourier Transform of the echo
echo_spectrum = fft(echo_solid_normalized, fft_points)
freqs = fftfreq(fft_points, d=1/sampling_rate)

# Shift the Fourier Transform for visualization
FT_p_echo = np.fft.fftshift(echo_spectrum)
shifted_freqs = np.fft.fftshift(freqs)
# Filter only positive frequencies
positive_freqs = shifted_freqs[shifted_freqs >= 0]
FT_p_echo_positive = FT_p_echo[shifted_freqs >= 0]
# Create mask to cut frequencies
cutoff_freq = extract_upper_frequency(source)
mask = positive_freqs <= cutoff_freq
# Apply mask
filtered_freqs = positive_freqs[mask]
filtered_spectrum = FT_p_echo_positive[mask]


plt.figure(figsize=(10, 6))
info_text = (f'Source: {source}\n'
            f'fft points: {fft_points}\n'
            f'Sampling rate: {sampling_rate} Hz\n'
            f'Slant range (r): {r} m\n'
            f'Field: {field}\n'
            f'Surrounding fluid (Medium 1): {params["medium1"]}\n'
            f'Number of modes used (n_max): {n_max_solid}')
plt.plot(filtered_freqs, np.abs(filtered_spectrum))
plt.text(0.98, 0.02, info_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='bottom',
                    horizontalalignment='right',  # Alinea el texto a la derecha
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.title(f"FFT Spectrum of the echo from a {target} (radius: {a:.3f} m) in {condition}", fontsize=18)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


#---------------------------------------------------------------------------------------
#------------Echo in time-frequency domain--------------------------------------------------
f, t_spec, Sxx = stft(echo_rigid_normalized_truncated, sampling_rate, 2, R=10)  # L en milisegundos
plt.figure(figsize=(10, 6))

info_text = (f'Source: {source}\n'
            f'fft points: {fft_points}\n'
            f'Sampling rate: {sampling_rate} Hz\n'
            f'Slant range (r): {r} m\n'
            f'Field: {field}\n'
            f'Surrounding fluid (Medium 1): {params["medium1"]}\n'
            f'Number of modes used (n_max): {n_max_solid}')
plt.title(f'Time-Frequency Representation of {target} in {condition}')

plt.pcolormesh(t_spec*1000, f, 20*np.log10(np.abs(Sxx)), shading='gouraud')
plt.text(0.98, 0.02, info_text,
            transform=plt.gca().transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',  # Alinea el texto a la derecha
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.ylabel('Frequency (kHz)')
plt.xlabel('Time (ms)')
plt.colorbar(label='Magnitude (dB)')
plt.ylim(0, cutoff_freq)
plt.show()
