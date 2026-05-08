import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import spherical_jn, spherical_yn, eval_legendre, jvp, yvp
from scipy.fft import fft, fftfreq, ifft
import warnings
import pandas as pd
import os
from scipy.signal import chirp, convolve, spectrogram, correlate
from scipy.signal.windows import hann
import tkinter as tk
from tkinter import filedialog
import re
import h5py
import plotly.graph_objects as go
#import segyio
import matplotlib.colors as mcolors
from tkinter.filedialog import askopenfilename
from tkinter import Tk


warnings.filterwarnings("ignore", category=RuntimeWarning)

#Variable used for the segy files
SPEED_OF_SOUND = 1486  # in meters per second. Used for the reading of the segy files


#MAXIMUM NUMBER OF CHARACTERS PER LINE: 100 (PEP 8)
#Menus------------------------------------------------------------------------------------------
def MainMenu():
    print("\n........................................Acoustic Scattering Simulator V1.5.......................................")

    print("\nWelcome to the first open source underwater acoustic scattering simulator." \
    "\nThe FIRST part is used to verify the results of the software.\
    It gives results that the user can compare with scientific papers.\nThe SECOND part computes the\
    form functions and synthetic echoes for different spherical and cylindrical shapes")

    print("\n__________________________________\n                                   MAIN MENU")
    print("\n-----------------------FIRST SECTION: VERIFICATION OF THE SIMULATOR----------------------")
    print("0. VERIFICATION: Check the operation of the simulator. ")

    print("\n---------------SECOND SECTION: FORM FUNCTIONS AND SYNTHETIC ECHOES-----------------")
    print("1. FORM FUNCTIONS FOR SPHERICAL SHAPES: Generate and plot Form Functions for different spherical shapes.")
    print("3. ECHOES FROM SPHERICAL SHAPES: Generate synthetic echoes for different spherical shapes and save the results.")

    print("15. Exit")



def menuSphericalFormFunctions():
    print("__________________________________\nFORM FUNCTIONS MENU")
    print("1. Rigid sphere")
    print("2. Elastic solid sphere.")
    print("3. Vacuum-filled elastic spherical shell.")
    print("4. Fluid-filled elastic spherical shell.")
    print("5. Exit.")



def menuEchoesSpherical():
    print("__________________________________\nECHOES MENU")
    print("1. Generate Echo for the rigid sphere.")
    print("2. Generate Echo for the elastic solid sphere.")
    print("3. Generate Echo for the vacuum-filled elastic spherical shell.")
    print("4. Generate Echo for the fluid-filled elastic spherical shell.")
    print("5. Exit.")



def menuReferenceFormFunEchoes():
    print("__________________________________\nVERIFICATION  MENU")
    print("\nPlease select the option you want to check:")
    print("1. Form function for a rigid sphere in free field.")
    print("2. Form function for an elastic solid sphere in free field.")
    print("3. Form function for a vacuum-filled elastic spherical shell in free field.")
    print("4. Form function for a fluid-filled elastic spherical shell in free field.")
    print("5. Echo from a rigid sphere in free field using a Sine incident signal.")
    print("6. Echo from a fluid-filled spherical shell in free field using a Ricker incident signal.")
    print("7. Echo from a fluid-filled spherical shell in free field using a Chirp incident signal.")
    print("8. Exit.")



def SaveData(filename, echo_normalized, sampling_rate, source, field, params, target, condition):

    choice = input("Do you want to save the data? (1: Yes, 2: No): ")

    if choice == '1':
        params = {
            'sonar_position': params['sonar_position'],
            'target_position': params['target_position'],
            'sampling_rate': sampling_rate,
            'source': source,
            'field': field,
            'target': target,
            'condition': condition,
            'r': params['r'],
            'material_sphere': params['material_sphere'],
            'inner_material': params['inner_material'],
            'medium1': params['medium1'],
            'rho1': params['rho1'],
            'c1': params['c1'],
            'rho2': params['rho2'],
            'cd2': params['cd2'],
            'cs2': params['cs2'],
            'a': params['a'],
            'b': params['b'],
            'h': params['h'],
            'rho3': params['rho3'],
            'c3': params['c3'],
            'sediment': params['sediment'],#-----
            'rho4': params['rho4'],#-----
            'c4': params['c4'],#-----
            'delta': params['delta'],#-----
            'n_max_fluid_filled': params['n_max_fluid_filled'],
            'n_max_rigid': params['n_max_rigid'],
            'n_max_solid': params['n_max_solid'],
            'n_max_vacuum': params['n_max_vacuum'],
            'fft_points': params['fft_points'],
            'theta_i': params['theta_i'],
            'xmin': params['xmin'],
            'xmax': params['xmax'],
            'fmin': params['fmin'],
            'fmax': params['fmax'],
            'frequency_points': params['frequency_points'],
        }


        #output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', '7. DataMarch2025', filename))
        output_path = os.path.abspath(os.path.join(r'C:\Users\camih\Documents\4. Python\3. OCEANS 2026', filename))
        with h5py.File(output_path, 'w') as f:
                # Guarda el eco
                f.create_dataset('eco', data=echo_normalized, compression='gzip')

                # Guarda los parámetros
                params_group = f.create_group('parameters')
                for key, value in params.items():
                    params_group.attrs[key] = value

        print(f"Data saved to {output_path}")
    else:
        print("You chose not to save the data.")


#------------------------------------------------------------------------------------------
#Option 0
def CheckSimulator():
    print("\n...............................................................................")
    print("\nYou are now in the CheckSimulator function.")

    while(True):
        menuReferenceFormFunEchoes()
        option = input("\nSelect an option: ")

        if option == "1":
            CheckRigidSphereFormFunction()
        elif option == "2":
            CheckSolidSphereFormFunction()
        elif option == "3":
            CheckVacuumSphereFormFunction()
        elif option == "4":
            CheckFluidSphereFormFunction()
        elif option == "5":
            CheckEchoRigidSine()
        elif option == "6":
            CheckEchoFluidRicker()
        elif option == "7":
            CheckEchoFluidChirp()
        elif option == "8":
            print("Exiting verification.")
            break
        else:
            print("Invalid option, please try again.")



def SetReferenceParameters(default_freq_points, default_max_modes, default_r, default_xmax, default_a, default_c1):

    user_input = input(f"\nEnter the number of points in frequency you want to compute for the form\
 function (Press Enter to use default value = {default_freq_points}): ")
    freq_points = int(user_input) if user_input.strip() else default_freq_points

    user_input = input(f"Enter the number of modes you want to compute for the form function (Press\
 Enter to use default value = {default_max_modes}): ")
    max_modes = int(user_input) if user_input.strip() else default_max_modes

    user_input = input(f"Enter the maximum ka you want to compute for the form function (Press Enter\
 to use default value = {default_xmax}): ")
    xmax = int(user_input) if user_input.strip() else default_xmax

    user_input = input(f"Enter the radius of the spherical target (Press Enter to use default value\
 = {default_a}): ")
    a = float(user_input) if user_input.strip() else default_a

    fmax = xmax * default_c1 / (2 * np.pi * a)

    return freq_points, max_modes, xmax, a, fmax



def CheckRigidSphereFormFunction():

    print("\n...............................................................................")
    print("\nYou are now in the CheckRigidSphereFormFunction function.")

    print("\nThe reference paper is: Rudgers, A. J. (1969). Acoustic pulses scattered by a rigid sphere\
 immersed in a fluid. The Journal of the Acoustical Society of America, 45(4), 900-910.")

    print("The paper uses the far field expression of the form function. We use the same approach\
 to get exactly the same image, despite the simulator is capable of computing the general form function\
 (near and far field).")

    print("The paper does not specify the speed of sound in water neither the radius of the sphere.\
 We use 1500 m/s as speed of sound in water, and 0.53 m as the radius of the sphere.\
 Since it is a rigid sphere, its material properties are not needed. The thickness of the sphere is not important,\
 since the model is for a rigid sphere. The paper does not mention either\
 the number of modes and the number of points in frequency used.")

    medium1 = "Water"
    default_c1 = 1500
    default_h = 0 #Does not matter for this case
    default_a = 0.53

    default_r = 1.73
    default_theta_i = 0 #Does not matter for this case

    default_max_modes, xmin, default_xmax, fmin, default_fmax, default_freq_points =\
         DefaultFormFunctionParametersRigidSphere(default_c1, default_a)

    print("\nYou can change the default parameters for the form function. Be aware that doing so may generate\
 a different result from the reference paper.")
    result = SetReferenceParameters(default_freq_points, default_max_modes, default_r, default_xmax, default_a, default_c1)
    freq_points = result[0]
    max_modes = result[1]
    xmax = result[2]
    a = result[3]
    fmax = result[4]

    params = {
        'fmin': fmin,
        'fmax': fmax,
        'frequency_points': freq_points,
        'xmin': xmin,
        'xmax': xmax,
        'n_max': max_modes,
        'c1': default_c1,
        'a': a,
        'theta_i': default_theta_i,
        'r': default_r,
        'h': default_h,
        'medium1' : medium1
    }

    print("The final parameters are:")
    print(params)
    print(f"\nThe FREQUENCY STEP  to get the form function is: {(fmax-fmin)/freq_points} Hz")

    print("\nChecking the form function for a rigid sphere in free field...")
    ComputeFormFunctionRigid(params, True)



def CheckSolidSphereFormFunction():

    print("\n...............................................................................")
    print("\nYou are now in the CheckSolidSphereFormFunction function.")

    print("\nThe reference paper is: Hickling, R. (1962). Analysis of echoes from a solid elastic\
 sphere in water. the Journal of the Acoustical Society of America, 34(10), 1582-1592.")

    print("\nThe paper uses the far field expression of the form function. We use the same approach\
 to get exactly the same image, despite the simulator is capable of computing the general form function\
 (near and far field).")

    print("The paper presents results for several materials. We will show the one corresponding to\
 Beryllium. The paper does not specify the radius of the sphere. We will use 0.25 m. The thickness of\
 the sphere is not important,\n since the model is for a solid sphere. The paper does not mention either\
the number of modes and the number of points in frequency used.")

    medium1 = "Water"
    default_rho1 = 1000
    default_c1 = 1410
    default_material_sphere = 'Beryllium'
    default_rho2 = 1870
    default_cd2 = 12890
    default_cs2 = 8880
    default_a = 0.25
    default_h = 0

    default_r = 1.73
    default_theta_i = 0 #It's not important for this case

    default_max_modes, xmin, default_xmax, fmin, default_fmax, default_freq_points =\
         DefaultFormFunctionParametersSolidSphere(default_c1, default_a)

    print("\nYou can change the default parameters for the form function. Be aware that doing so may generate\
 a different result from the reference paper.")
    result = SetReferenceParameters(default_freq_points, default_max_modes, default_r, default_xmax, default_a, default_c1)
    freq_points = result[0]
    max_modes = result[1]
    xmax = result[2]
    a = result[3]
    fmax = result[4]

    params = {
        'fmin': fmin,
        'fmax': fmax,
        'frequency_points': freq_points,
        'xmin': xmin,
        'xmax': xmax,
        'n_max': max_modes,
        'c1': default_c1,
        'a': a,
        'r': default_r,
        'rho1': default_rho1,
        'rho2': default_rho2,
        'material_sphere': default_material_sphere,
        'cd2': default_cd2,
        'cs2': default_cs2,
        'h': default_h,
        'theta_i': default_theta_i,
        'medium1' : medium1
    }

    print("The final parameters are:")
    print(params)
    print(f"\nThe FREQUENCY STEP  to get the form function is: {(fmax-fmin)/freq_points} Hz")


    print(f"\nChecking the form function for an elastic {default_material_sphere} sphere in free field.")

    ComputeFormFunctionSolid(params, True)



def CheckVacuumSphereFormFunction():

    print("\n...............................................................................")
    print("\nYou are now in the CheckVacuumSphereFormFunction function.")

    print("\nThe reference paper is: Kargl, S. G., & Marston, P. L. (1991). Ray synthesis of the form\
 function for backscattering from an elastic spherical shell: Leaky Lamb waves and longitudinal resonances.\
 The Journal of the Acoustical Society of America, 89(6), 2545-2558.")

    print("\nThe paper uses the far field expression of the form function. We use the same approach\
 to get exactly the same image, despite the simulator is capable of computing the general form function\
 (near and far field). The paper does not specify the radius of the sphere. We will use 0.15 m. The paper\
 does not mention either the number of modes and the number of points in frequency used.")

    medium1 = "Water"
    default_rho1 = 1000
    default_c1 = 1479
    default_material_sphere = '440c stainless steel'
    default_rho2 = 7840
    default_cd2 = 5854
    default_cs2 = 3150
    default_a = 0.15
    default_b = 0.838 * default_a
    default_h = 1-default_b/default_a
    default_rho_tilde = default_rho1/default_rho2

    default_r = 1.73
    default_theta_i = 0 #It's not important for this case

    default_max_modes, xmin, default_xmax, fmin, default_fmax, default_freq_points =\
         DefaultFormFunctionParametersVacuumFilledSphere(default_c1, default_a)

    print("\nYou can change the default parameters for the form function. Be aware that doing so may generate\
 a different result from the reference paper.")
    result = SetReferenceParameters(default_freq_points, default_max_modes, default_r, default_xmax, default_a, default_c1)
    freq_points = result[0]
    max_modes = result[1]
    xmax = result[2]
    a = result[3]
    fmax = result[4]

    default_b = 0.838 * a


    params = {
        'fmin': fmin,
        'fmax': fmax,
        'frequency_points': freq_points,
        'xmin': xmin,
        'xmax': xmax,
        'n_max': max_modes,
        'c1': default_c1,
        'a': a,
        'b': default_b,
        'r': default_r,
        'rho1': default_rho1,
        'rho2': default_rho2,
        'rho_tilde' : default_rho_tilde,
        'material_sphere': default_material_sphere,
        'cd2': default_cd2,
        'cs2': default_cs2,
        'h': default_h,
        'theta_i': default_theta_i,
        'medium1' : medium1
    }

    print("\nThe final parameters are:")
    print(params)
    print(f"\nThe FREQUENCY STEP  to get the form function is: {(fmax-fmin)/freq_points} Hz")


    print(f"\nChecking the form function for a vacuum-filled {default_material_sphere} spherical shell in free field.")

    ComputeFormFunctionVacuum(params, True)



def CheckFluidSphereFormFunction():

    print("\n...............................................................................")
    print("\nYou are now in the CheckFluidSphereFormFunction function.")

    print("\nThe reference paper is: Ayres, V. M., Gaunaurd, G. C., Tsui, C. Y., & Werby, M. F. (1987).\
 The effects of Lamb waves on the sonar cross-sections of elastic spherical shells. International journal\
 of solids and structures, 23(7), 937-946.")

    print("\nThe paper uses the far field expression of the form function. We use the same approach\
 to get exactly the same image, despite the simulator is capable of computing the general form function\
 (near and far field). The paper does not specify the radius of the sphere. We will use 0.25 m. The paper\
 does not mention either the number of modes and the number of points in frequency used.")


    medium1 = "Water"
    default_rho1 = 1000
    default_c1 = 1500
    default_material_sphere = 'Tungsten Carbide (WC)'
    default_rho2 = 13100
    default_cd2 = 6950
    default_cs2 = 3940
    default_a = 0.25
    default_b = 0.99 * default_a
    default_h = 1-default_b/default_a
    default_inner_material = 'Air'
    default_rho3 = 1.225
    default_c3 = 344

    default_r = 1.73
    default_theta_i = 0 #It's not important for this case

    default_max_modes, xmin, default_xmax, fmin, default_fmax, default_freq_points =\
         DefaultFormFunctionParametersFluidFilledSphere(default_c1, default_a)

    print("\nYou can change the default parameters for the form function. Be aware that doing so may generate\
 a different result from the reference paper.")
    result = SetReferenceParameters(default_freq_points, default_max_modes, default_r, default_xmax, default_a, default_c1)
    freq_points = result[0]
    max_modes = result[1]
    xmax = result[2]
    a = result[3]
    fmax = result[4]

    default_b = 0.99 * a


    params = {
        'fmin': fmin,
        'fmax': fmax,
        'frequency_points': freq_points,
        'xmin': xmin,
        'xmax': xmax,
        'n_max': max_modes,
        'c1': default_c1,
        'a': a,
        'b': default_b,
        'r': default_r,
        'rho1': default_rho1,
        'rho2': default_rho2,
        'material_sphere': default_material_sphere,
        'cd2': default_cd2,
        'cs2': default_cs2,
        'h': default_h,
        'inner_material': default_inner_material,
        'rho3': default_rho3,
        'c3': default_c3,
        'theta_i': default_theta_i,
        'medium1' : medium1
    }

    print("\nThe final parameters are:")
    print(params)
    print(f"\nThe FREQUENCY STEP  to get the form function is: {(fmax-fmin)/freq_points} Hz")


    print(f"\nChecking the form function for a {default_inner_material}-filled {default_material_sphere} spherical shell in free field.")

    ComputeFormFunctionFluid(params, True)



def CheckEchoRigidSine():

    print("\n...............................................................................")
    print("\nYou are now in the CheckEchoRigidSine function.")


    print("\nThe reference paper is: Rudgers, A. J. (1969). Acoustic pulses scattered by a rigid sphere\
 immersed in a fluid. The Journal of the Acoustical Society of America, 45(4), 900-910.")

    print("The paper uses the far field expression of the form function. We use the general expression.")

    print("The paper does not specify the speed of sound in water neither the radius of the sphere.\
 We use 1500 m/s as speed of sound in water, and 0.53 m as the radius of the sphere.\
 Since it is a rigid sphere, its material properties are not needed.")

    print("The document does not mention either the distance between the sonar and the target,\
 For this test, we will use a distance of 1.73 meter. The thickness of the sphere is not important.")

    #Form function-------------------------------------------------------------------------------------
    medium1 = "Water"
    default_c1 = 1500
    default_h = 0
    default_a = 0.53

    default_r = 1.73
    default_theta_i = 0 #It's not important for this case

    default_max_modes, xmin, default_xmax, fmin, default_fmax, default_freq_points =\
         DefaultFormFunctionParametersRigidSphere(default_c1, default_a)

    print("\nYou can change the default parameters for the form function. Be aware that doing so may generate\
 a different result from the reference paper. In this case, changing the number of points in frequency and the\
 maximum ka will not affect the computation of the echo, since these parameters depend on the characteristics of the incident signal.")
    result = SetReferenceParameters(default_freq_points, default_max_modes, default_r, default_xmax, default_a, default_c1)
    freq_points = result[0]
    max_modes = result[1]
    xmax = result[2]
    a = result[3]
    fmax = result[4]


    params = {
        'fmin': fmin,
        'fmax': fmax,
        'frequency_points': freq_points,
        'xmin': xmin,
        'xmax': xmax,
        'n_max': max_modes,
        'c1': default_c1,
        'a': a,
        'theta_i': default_theta_i,
        'r': default_r,
        'h': default_h,
        'medium1' : medium1,
        'fft_points' : 15000
    }

    #Source--------------------------------------------------------------------------------------------
    print("\nYou can change the default parameters for the sine incident signal. Be aware that doing so may generate\
 a different result from the reference paper.")
    f0, duration, sampling_rate = SinusoidDefaultParameters()
    source_signal, t = sinusoid_source(f0, duration, sampling_rate)
    source = f'Sine{duration*1000:.2f}ms{int(f0/1000)}kHz'

    S_f = fft(source_signal, params['fft_points'])
    freqs = fftfreq(params['fft_points'], d=1/sampling_rate)
    rayleigh_distance, field = RayleighDistance(params['a'], f0, params['c1'], params['r'])
    freqs_positive = np.abs(freqs[:params['fft_points']//2 + 1])


    #---------------------------------------------------------------------------------------------------
    print("The final parameters are:")
    PrintSummarySourceEcho(source, duration, sampling_rate, params['fft_points'], f0)
    print(params)
    print(f"\nThe FREQUENCY STEP  to get the form function with which the echo is computed is: {freqs_positive[1] - freqs_positive[0]} Hz")


    print("\nChecking the echo from a rigid sphere in free field using a sine incident signal.")

    GenerateEchoesRigid(S_f, freqs_positive, sampling_rate, source, field, params, True)



def CheckEchoFluidRicker():

    print("\n...............................................................................")
    print("\nYou are now in the CheckEchoFluidRicker function.")

    print("\nThe reference paper is: Tesei, A., Fawcett, J. A., & Lim, R. (2008). Physics-based\
 detection of man-made elastic objects buried in high-density-clutter areas of saturated\
 sediments. Applied Acoustics, 69(5), 422-437.")

    print("The paper uses the general expression (near and far field) of the form function as we do.")

    medium1 = "Water"
    default_rho1 = 1000
    default_c1 = 1500
    default_material_sphere = 'Steel'
    default_rho2 = 7700
    default_cd2 = 5950
    default_cs2 = 3240
    default_a = 0.53
    default_b = 0.5
    default_h = 1-default_b/default_a
    default_inner_material = 'Air'
    default_rho3 = 1.225
    default_c3 = 344

    default_r = 2
    default_theta_i = 0 #It's not important for this case
    target = f'{default_inner_material}-filled{default_material_sphere}SphShellReferencePaper'
    condition = 'FF'

    default_max_modes, xmin, default_xmax, fmin, default_fmax, default_freq_points =\
         DefaultFormFunctionParametersFluidFilledSphere(default_c1, default_a)

    print("\nYou can change the default parameters for the form function. Be aware that doing so may generate\
 a different result from the reference paper. In this case, changing the number of points in frequency and the\
 maximum ka will not affect the computation of the echo, since these parameters depend on the characteristics of the incident signal.")
    result = SetReferenceParameters(default_freq_points, default_max_modes, default_r, default_xmax, default_a, default_c1)
    freq_points = result[0]
    max_modes = result[1]
    xmax = result[2]
    a = result[3]
    fmax = result[4]

    default_h = 1-default_b/default_a

    params = {
        'fmin': fmin,
        'fmax': fmax,
        'frequency_points': freq_points,
        'xmin': xmin,
        'xmax': xmax,
        'n_max': max_modes,
        'c1': default_c1,
        'a': a,
        'b': default_b,
        'r': default_r,
        'rho1': default_rho1,
        'rho2': default_rho2,
        'material_sphere': default_material_sphere,
        'cd2': default_cd2,
        'cs2': default_cs2,
        'h': default_h,
        'inner_material': default_inner_material,
        'rho3': default_rho3,
        'c3': default_c3,
        'theta_i': default_theta_i,
        'medium1' : medium1,
        'fft_points' : 15000,
        'condition': condition,
        'target': target
    }

    #Source--------------------------------------------------------------------------------------------
    print("\nYou can change the default parameters for the Ricker incident signal. Be aware that doing so may generate\
 a different result from the reference paper.")
    f0, duration, sampling_rate = RickerDefaultParameters()
    source_signal, t = ricker_source(f0, duration, sampling_rate)
    source = f'Ricker{duration*1000:.2f}ms{int(f0/1000)}kHz'
    params['sampling_rate'] = sampling_rate
    params['source'] = source

    S_f = fft(source_signal, params['fft_points'])
    freqs = fftfreq(params['fft_points'], d=1/sampling_rate)
    rayleigh_distance, field = RayleighDistance(params['a'], f0, params['c1'], params['r'])
    params['field'] = field
    freqs_positive = np.abs(freqs[:params['fft_points']//2 + 1])

    #---------------------------------------------------------------------------------------------------
    print("The final parameters are:")
    PrintSummarySourceEcho(source, duration, sampling_rate, params['fft_points'], f0)
    print(params)
    print(f"\nThe FREQUENCY STEP  to get the form function with which the echo is computed is: {freqs_positive[1] - freqs_positive[0]} Hz")


    print(f"\nChecking the echo from a {default_inner_material}-filled {default_material_sphere} spherical\
 shell in free field using a Ricker incident signal.")

    GenerateEchoesFluid(S_f, freqs_positive, sampling_rate, source, field, params, True)



def CheckEchoFluidChirp():

    print("\n...............................................................................")
    print("\nYou are now in the CheckEchoFluidChirp function.")

    print("\nThe reference paper is: Dmitrieva, M., Valdenegro-Toro, M., Brown, K., Heald, G.,\
 & Lane, D. (2017, September). Object classification with convolution neural network based\
 on the time-frequency representation of their echo. In 2017 IEEE 27th International\
 Workshop on Machine Learning for Signal Processing (MLSP) (pp. 1-6). IEEE.")

    print("\nThe paper does not specify the thickness of the shell, so we use 0.07m as inner radius.\
 They don't mention the distance between the sonar and the target, so we use 1.7m.")

    medium1 = "Water"
    default_rho1 = 1000
    default_c1 = 1500
    default_material_sphere = 'Aluminum'
    default_rho2 = 2700
    default_cd2 = 6300
    default_cs2 = 3100
    default_a = 0.075
    default_b = 0.07
    default_h = 1-default_b/default_a
    default_inner_material = 'Water'
    default_rho3 = 1000
    default_c3 = 1500

    default_r = 1.7
    default_theta_i = 0 #It's not important for this case
    target = f'{default_inner_material}-filled{default_material_sphere}SphShellReferencePaper'
    condition = 'FF'

    default_max_modes, xmin, default_xmax, fmin, default_fmax, default_freq_points =\
         DefaultFormFunctionParametersFluidFilledSphere(default_c1, default_a)

    print("\nYou can change the default parameters for the form function. Be aware that doing so may generate\
 a different result from the reference paper. In this case, changing the number of points in frequency and the\
 maximum ka will not affect the computation of the echo, since these parameters depend on the characteristics of the incident signal.")
    result = SetReferenceParameters(default_freq_points, default_max_modes, default_r, default_xmax, default_a, default_c1)
    freq_points = result[0]
    max_modes = result[1]
    xmax = result[2]
    a = result[3]
    fmax = result[4]

    default_h = 1-default_b/default_a

    params = {
        'fmin': fmin,
        'fmax': fmax,
        'frequency_points': freq_points,
        'xmin': xmin,
        'xmax': xmax,
        'n_max': max_modes,
        'c1': default_c1,
        'a': a,
        'b': default_b,
        'r': default_r,
        'rho1': default_rho1,
        'rho2': default_rho2,
        'material_sphere': default_material_sphere,
        'cd2': default_cd2,
        'cs2': default_cs2,
        'h': default_h,
        'inner_material': default_inner_material,
        'rho3': default_rho3,
        'c3': default_c3,
        'theta_i': default_theta_i,
        'medium1' : medium1,
        'fft_points' : 15000,
        'condition': condition,
        'target': target
    }

    #Source--------------------------------------------------------------------------------------------
    print("\nYou can change the default parameters for the Chirp incident signal. Be aware that doing so may generate\
 a different result from the reference paper.")
    f_start, f_end, duration, sampling_rate, Hanning, MintoMax = ChirpDefaultParametersReference()
    source_signal, t, f0 = chirp_source(f_start, f_end, duration, sampling_rate, MintoMax, Hanning)
    source = f'Chirp{duration*1000:.2f}ms{f_start/1000}to{int(f_end/1000)}kHz{Hanning}Hanning{MintoMax}MintoMax'
    params['sampling_rate'] = sampling_rate
    params['source'] = source


    S_f = fft(source_signal, params['fft_points'])
    freqs = fftfreq(params['fft_points'], d=1/sampling_rate)
    rayleigh_distance, field = RayleighDistance(params['a'], f0, params['c1'], params['r'])
    params['field'] = field
    freqs_positive = np.abs(freqs[:params['fft_points']//2 + 1])

    #---------------------------------------------------------------------------------------------------
    print("The final parameters are:")
    PrintSummarySourceEcho(source, duration, sampling_rate, params['fft_points'], f0, f_start, f_end, MintoMax, Hanning)
    print(params)
    print(f"\nThe FREQUENCY STEP  to get the form function with which the echo is computed is: {freqs_positive[1] - freqs_positive[0]} Hz")


    print(f"\nChecking the echo from a {default_inner_material}-filled {default_material_sphere} spherical\
 shell in free field using a chirp incident signal.")

    GenerateEchoesFluid(S_f, freqs_positive, sampling_rate, source, field, params, True)



#------------------------------------------------------------------------------------------
#Option 1
def ComputeFormFunctionSpherical():

    print("\n........................................FORM FUNCTIONS.......................................")

    params = GetParameters()

    while True:
        menuSphericalFormFunctions()
        option = input("Select an option: ")

        if option == "1":
            ComputeFormFunctionRigid(params)
        elif option == "2":
            ComputeFormFunctionSolid(params)
        elif option == "3":
            ComputeFormFunctionVacuum(params)
        elif option == "4":
            ComputeFormFunctionFluid(params)
        elif option == "5":
            print("Exiting Form Function generator.\n")
            break
        else:
            print("Invalid option, please try again.")



def ComputeFormFunctionRigid(params, check = False):

    print("\n...............................................................................")
    print("\nYou are now in the ComputeFormFunctionRigid function.")

    if not check: 
        fmin, fmax, frequency_points, xmin, xmax = GetFreqsFormfunction(params['fmin'], params['fmax'],\
                                                                     params['frequency_points'], params['xmin'], params['xmax'], 0)
        n_max_rigid = params['n_max_rigid']
    else: #Enter here if we are checking the form function for a rigid sphere
        fmin, fmax, frequency_points, xmin, xmax = params['fmin'], params['fmax'], params['frequency_points'], params['xmin'], params['xmax']
        n_max_rigid = params['n_max']

    c1 = params['c1']
    a = params['a']
    theta_i = params['theta_i']
    r = params['r']
    params['n_max'] = n_max_rigid

    freqs = np.linspace(fmin, fmax, frequency_points)
    k = 2 * np.pi * np.abs(freqs) / c1
    ka = k * a

    if check:
        print("Checking the form function for the rigid sphere in free field according to the reference...")
        f = np.zeros(freqs.size, dtype=complex)

        # Cálculo de los modos
        for n in tqdm(range(n_max_rigid), total=n_max_rigid, desc="Modes"):
            fn = ModesRigid(n, k, freqs, np.pi, 0, a, r, 'far-field')
            f += fn

        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        #f_farfield = ((2 * f)/(1j * ka))
        f_farfield = ((-2 * f)/(ka))#Da el mismo resultado que el anterior, pero sólo en magnitud. En fase sí cambia un poco.


        print("\nPlotting the form function for the rigid sphere in free field according to the reference...")
        PlotFormFunction(f_farfield, ka, "rigid sphere (according to the reference paper)", "free field - far field",\
             params, fmin, fmax, frequency_points, xmin, xmax)

        # print("\nPlotting the NORMALIZED form function for the rigid sphere in free field...")
        # f = f/np.max(np.abs(f))
        # PlotFormFunction(f, ka, "rigid sphere (according to the reference paper)", "free field - Normalized Form function", params, fmin, fmax, frequency_points, xmin, xmax)

    else:
        print("The FREQUENCY STEP to get the form function is: ", (fmax-fmin)/frequency_points, "Hz")
        choice = input("\nDo you want to compute the form function of the rigid sphere in free field or\
 above the seabed? (1: Free field, 2: Above the seabed): ")

        if choice == "1":

            print("Computing form function for the rigid sphere in free field...")

            f = np.zeros(freqs.size, dtype=complex)

            # Cálculo de los modos
            for n in tqdm(range(n_max_rigid), total=n_max_rigid, desc="Modes"):
                fn = ModesRigid(n, k, freqs, np.pi, 0, a, r)
                f += fn

            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

            PlotFormFunction(f, ka, "rigid sphere", "free field", params, fmin, fmax, frequency_points, xmin, xmax)
        else:
            print(f"Computing form function for the rigid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            gamma_11 = reflec_coef(theta_i, np.abs(freqs), params['rho1'], params ['rho4'], c1, params['c4'])
            gamma_11_squared = np.power(gamma_11, 2)

            print(f"Computing the monostatic scattering form function for the rigid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f1 = np.zeros(freqs.size, dtype=complex)
            for n in tqdm(range(n_max_rigid), total=n_max_rigid, desc="Modes"):
                fn = ModesRigid(n, k, freqs, np.pi, 0, a, r)
                f1 += fn

            print(f"Computing the bistatic scattering form function for the rigid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f2 = np.zeros(freqs.size, dtype=complex)
            for n in tqdm(range(n_max_rigid), total=n_max_rigid, desc="Modes"):
                fn = ModesRigid(n, k, freqs, theta_i, 1, a, r)
                f2 += fn

            formfunction = GetFormFunctionAbove(f1, f2, gamma_11, gamma_11_squared, k, a, theta_i)

            f = np.nan_to_num(formfunction, nan=0.0, posinf=0.0, neginf=0.0)

            PlotFormFunction(f, ka, "rigid sphere", "Above the seabed", params, fmin, fmax, frequency_points, xmin, xmax)



def ComputeFormFunctionSolid(params, check = False):

    print("\n...............................................................................")
    print("\nYou are now in the ComputeFormFunctionSolid function.")

    if not check:
        fmin, fmax, frequency_points, xmin, xmax = GetFreqsFormfunction(params['fmin'], params['fmax'],\
                                                                     params['frequency_points'], params['xmin'], params['xmax'], 1)
        n_max_solid = params['n_max_solid']
    else:
        fmin, fmax, frequency_points, xmin, xmax = params['fmin'], params['fmax'], params['frequency_points'],\
                                                                     params['xmin'], params['xmax']
        n_max_solid = params['n_max']

    c1 = params['c1']
    a = params['a']
    theta_i = params['theta_i']
    material_sphere = params['material_sphere']
    rho1 = params['rho1']
    rho2 = params['rho2']
    r = params['r']
    params['n_max'] = n_max_solid


    n1 = c1/params['cd2']
    n2 = c1/params['cs2']

    freqs = np.linspace(fmin, fmax, frequency_points)#La cantidad de puntos podría ser un parámetro a variar

    # Precalcular los valores que dependen solo de la frecuencia
    k = (2 * np.pi * np.abs(freqs)) / c1 #wavenumber in the liquid
    x = k * a #wave radius
    x1 = n1 * x
    x2 = n2 * x
    ka = (2 * np.pi * freqs * a) / c1


    if check:
        print(f"\nChecking the form function for the {material_sphere} solid sphere in free field according to the reference...")

        f = np.zeros(freqs.size, dtype=complex)

        for n in tqdm(range(n_max_solid), total=n_max_solid, desc="Modes"):
            fn = ModesSolid(n, k, freqs, x, x1, x2, np.pi, 0, rho1, rho2, r, 'far-field')
            f += fn

        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        f_farfield = ((2 * f)/(1j * ka))

        print(f"\nPlotting the form function for the {material_sphere} solid sphere in free field according to the reference...")
        PlotFormFunction(f_farfield, ka, "solid " + material_sphere + " sphere (according to the reference paper)", "free field - far field", params, fmin,\
                          fmax, frequency_points, xmin, xmax)

        #print(f"\nPlotting the NORMALIZED form function for the {material_sphere} solid sphere in free field...")
        #f_farfield_norm = f/np.max(np.abs(f))
        #PlotFormFunction(f_farfield_norm, ka, "solid " + material_sphere + " sphere (according to the reference paper)", "free field - far field, normalized Form function", params, fmin,\
        #                  fmax, frequency_points, xmin, xmax)

    else:
        print("The FREQUENCY STEP to get the form function is: ", (fmax-fmin)/frequency_points, "Hz")
        choice = input(f"\nDo you want to compute the form function of the {material_sphere} solid sphere\
 in free field or above the seabed? (1: Free field, 2: Above the seabed): ")
        if choice == "1":
            print(f"Computing form function for the ({material_sphere}) solid sphere in free field...")

            f = np.zeros(freqs.size, dtype=complex)

            for n in tqdm(range(n_max_solid), total=n_max_solid, desc="Modes"):
                fn = ModesSolid(n, k, freqs, x, x1, x2, np.pi, 0,rho1, rho2, r)
                f += fn

            #f = ((2 * f)/(1j * ka))
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

            PlotFormFunction(f, ka, "solid " + material_sphere + " sphere", "free field",\
                            params, fmin, fmax, frequency_points, xmin, xmax)
        else:
            print(f"Computing form function for the ({material_sphere}) solid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            gamma_11 = reflec_coef(theta_i, np.abs(freqs), rho1, params['rho4'], c1, params['c4'])
            gamma_11_squared = np.power(gamma_11, 2)

            print(f"Computing the monostatic scattering form function for the solid ({material_sphere}) elastic sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f1 = np.zeros(freqs.size, dtype=complex)
            for n in tqdm(range(n_max_solid), total=n_max_solid, desc="Modes"):
                fn = ModesSolid(n, k, freqs, x, x1, x2, np.pi, 0, rho1, rho2, r)
                f1 += fn

            print(f"Computing the bistatic scattering form function for the solid ({material_sphere}) elastic sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f2 = np.zeros(freqs.size, dtype=complex)
            for n in tqdm(range(n_max_solid), total=n_max_solid, desc="Modes"):
                fn = ModesSolid(n, k, freqs, x, x1, x2, theta_i, 1, rho1, rho2, r)
                f2 += fn

            formfunction = GetFormFunctionAbove(f1, f2, gamma_11, gamma_11_squared, k, a, theta_i)

            f = np.nan_to_num(formfunction, nan=0.0, posinf=0.0, neginf=0.0)

            PlotFormFunction(f, ka, "solid " + material_sphere + " sphere", "Above the seabed", params, fmin, fmax, frequency_points, xmin, xmax)



def ComputeFormFunctionVacuum(params, check = False):

    print("\n...............................................................................")
    print("\nYou are now in the ComputeFormFunctionVacuum function.")

    if not check:
        fmin, fmax, frequency_points, xmin, xmax = GetFreqsFormfunction(params['fmin'], params['fmax'],\
                                                                     params['frequency_points'], params['xmin'], params['xmax'], 2)
        n_max_vacuum = params['n_max_vacuum']
    else:
        fmin, fmax, frequency_points, xmin, xmax = params['fmin'], params['fmax'], params['frequency_points'], params['xmin'], params['xmax']
        n_max_vacuum = params['n_max']

    c1 = params['c1']
    a = params['a']
    material_sphere = params['material_sphere']
    theta_i = params['theta_i']
    r = params['r']
    b = params['b']
    rho_tilde = params['rho_tilde']
    params['n_max'] = n_max_vacuum


    freqs = np.linspace(fmin, fmax, frequency_points)#La cantidad de puntos podría ser un parámetro a variar

    k1 = 2 * np.pi * freqs / c1
    x = params['a'] * k1
    xs = x * c1/params['cs2']
    xL = x * c1/params['cd2']
    ys = xs * b/a
    yL = xL * b/a
    ka = (2 * np.pi * freqs * a) / c1

    if check:
        print(f"\nChecking the form function for the vacuum-filled {material_sphere} spherical shell\
 in free field according to the reference...")

        f = np.zeros(freqs.size, dtype=complex)

        for n in tqdm(range(n_max_vacuum), total=n_max_vacuum, desc="Modes"):
                fn = ModesVacuum(n, k1, freqs, x, xs, xL, ys, yL, np.pi, 0, rho_tilde, r, 'far-field')
                f += fn

        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        f = ((2 * f)/(1j * ka))

        print(f"\nPlotting the form function for the vacuum-filled {material_sphere} spherical shell in free field according to the reference...")
        PlotFormFunction(f, ka, f"Vacuum-filled {material_sphere} spherical shell (according to the reference paper)", "free field - far field", params, fmin, fmax, frequency_points, xmin, xmax)

        # print(f"\nPlotting the NORMALIZED form function for the vacuum-filled {material_sphere} spherical shell in free field...")
        # f = f/np.max(np.abs(f))
        # PlotFormFunction(f, ka, f"Vacuum-filled {material_sphere} spherical shell (according to the reference paper)", "free field - Normalized Form function", params, fmin, fmax, frequency_points, xmin, xmax)

    else:
        print("The FREQUENCY STEP to get the form function is: ", (fmax-fmin)/frequency_points, "Hz")
        choice = input(f"\nDo you want to compute the form function of the vacuum-filled {material_sphere}\
 spherical shell in free field or above the seabed? (1: Free field, 2: Above the seabed): ")
        if choice == "1":
            print(f"Computing form function for the vacuum-filled ({material_sphere}) spherical shell in free field...")

            f = np.zeros(freqs.size, dtype=complex)

            for n in tqdm(range(n_max_vacuum), total=n_max_vacuum, desc="Modes"):
                fn = ModesVacuum(n, k1, freqs, x, xs, xL, ys, yL, np.pi, 0, rho_tilde, r)
                f += fn

            #f = ((2 * f)/(1j * ka))
            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

            PlotFormFunction(f, ka, f"Vacuum-filled {material_sphere} spherical shell", "free field",\
                            params, fmin, fmax, frequency_points, xmin, xmax)
        else:
            print(f"Computing form function for the vacuum-filled {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            gamma_11 = reflec_coef(theta_i, np.abs(freqs), params['rho1'], params['rho4'], c1, params['c4'])
            gamma_11_squared = np.power(gamma_11, 2)

            print(f"Computing the monostatic scattering form function for the vacuum-filled {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f1 = np.zeros(freqs.size, dtype=complex)
            for n in tqdm(range(n_max_vacuum), total=n_max_vacuum, desc="Modes"):
                fn = ModesVacuum(n, k1, freqs, x, xs, xL, ys, yL, np.pi, 0, rho_tilde, r)
                f1 += fn

            print(f"Computing the bistatic scattering form function for the vacuum-filled {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f2 = np.zeros(freqs.size, dtype=complex)
            for n in tqdm(range(n_max_vacuum), total=n_max_vacuum, desc="Modes"):
                fn = ModesVacuum(n, k1, freqs, x, xs, xL, ys, yL, theta_i, 1, rho_tilde, r)
                f2 += fn

            formfunction = GetFormFunctionAbove(f1, f2, gamma_11, gamma_11_squared, k1, a, theta_i)

            #f = ((2 * f)/(1j * ka))
            f = np.nan_to_num(formfunction, nan=0.0, posinf=0.0, neginf=0.0)

            PlotFormFunction(f, ka, f"Vacuum-filled {material_sphere} spherical shell", "Above the seabed", params,\
                            fmin, fmax, frequency_points, xmin, xmax)



def ComputeFormFunctionFluid(params, check = False):

    print("\n...............................................................................")
    print("\nYou are now in the ComputeFormFunctionFluid function.")

    if not check:
        fmin, fmax, frequency_points, xmin, xmax = GetFreqsFormfunction(params['fmin'], params['fmax'],\
                                                                     params['frequency_points'], params['xmin'], params['xmax'], 3)
        n_max_fluid_filled = params['n_max_fluid_filled']
    else:
        fmin, fmax, frequency_points, xmin, xmax = params['fmin'], params['fmax'], params['frequency_points'], params['xmin'], params['xmax']
        n_max_fluid_filled = params['n_max']

    c1 = params['c1']
    a = params['a']
    material_sphere = params['material_sphere']
    theta_i = params['theta_i']
    r = params['r']
    inner_material = params['inner_material']
    rho1 = params['rho1']
    rho2 = params['rho2']
    rho3 = params['rho3']
    b = params['b']
    params['n_max'] = n_max_fluid_filled


    freqs = np.linspace(fmin, fmax, frequency_points)#La cantidad de puntos podría ser un parámetro a variar

    k1 = 2 * np.pi * np.abs(freqs) / c1
    k_d2 = 2 * np.pi * np.abs(freqs) / params['cd2']
    k_s2 = 2 * np.pi * np.abs(freqs) / params['cs2']
    k3 = 2 * np.pi * np.abs(freqs) / params['c3']

    ka = (2 * np.pi * freqs * a) / c1


    if check:
        print(f"\nChecking the form function for the {inner_material}-filled {material_sphere} spherical\
 shell in free field according to the reference...")

        f = np.zeros(freqs.size, dtype=complex)

        # Cálculo de los modos
        for n in tqdm(range(n_max_fluid_filled), total=n_max_fluid_filled, desc="Modes"):
            fn = ModesFluid(n, k1, k_d2, k_s2, k3, freqs, np.pi, 0, rho1, rho2, rho3, a, b, r, 'far-field')
            f += fn

        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        f = ((2 * f)/(1j * ka))

        print(f"\nPlotting the form function for the {inner_material}-filled {material_sphere} spherical shell in free field according to the reference...")
        PlotFormFunction(f, ka, f"{inner_material}-filled {material_sphere} spherical shell (according to the reference paper)",\
 "free field - far field",  params, fmin, fmax, frequency_points, xmin, xmax)

#         print(f"\nPlotting the NORMALIZED form function for the {inner_material}-filled {material_sphere} spherical shell in free field according to the reference...")
#         f = f/np.max(np.abs(f))
#         PlotFormFunction(f, ka, f"{inner_material}-filled {material_sphere} spherical shell (according to the reference paper)",\
#  "free field - Normalized Form function",  params, fmin, fmax, frequency_points, xmin, xmax)

    else:
        print("The FREQUENCY STEP to get the form function is: ", (fmax-fmin)/frequency_points, "Hz")
        choice = input(f"\nDo you want to compute the form function of the {inner_material}-filled\
 {material_sphere} spherical shell in free field or above the seabed? (1: Free field, 2: Above the seabed): ")

        if choice == "1":
            print(f"Computing form function for the {inner_material}-filled {material_sphere}\
 spherical shell in free field...")
            # Precalcular los valores que dependen solo de la frecuencia

            f = np.zeros(freqs.size, dtype=complex)

            # Cálculo de los modos
            for n in tqdm(range(n_max_fluid_filled), total=n_max_fluid_filled, desc="Modes"):
                fn = ModesFluid(n, k1, k_d2, k_s2, k3, freqs, np.pi, 0, rho1, rho2, rho3, a, b, r)
                f += fn

            f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

            PlotFormFunction(f, ka, f"{inner_material}-filled {material_sphere} spherical shell",\
    "free field",  params, fmin, fmax, frequency_points, xmin, xmax)
        else:
            print(f"Computing form function for the {inner_material}-filled {material_sphere}\
 spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")

            gamma_11 = reflec_coef(theta_i, np.abs(freqs), rho1, params['rho4'], c1, params['c4'])
            gamma_11_squared = np.power(gamma_11, 2)

            print(f"Computing the monostatic scattering form function for the {inner_material}-filled \
 {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f1 = np.zeros(freqs.size, dtype=complex)
            for n in tqdm(range(n_max_fluid_filled), total=n_max_fluid_filled, desc="Modes"):
                fn = ModesFluid(n, k1, k_d2, k_s2, k3, freqs, np.pi, 0, rho1, rho2, rho3, a, b, r)
                f1 += fn

            print(f"Computing the bistatic scattering form function for the {inner_material}-filled\
 {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f2 = np.zeros(freqs.size, dtype=complex)
            for n in tqdm(range(n_max_fluid_filled), total=n_max_fluid_filled, desc="Modes"):
                fn = ModesFluid(n, k1, k_d2, k_s2, k3, freqs, theta_i, 1, rho1, rho2, rho3, a, b, r)
                f2 += fn

            formfunction = GetFormFunctionAbove(f1, f2, gamma_11, gamma_11_squared, k1, a, theta_i)

            f = np.nan_to_num(formfunction, nan=0.0, posinf=0.0, neginf=0.0)

            PlotFormFunction(f, ka, f"{inner_material}-filled {material_sphere} spherical shell",\
    "above the seabed",  params, fmin, fmax, frequency_points, xmin, xmax)


#------------------------------------------------------------------------------------------
#Option 3
def GenerateEchoesSpherical():

    print("__________________________________\nYou are now in the GenerateEchoesSpherical function.")
    params = GetParameters()

    choice = input("\nWhat type of source do you want to use? (1: Chirp, 2: Ricker, 3: Sinusoid): ")
    if choice == '1':
        f_start, f_end, duration, sampling_rate, Hanning, MintoMax = ChirpDefaultParameters()
        source_signal, t, f0 = chirp_source(f_start, f_end, duration, sampling_rate, MintoMax, Hanning)
        source = f'Chirp{duration*1000:.2f}ms{f_start/1000}to{int(f_end/1000)}kHz{Hanning}Hanning{MintoMax}MintoMax'
        PrintSummarySourceEcho(source, duration, sampling_rate, params['fft_points'], f0, f_start, f_end, MintoMax, Hanning)
    elif choice == '2':
        f0, duration, sampling_rate = RickerDefaultParameters()
        source_signal, t = ricker_source(f0, duration, sampling_rate)
        source = f'Ricker{duration*1000:.2f}ms{int(f0/1000)}kHz'
        f_start, f_end = None, None
        PrintSummarySourceEcho(source, duration, sampling_rate, params['fft_points'], f0)
    else:
        f0, duration, sampling_rate = SinusoidDefaultParameters()
        source_signal, t = sinusoid_source(f0, duration, sampling_rate)
        source = f'Sine{duration*1000:.2f}ms{int(f0/1000)}kHz'
        f_start, f_end = None, None
        PrintSummarySourceEcho(source, duration, sampling_rate, params['fft_points'], f0)


    #1. COMPUTE THE SOURCE SPECTRUM--------------------------------------------------------------
    S_f = fft(source_signal, params['fft_points'])
    freqs = fftfreq(params['fft_points'], d=1/sampling_rate)

    # Compute the Fourier transform, with shift, positive frequencies and then filter below 200 kHz
    FT_p_source = np.fft.fftshift(S_f)
    shifted_freqs = np.fft.fftshift(freqs)
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

    #Compute Rayleigh distance
    rayleigh_distance, field = RayleighDistance(params['a'], f0, params['c1'], params['r'])
    print(f"\nApproximate expected time for the specular echo:  {(params['r']-params['a'])/params['c1']:.5f} s")
    freqs_positive = np.abs(freqs[:params['fft_points']//2 + 1])


    print("\n........................................ECHOES.......................................")

    while True:
            menuEchoesSpherical()
            option = input("Select an option: ")

            if option == "1":
                GenerateEchoesRigid(S_f, freqs_positive, sampling_rate, source, field, params)
            elif option == "2":
                GenerateEchoesSolid(S_f, freqs_positive, sampling_rate, source, field, params)
            elif option == "3":
                GenerateEchoesVacuum(S_f, freqs_positive, sampling_rate, source, field, params)
            elif option == "4":
                GenerateEchoesFluid(S_f, freqs_positive, sampling_rate, source, field, params)
            elif option == "5":
                print("Exiting Echo generator.\n")
                break
            else:
                print("Invalid option, please try again.")



def GenerateEchoesRigid(S_f, freqs_positive, sampling_rate, source, field, params, check = False):

    c1 = params['c1']
    a = params['a']
    theta_i = params['theta_i']
    r = params['r']
    fft_points = params['fft_points']
    k = 2 * np.pi * np.abs(freqs_positive) / c1
    ka = k * a

    if check:
        print("Computing the echo for the rigid sphere in free field according to the reference...")
        n_max_rigid = params['n_max']

        target = 'RigidSphereReferencePaper'
        condition = 'FF'

        f = np.zeros(freqs_positive.size, dtype=complex)
        for n in tqdm(range(n_max_rigid), total=n_max_rigid, desc="Modes"):
            #fn = ModesRigid(n, k, freqs_positive, np.pi, 0, a, r, 'far-field')
            fn = ModesRigid(n, k, freqs_positive, np.pi, 0, a, r)
            f += fn

        #f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        #f_farfield = ((-2 * f)/(ka))#Da el mismo resultado que el anterior, pero sólo en magnitud. En fase sí cambia un poco.

        PlotFormFunction(f, ka, "rigid sphere", "free field (computed to get the reference echo)", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(ka))

        f_final_rigid = GetFormFunctionEcho(f)
        echo_rigid, echo_rigid_normalized = GetEchoFF(S_f, f_final_rigid, fft_points, target)

        PlotEcho(echo_rigid_normalized, sampling_rate, source, field, params, target, condition, True)
    else:
        print(f"\nThe FREQUENCY STEP to get the form function with which the echo is computed is: {freqs_positive[1] - freqs_positive[0]} Hz")
        n_max_rigid = params['n_max_rigid']
        params['n_max'] = n_max_rigid
        choice = input("Do you want to compute the echo for the rigid sphere in free field or above the seabed?\
 (1: Free field, 2: Above the seabed): ")

        if choice == "1":
            print("Computing the echo for the rigid sphere in free field...")

            target = 'RigidSphere'
            condition = 'FF'

            f = np.zeros(freqs_positive.size, dtype=complex)
            for n in tqdm(range(n_max_rigid), total=n_max_rigid, desc="Modes"):
                fn = ModesRigid(n, k, freqs_positive, np.pi, 0, a, r)
                f += fn

            PlotFormFunction(f, ka, "rigid sphere", "free field (computed to get the echo)", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(ka))

            f_final_rigid = GetFormFunctionEcho(f)
            echo_rigid, echo_rigid_normalized = GetEchoFF(S_f, f_final_rigid, fft_points, target)

            PlotEcho(echo_rigid_normalized, sampling_rate, source, field, params, target, condition)

            filename = f"echo_{target}_{condition}_{source}_{int(sampling_rate/1000)}kSR_{a}mRad_{r:.2f}\
mSlant_{field}_{fft_points}FFT_NoTrunc.h5"
            SaveData(filename, echo_rigid_normalized, sampling_rate, source, field, params, target, condition)

        else:
            print(f"Computing the echo for the rigid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            target = 'RigidSphere'
            condition = f'Above({int(np.rad2deg(theta_i))}º)'

            gamma_11 = reflec_coef(theta_i, np.abs(freqs_positive), params['rho1'], params ['rho4'], c1, params['c4'])
            gamma_11_squared = np.power(gamma_11, 2)

            print(f"Computing the monostatic scattering form function for the rigid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f1 = np.zeros(freqs_positive.size, dtype=complex)
            for n in tqdm(range(n_max_rigid), total=n_max_rigid, desc="Modes"):
                fn = ModesRigid(n, k, freqs_positive, np.pi, 0, a, r)
                f1 += fn

            print(f"Computing the bistatic scattering form function for the rigid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f2 = np.zeros(freqs_positive.size, dtype=complex)
            for n in tqdm(range(n_max_rigid), total=n_max_rigid, desc="Modes"):
                fn = ModesRigid(n, k, freqs_positive, theta_i, 1, a, r)
                f2 += fn

            echo_rigid_above_normalized, f_above = GetEchoAbove(f1, f2, gamma_11, k, a, theta_i, gamma_11_squared, S_f, fft_points, 'rigid')

            PlotFormFunction(f_above, ka, "rigid sphere", f"Above the seabed (computed to get the echo) {int(np.rad2deg(theta_i))}º grazing angle", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(ka))

            PlotEcho(echo_rigid_above_normalized, sampling_rate, source, field, params, target, condition)

            filename = f"echo_{target}_{condition}_{source}_{int(sampling_rate/1000)}kSR_{a}mRad_{r:.2f}\
 mSlant_{field}_{fft_points}FFT_NoTrunc.h5"
            SaveData(filename, echo_rigid_above_normalized, sampling_rate, source, field, params, target, condition)



def GenerateEchoesSolid(S_f, freqs_positive, sampling_rate, source, field, params):

    n_max_solid = params['n_max_solid']
    params['n_max'] = n_max_solid
    c1 = params['c1']
    a = params['a']
    r = params['r']
    theta_i = params['theta_i']
    fft_points = params['fft_points']
    material_sphere = params['material_sphere']

    # Precalcular los valores que dependen solo de la frecuencia
    n1 = c1/params['cd2']
    n2 = c1/params['cs2']
    k = (2 * np.pi * np.abs(freqs_positive)) / c1 #wavenumber in the liquid
    x = k * a #wave radius
    x1 = n1 * x
    x2 = n2 * x

    print(f"\nThe FREQUENCY STEP to get the form function with which the echo is computed is: {freqs_positive[1] - freqs_positive[0]} Hz")
    choice = input(f"Do you want to compute the echo for the {material_sphere} solid sphere in free\
 field or above the seabed? (1: Free field, 2: Above the seabed): ")
    if choice == "1":
        print(f"Computing the echo for the {material_sphere} solid sphere in free field...")

        target = 'Sol' + f'{material_sphere}' + 'Sphere'
        condition = 'FF'

        f = np.zeros(freqs_positive.size, dtype=complex)

        for n in tqdm(range(n_max_solid), total=n_max_solid, desc="Modes"):
            fn = ModesSolid(n, k, freqs_positive, x, x1, x2, np.pi, 0, params['rho1'], params['rho2'], r)
            f += fn

        PlotFormFunction(f, x, "solid " + material_sphere + " sphere", "free field (computed to get the echo)", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(x))

        f_final_solid = GetFormFunctionEcho(f)
        echo_solid, echo_solid_normalized = GetEchoFF(S_f, f_final_solid, fft_points, target)

        PlotEcho(echo_solid_normalized, sampling_rate, source, field, params, target, condition)

        filename = f"echo_{target}_{condition}_{source}_{int(sampling_rate/1000)}kSR_{a}mRad_{r:.2f}\
mSlant_{field}_{fft_points}FFT_NoTrunc.h5"
        SaveData(filename, echo_solid_normalized, sampling_rate, source, field, params, target, condition)

    else:
        print(f"Computing the echo for the {params['material_sphere']} solid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
        target = 'Sol' + f'{params["material_sphere"]}' + 'Sphere'
        condition = f'Above({int(np.rad2deg(theta_i))}º)'

        gamma_11 = reflec_coef(theta_i, freqs_positive, params['rho1'], params['rho4'], c1, params['c4'])
        gamma_11_squared = np.power(gamma_11, 2)

        print(f"Computing the monostatic scattering form function for the {material_sphere} solid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
        f1 = np.zeros(freqs_positive.size, dtype=complex)
        for n in tqdm(range(n_max_solid), total=n_max_solid, desc="Modes"):
            fn = ModesSolid(n, k, freqs_positive, x, x1, x2, np.pi, 0, params['rho1'], params['rho2'], params['r'])
            f1 += fn

        print(f"Computing the bistatic scattering form function for the {material_sphere} solid sphere above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
        f2 = np.zeros(freqs_positive.size, dtype=complex)
        for n in tqdm(range(n_max_solid), total=n_max_solid, desc="Modes"):
            fn = ModesSolid(n, k, freqs_positive, x, x1, x2, theta_i, 1, params['rho1'], params['rho2'], params['r'])
            f2 += fn

        echo_solid_above_normalized, f_above = GetEchoAbove(f1, f2, gamma_11, k, a, theta_i, gamma_11_squared, S_f, fft_points, 'solid')

        PlotFormFunction(f_above, x, "solid " + material_sphere + " sphere", f"Above the seabed (computed to get the echo) {int(np.rad2deg(theta_i))}º grazing angle", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(x))

        PlotEcho(echo_solid_above_normalized, sampling_rate, source, field, params, target, condition)

        filename = f"echo_{target}_{condition}_{source}_{int(sampling_rate/1000)}kSR_{a}mRad_{r:.2f}\
mSlant_{field}_{fft_points}FFT_NoTrunc.h5"
        SaveData(filename, echo_solid_above_normalized, sampling_rate, source, field, params, target, condition)



def GenerateEchoesVacuum(S_f, freqs_positive, sampling_rate, source, field, params):

    n_max_vacuum = params['n_max_vacuum']
    params['n_max'] = n_max_vacuum
    c1 = params['c1']
    a = params['a']
    b = params['b']
    r = params['r']
    theta_i = params['theta_i']
    fft_points = params['fft_points']
    material_sphere = params['material_sphere']

    k1 = 2*np.pi*freqs_positive/params['c1']
    x = a * k1
    xs = x * c1/params['cs2']
    xL = x * c1/params['cd2']
    ys = xs * b/a
    yL = xL * b/a

    print(f"\nThe FREQUENCY STEP to get the form function with which the echo is computed is: {freqs_positive[1] - freqs_positive[0]} Hz")
    choice = input(f"Do you want to compute the echo for the vacuum-filled {material_sphere} spherical\
 shell in free field or above the seabed? (1: Free field, 2: Above the seabed): ")

    if choice == "1":
        print(f"Computing the echo for the vacuum-filled {material_sphere} spherical shell in free field...")

        target = 'Vac' + f'{material_sphere}' + 'SphShell'
        condition = 'FF'

        f = np.zeros(freqs_positive.size, dtype=complex)

        for n in tqdm(range(n_max_vacuum), total = n_max_vacuum, desc="Modes"):
            fn = ModesVacuum(n, k1, freqs_positive, x, xs, xL, ys, yL, np.pi, 0, params['rho_tilde'], r)
            f += fn

        PlotFormFunction(f, x, f"Vacuum-filled {material_sphere} spherical shell", "free field (computed to get the echo)", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(x))

        f_final_vacuum = GetFormFunctionEcho(f)
        echo_vacuum, echo_vacuum_normalized = GetEchoFF(S_f, f_final_vacuum, params['fft_points'], target)

        PlotEcho(echo_vacuum_normalized, sampling_rate, source, field, params, target, condition)

        filename = f"echo_{target}_{condition}_{source}_{int(sampling_rate/1000)}kSR_{a}mRad_{params['h']*100:.2f}%_{r:.2f}\
mSlant_{field}_{params['fft_points']}FFT_NoTrunc.h5"
        SaveData(filename, echo_vacuum_normalized, sampling_rate, source, field, params, target, condition)
    else:
        print(f"Computing the echo for the vacuum-filled {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")

        target = 'Vac' + f'{material_sphere}' + 'SphShell'
        condition =  f'Above({int(np.rad2deg(theta_i))}º)'

        gamma_11 = reflec_coef(theta_i, freqs_positive, params['rho1'], params['rho4'], c1, params['c4'])
        gamma_11_squared = np.power(gamma_11, 2)

        print(f"Computing the monostatic scattering form function for the vacuum {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
        f1 = np.zeros(freqs_positive.size, dtype=complex)
        for n in tqdm(range(n_max_vacuum), total=n_max_vacuum, desc="Modes"):
            fn = ModesVacuum(n, k1, freqs_positive, x, xs, xL, ys, yL, np.pi, 0, params['rho_tilde'], r)
            f1 += fn

        print(f"Computing the bistatic scattering form function for the vacuum {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
        f2 = np.zeros(freqs_positive.size, dtype=complex)
        for n in tqdm(range(n_max_vacuum), total=n_max_vacuum, desc="Modes"):
            fn = ModesVacuum(n, k1, freqs_positive, x, xs, xL, ys, yL, theta_i, 1, params['rho_tilde'], r)
            f2 += fn

        echo_vacuum_above_normalized, f_above = GetEchoAbove(f1, f2, gamma_11, k1, a, theta_i, gamma_11_squared, S_f, fft_points, 'vacuum')
        PlotFormFunction(f_above, x, f"Vacuum-filled {material_sphere} spherical shell", f"Above the seabed (computed to get the echo) {int(np.rad2deg(theta_i))}º grazing angle", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(x))

        PlotEcho(echo_vacuum_above_normalized, sampling_rate, source, field, params, target, condition)

        filename = f"echo_{target}_{condition}_{source}_{int(sampling_rate/1000)}kSR_{a}mRad_{params['h']*100:.2f}%_{r:.2f}\
 mSlant_{field}_{fft_points}FFT_NoTrunc.h5"
        SaveData(filename, echo_vacuum_above_normalized, sampling_rate, source, field, params, target, condition)



def GenerateEchoesFluid(S_f, freqs_positive, sampling_rate, source, field, params, check = False):

    c1 = params['c1']
    a = params['a']
    b = params['b']
    r = params['r']
    theta_i = params['theta_i']
    fft_points = params['fft_points']
    material_sphere = params['material_sphere']
    inner_material = params['inner_material']
    rho1 = params['rho1']
    rho2 = params['rho2']
    rho3 = params['rho3']

    # Precalcular los valores que dependen solo de la frecuencia
    k1 = 2 * np.pi * freqs_positive / c1
    ka = k1 * a
    k_d2 = 2 * np.pi * freqs_positive / params['cd2']
    k_s2 = 2 * np.pi * freqs_positive / params['cs2']
    k3 = 2 * np.pi * freqs_positive / params['c3']

    if check:
        print(f"Computing the echo for the {inner_material}-filled {material_sphere} spherical shell in free field according to the reference...")
        n_max_fluid_filled = params['n_max']

        target = f'{inner_material}-filled{material_sphere}SphShellReferencePaper'
        condition = 'FF'

        f = np.zeros(freqs_positive.size, dtype=complex)

        # Cálculo de los modos
        for n in tqdm(range(n_max_fluid_filled), total=n_max_fluid_filled, desc="Modes"):
            fn = ModesFluid(n, k1, k_d2, k_s2, k3, freqs_positive, np.pi, 0, rho1, rho2, rho3, a, b, r)
            f += fn

        PlotFormFunction(f, ka, f"{inner_material}-filled {material_sphere} spherical shell", "free field (computed to get the reference echo)", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(ka))

        f_final_fluid_filled = GetFormFunctionEcho(f)

        echo_fluid_filled, echo_fluid_filled_normalized = GetEchoFF(S_f, f_final_fluid_filled, fft_points, target)

        PlotEcho(echo_fluid_filled_normalized, sampling_rate, source, field, params, target, condition, True)

        PlotEchoinFreq(echo_fluid_filled_normalized, params, True)
    else:
        print(f"\nThe FREQUENCY STEP to get the form function with which the echo is computed is: {freqs_positive[1] - freqs_positive[0]} Hz")
        n_max_fluid_filled = params['n_max_fluid_filled']
        params['n_max'] = n_max_fluid_filled
        choice = input(f"Do you want to compute the echo for the {inner_material}-filled {material_sphere}\
 spherical shell in free field or above the seabed? (1: Free field, 2: Above the seabed): ")

        if choice == "1":

            print(f"Computing the echo for the {inner_material}-filled {material_sphere} spherical shell in free-field...")

            target = f'{inner_material}-filled{material_sphere}SphShell'
            condition = 'FF'

            f = np.zeros(freqs_positive.size, dtype=complex)

            # Cálculo de los modos
            for n in tqdm(range(n_max_fluid_filled), total=n_max_fluid_filled, desc="Modes"):
                fn = ModesFluid(n, k1, k_d2, k_s2, k3, freqs_positive, np.pi, 0, rho1, rho2, rho3, a, b, r)
                f += fn

            PlotFormFunction(f, ka, f"{inner_material}-filled {material_sphere} spherical shell", "free field (computed to get the echo)", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(ka))

            f_final_fluid_filled = GetFormFunctionEcho(f)
            echo_fluid_filled, echo_fluid_filled_normalized = GetEchoFF(S_f, f_final_fluid_filled, params['fft_points'], target)

            PlotEcho(echo_fluid_filled_normalized, sampling_rate, source, field, params, target, condition)

            #PlotEcho(echo_fluid_filled, sampling_rate, source, field, params, target, condition) #Cambia la magnitud

            filename = f"echo_{target}_{condition}_{source}_{int(sampling_rate/1000)}kSR_{a}mRad_{params['h']*100:.2f}%_{r:.2f}\
mSlant_{field}_{params['fft_points']}FFT_NoTrunc.h5"
            SaveData(filename, echo_fluid_filled_normalized, sampling_rate, source, field, params, target, condition)

        else:
            print(f"Computing the echo for the {inner_material}-filled {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")

            target = f'{inner_material}-filled{material_sphere}SphShell'
            condition = f'Above({int(np.rad2deg(theta_i))}º)'

            gamma_11 = reflec_coef(theta_i, np.abs(freqs_positive), rho1, params['rho4'], c1, params['c4'])
            gamma_11_squared = np.power(gamma_11, 2)

            print(f"Computing the monostatic scattering form function for the {inner_material}-filled {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f1 = np.zeros(freqs_positive.size, dtype=complex)
            for n in tqdm(range(n_max_fluid_filled), total=n_max_fluid_filled, desc="Modes"):
                fn = ModesFluid(n, k1, k_d2, k_s2, k3, freqs_positive, np.pi, 0, rho1, rho2, rho3, a, b, r)
                f1 += fn

            print(f"Computing the bistatic scattering form function for the {inner_material}-filled {material_sphere} spherical shell above the seabed with {int(np.rad2deg(theta_i))}º grazing angle...")
            f2 = np.zeros(freqs_positive.size, dtype=complex)
            for n in tqdm(range(n_max_fluid_filled), total=n_max_fluid_filled, desc="Modes"):
                fn = ModesFluid(n, k1, k_d2, k_s2, k3, freqs_positive, theta_i, 1, rho1, rho2, rho3, a, b, r)
                f2 += fn

            echo_fluid_filled_above_normalized, f_above = GetEchoAbove(f1, f2, gamma_11, k1, a, theta_i, gamma_11_squared, S_f, fft_points, 'fluid-filled')
            PlotFormFunction(f_above, ka, f"{inner_material}-filled {material_sphere} spherical shell", f"Above the seabed (computed to get the echo) {int(np.rad2deg(theta_i))}º grazing angle", params, 0, np.max(freqs_positive), len(freqs_positive), 0, np.max(ka))

            PlotEcho(echo_fluid_filled_above_normalized, sampling_rate, source, field, params, target, condition)

            filename = f"echo_{target}_{condition}_{source}_{int(sampling_rate/1000)}kSR_{a}mRad_{params['h']*100:.2f}%_{r:.2f}\
 mSlant_{field}_{fft_points}FFT_NoTrunc.h5"
            SaveData(filename, echo_fluid_filled_above_normalized, sampling_rate, source, field, params, target, condition)




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




def PlotEcho(echo_normalized, sampling_rate, source, field, params, target, condition, check = False):

    choice = input(f"\nDo you want to plot the echo for the {target} in {condition}? (1: Yes, 2: No): ")
    if choice == '1':
        if check:
            n_max = params['n_max']
        else:
            n_max = Get_nMax(params['n_max_rigid'], params['n_max_solid'], params['n_max_vacuum'], params['n_max_fluid_filled'], target)

        t = np.linspace(0, len(echo_normalized) / sampling_rate, len(echo_normalized))
        specular_echo  = (params['r'] - params['a'])/params['c1']
        plt.figure(figsize=(12, 6))
        info_text = (f'Source: {source}\n'
                    f'fft points: {params["fft_points"]}\n'
                    f'Sampling rate: {int(sampling_rate)} Hz\n'
                    f'Slant range (r): {params["r"]:.2f} m\n'
                    f'Thickness of the target: {params["h"]*100:.2f} % \n'
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
        plt.axvline(x=(params['r']-params['a'])*1000/params['c1'], color='red', linestyle='--', label=f'Specular echo = {specular_echo:5f} s')
        plt.legend()
        plt.show()
        plt.clf()
        plt.close()
    else:
        print(f"You chose not to plot the echo for the {target} in {condition}.")



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



def PlotEchoinFreq(echo, params, check = False):
    choice = input("\nDo you want to plot the data in the frequency domain? (1: Yes, 2: No): ")

    if choice == '1':
        if check:
            n_max = params['n_max']
        else:
            n_max = Get_nMax(params['n_max_rigid'], params['n_max_solid'], params['n_max_vacuum'], params['n_max_fluid_filled'], params['target'])

        # Compute the Fourier Transform of the echo
        echo_spectrum = fft(echo, params['fft_points'])
        freqs = fftfreq(params['fft_points'], d=1/params['sampling_rate'])

        # Shift the Fourier Transform for visualization
        FT_p_echo = np.fft.fftshift(echo_spectrum)
        shifted_freqs = np.fft.fftshift(freqs)
        # Filter only positive frequencies
        positive_freqs = shifted_freqs[shifted_freqs >= 0]
        FT_p_echo_positive = FT_p_echo[shifted_freqs >= 0]
        # Create mask to cut frequencies
        cutoff_freq = extract_upper_frequency(params['source'])
        mask = positive_freqs <= cutoff_freq
        # Apply mask
        filtered_freqs = positive_freqs[mask]
        filtered_spectrum = FT_p_echo_positive[mask]

        plt.figure(figsize=(10, 6))
        info_text = (f'Source: {params["source"]}\n'
                    f'fft points: {params["fft_points"]}\n'
                    f'Sampling rate: {int(params["sampling_rate"])} Hz\n'
                    f'Slant range (r): {params["r"]:.2f} m\n'
                    f'Thickness of the target: {params["h"]*100:.2f} % \n'
                    f'Field: {params["field"]}\n'
                    f'Surrounding fluid (Medium 1): {params["medium1"]}\n'
                    f'Number of modes used (n_max): {n_max}')
        plt.plot(filtered_freqs, np.abs(filtered_spectrum))
        plt.text(0.98, 0.02, info_text,
                            transform=plt.gca().transAxes,
                            verticalalignment='bottom',
                            horizontalalignment='right',  # Alinea el texto a la derecha
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title(f"FFT Spectrum of the echo from a {params['target']} (radius: {params['a']:.3f} m) in {params['condition']}", fontsize=18)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()
    else:
        print("No file selected. Returning to the main menu.")



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
                    f'Slant range (r): {params["r"]:.2f} m\n'
                    f'Thickness of the target: {params["h"]*100:.2f} % \n'
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


        #Form function in dB. Se supone que así lo hace Tesei----------------------------------------------------------------------------------
        f_magnitude = 20 * np.log10(np.abs(f)*params['r'])
        plt.figure(figsize=(12, 6))
        info_text = (f'fmin: {fmin:.1f} Hz\n'
            f'fmax: {fmax:.2f} Hz\n'
            f'Points in frequency: {frequency_points}\n'
            f'Slant range (r): {params["r"]:.2f} m\n'
            f'Thickness of the target: {params["h"]*100:.2f} % \n'
            f'Sound speed in water (c1): {params["c1"]} m/s\n'
            f'Surrounding fluid (Medium 1): {params["medium1"]}\n'
            f'Number of modes used (n_max): {params["n_max"]}')
        plt.plot(ka, f_magnitude)
        plt.text(0.98, 0.02, info_text,
            transform=plt.gca().transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',  # Alinea el texto a la derecha
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('ka', fontsize = 18)
        plt.ylabel('Magnitude of form function (dB)', fontsize = 18)
        if any(word in target for word in ['rigid', 'solid', 'vacuum']):
            plt.title(f'Backscattered target strength of a {target}\n (radius: {params["a"]:.3f} m) in {condition}',\
                       fontsize=18)
        else:
            plt.title(f'Backscattered target strength of a {target}\n ((radius: {params["a"]} m, thickness (h) =\
 {params["h"] * 100:.1f} %) in {condition}', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.xlim(xmin, xmax)  # Establece el límite del eje x de 0 a 30
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
            f'Slant range (r): {params["r"]:.2f} m\n'
            f'Thickness of the target: {params["h"]*100:.2f} % \n'
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




#Getting---------------------------------------------------------------------------------------
def GetParameters():

    #0. PARAMETERS OF THE MEDIA-------------------------------------------------------------------------
    default_medium1, default_rho1, default_c1, default_material_sphere, default_rho2, default_cd2, \
        default_cs2, default_a, default_b, default_h, default_inner_material, default_rho3, default_c3, \
            default_sediment, default_rho4, default_c4, default_delta = DefaultParametersMedia()

    default_rho_tilde = default_rho1/default_rho2


    #1. Parameters of the sonar and target--------------------------------------
    default_sonar_position, default_target_position, default_r, default_theta_i, default_fft_points = DefaultParametersSonarTarget()


    #2. Parameters of the form functions
    default_xmin_rigid, default_xmax_rigid, default_fmin_rigid, default_fmax_rigid, default_frequency_points_rigid,\
              default_xmin_solid, default_xmax_solid, default_fmin_solid, default_fmax_solid, default_frequency_points_solid,\
                default_xmin_vacuum, default_xmax_vacuum, default_fmin_vacuum, default_fmax_vacuum, default_frequency_points_vacuum,\
                    default_xmin_fluid_filled, default_xmax_fluid_filled, default_fmin_fluid_filled, default_fmax_fluid_filled,\
                        default_frequency_points_fluid_filled, default_n_max_fluid_filled, default_n_max_rigid, default_n_max_solid,\
                              default_n_max_vacuum = GetDefaultFormFuctionParameters(default_c1, default_a)


    default_xmin = [default_xmin_rigid, default_xmin_solid, default_xmin_vacuum, default_xmin_fluid_filled]
    default_xmax = [default_xmax_rigid, default_xmax_solid, default_xmax_vacuum, default_xmax_fluid_filled]
    default_fmin = [default_fmin_rigid, default_fmin_solid, default_fmin_vacuum, default_fmin_fluid_filled]
    default_fmax = [default_fmax_rigid, default_fmax_solid, default_fmax_vacuum, default_fmax_fluid_filled]
    default_frequency_points = [default_frequency_points_rigid, default_frequency_points_solid,\
                         default_frequency_points_vacuum, default_frequency_points_fluid_filled]


    printSimulationParameters(default_sonar_position,\
                                    default_target_position, default_r, default_medium1,\
                                    default_material_sphere, default_inner_material, default_rho1,\
                                    default_c1, default_rho2, default_cd2, default_cs2, default_a,\
                                    default_b, default_h, default_rho3, default_c3, default_sediment,\
                                    default_rho4, default_c4, default_delta,\
                                    default_n_max_fluid_filled, default_n_max_rigid, default_n_max_solid,\
                                    default_n_max_vacuum, default_theta_i, default_fft_points,\
                                    default_xmin, default_xmax, default_fmin, default_fmax, default_frequency_points)

    print("\n...............................................................................")
    choice= input("\nDo you want to change the default parameters of the ENTIRE simulation (echoes\
 and form functions)? (1: Yes, 2: No): ")
    if choice == '1':
        #Sonar position
        sonar_position = SetSonarPosition(default_sonar_position)

        #Target position
        target_position = SetTargetPosition(default_target_position)

        #El grazing angle y la distancia r se calculan a partir de la posición del sonar y del target
        #Podría preguntar antes si quiere definir el ángulo de incidencia y la distancia r en lugar de
        # las posiciones del sonar y del target
        theta_i, r = inc_graz_angle(sonar_position[0], sonar_position[1], sonar_position[2], target_position)

        #Target size
        a, b = SetTargetSize(default_a, default_b)
        h = 1 -b/a

        #FFT points
        fft_points = SetFFTPoints(default_fft_points)

        medium1, rho1, c1, material_sphere, rho2, cd2, cs2, inner_material, rho3, c3, sediment, rho4, c4, delta, rho_tilde =\
              SetMediaParameters(default_medium1, default_rho1, default_c1, default_material_sphere, default_rho2, default_cd2,\
                           default_cs2, default_inner_material, default_rho3, default_c3, default_sediment, default_rho4,\
                            default_c4, default_delta)


        n_max_fluid_filled, n_max_rigid, n_max_solid, n_max_vacuum, xmin_rigid, xmax_rigid, fmin_rigid, fmax_rigid,\
            frequency_points_rigid, xmin_solid, xmax_solid, fmin_solid, fmax_solid, frequency_points_solid,\
                xmin_vacuum, xmax_vacuum, fmin_vacuum, fmax_vacuum, frequency_points_vacuum, xmin_fluid_filled,\
                    xmax_fluid_filled, fmin_fluid_filled, fmax_fluid_filled, frequency_points_fluid_filled =\
                         SetFormFunctionParameters(default_n_max_fluid_filled, default_n_max_rigid, default_n_max_solid, default_n_max_vacuum,\
                              c1, a, default_xmin_rigid, default_xmax_rigid, default_fmin_rigid, default_fmax_rigid,\
                            default_frequency_points_rigid, default_xmin_solid, default_xmax_solid, default_fmin_solid,\
                            default_fmax_solid, default_frequency_points_solid, default_xmin_vacuum, default_xmax_vacuum,\
                            default_fmin_vacuum, default_fmax_vacuum, default_frequency_points_vacuum, default_xmin_fluid_filled,\
                            default_xmax_fluid_filled, default_fmin_fluid_filled, default_fmax_fluid_filled, default_frequency_points_fluid_filled)


        xmin = [xmin_rigid, xmin_solid, xmin_vacuum, xmin_fluid_filled]
        xmax = [xmax_rigid, xmax_solid, xmax_vacuum, xmax_fluid_filled]
        fmin = [fmin_rigid, fmin_solid, fmin_vacuum, fmin_fluid_filled]
        fmax = [fmax_rigid, fmax_solid, fmax_vacuum, fmax_fluid_filled]
        frequency_points = [frequency_points_rigid, frequency_points_solid, frequency_points_vacuum, frequency_points_fluid_filled]

        print("\n............................UPDATED PARAMETERS........................................")
        printSimulationParameters(sonar_position, target_position, r, medium1,\
                                        material_sphere, inner_material, rho1,\
                                        c1, rho2, cd2, cs2, a,\
                                        b, h, rho3, c3, sediment,\
                                        rho4, c4, delta,\
                                        n_max_fluid_filled, n_max_rigid, n_max_solid,\
                                        n_max_vacuum, theta_i, fft_points,\
                                        xmin, xmax, fmin, fmax, frequency_points)

    else:
        print("You chose not to change the default parameters of the simulation.")
        sonar_position = default_sonar_position
        target_position = default_target_position
        theta_i, r = inc_graz_angle(sonar_position[0], sonar_position[1], sonar_position[2], target_position)
        medium1 = default_medium1
        rho1 = default_rho1
        c1 = default_c1
        material_sphere = default_material_sphere
        rho2 = default_rho2
        cd2 = default_cd2
        cs2 = default_cs2
        rho_tilde = default_rho_tilde
        inner_material = default_inner_material
        rho3 = default_rho3
        c3 = default_c3
        sediment = default_sediment
        rho4 = default_rho4
        c4 = default_c4
        delta = default_delta
        a = default_a
        b = default_b
        h = default_h
        n_max_fluid_filled = default_n_max_fluid_filled
        n_max_rigid = default_n_max_rigid
        n_max_solid = default_n_max_solid
        n_max_vacuum = default_n_max_vacuum
        fft_points = default_fft_points
        xmin = [default_xmin_rigid, default_xmin_solid, default_xmin_vacuum, default_xmin_fluid_filled]
        xmax = [default_xmax_rigid, default_xmax_solid, default_xmax_vacuum, default_xmax_fluid_filled]
        fmin = [default_fmin_rigid, default_fmin_solid, default_fmin_vacuum, default_fmin_fluid_filled]
        fmax = [default_fmax_rigid, default_fmax_solid, default_fmax_vacuum, default_fmax_fluid_filled]
        frequency_points = [default_frequency_points_rigid, default_frequency_points_solid, default_frequency_points_vacuum,\
                             default_frequency_points_fluid_filled]

    return {
        'sonar_position': sonar_position,
        'target_position': target_position,
        'r': r,
        'medium1': medium1,
        'material_sphere': material_sphere,
        'inner_material': inner_material,
        'rho1': rho1,
        'c1': c1,
        'rho2': rho2,
        'cd2': cd2,
        'cs2': cs2,
        'a': a,
        'b': b,
        'h': h,
        'rho3': rho3,
        'c3': c3,
        'sediment': sediment,#-----
        'rho4': rho4,#-----
        'c4': c4,#-----
        'delta': delta,#-----
        'rho_tilde': rho_tilde,
        'n_max_fluid_filled': n_max_fluid_filled,
        'n_max_rigid': n_max_rigid,
        'n_max_solid': n_max_solid,
        'n_max_vacuum': n_max_vacuum,
        'theta_i': theta_i,#-----
        'fft_points': fft_points,
        'xmin': xmin,
        'xmax': xmax,
        'fmin': fmin,
        'fmax': fmax,
        'frequency_points': frequency_points#---Para la form function
    }



def GetDefaultFormFuctionParameters(default_c1, default_a):

    #3. Parameters for Form function of rigid sphere
    default_n_max_rigid, default_xmin_rigid, default_xmax_rigid, default_fmin_rigid, default_fmax_rigid, \
        default_frequency_points_rigid = DefaultFormFunctionParametersRigidSphere(default_c1, default_a)


    #4. Parameters for Form function of the solid sphere
    default_n_max_solid, default_xmin_solid, default_xmax_solid, default_fmin_solid, default_fmax_solid, \
        default_frequency_points_solid = DefaultFormFunctionParametersSolidSphere(default_c1, default_a)


    #5. Parameters for Form function of the vacuum-filled elastic spherical shell
    default_n_max_vacuum, default_xmin_vacuum, default_xmax_vacuum, default_fmin_vacuum, default_fmax_vacuum,\
          default_frequency_points_vacuum = DefaultFormFunctionParametersVacuumFilledSphere(default_c1, default_a)


    #6. Form function of the fluid-filled elastic spherical shell
    default_n_max_fluid_filled, default_xmin_fluid_filled, default_xmax_fluid_filled, default_fmin_fluid_filled,\
          default_fmax_fluid_filled, default_frequency_points_fluid_filled = DefaultFormFunctionParametersFluidFilledSphere(default_c1, default_a)

    return default_xmin_rigid, default_xmax_rigid, default_fmin_rigid, default_fmax_rigid, default_frequency_points_rigid,\
              default_xmin_solid, default_xmax_solid, default_fmin_solid, default_fmax_solid, default_frequency_points_solid,\
                default_xmin_vacuum, default_xmax_vacuum, default_fmin_vacuum, default_fmax_vacuum, default_frequency_points_vacuum,\
                    default_xmin_fluid_filled, default_xmax_fluid_filled, default_fmin_fluid_filled, default_fmax_fluid_filled,\
                        default_frequency_points_fluid_filled, default_n_max_fluid_filled, default_n_max_rigid, default_n_max_solid, default_n_max_vacuum



def DefaultParametersMedia():
    default_medium1 = "Water"
    default_rho1 = 998.2 # 1000.0
    default_c1 = 1486 # 1500

    #Medium 2: exterior of a typical target----------------------------------------------
    default_material_sphere = "Steel"
    default_rho2 = 7700
    default_cd2 = 5950
    default_cs2 = 3240
    default_a = 0.125 #0.25
    default_b = 0.12 #0.235 #Estaba 0.093
    h = 1 - default_b/default_a

    #Medium 3: inner of the target----------------------------------------------
    #Water
    default_inner_material = "Water"
    default_rho3 = 1000.0
    default_c3 = 1500

    #Medium 4: Sediment----------------------------------------------
    default_sediment = "Sand"
    default_rho4 = 2000
    default_c4 = 1694 # 1779
    #Dimensionless loss parameters
    default_delta = 0.008 #0.01

    return default_medium1, default_rho1, default_c1, default_material_sphere, default_rho2, default_cd2,\
          default_cs2, default_a, default_b, h, default_inner_material, default_rho3, default_c3,\
              default_sediment, default_rho4, default_c4, default_delta




def DefaultParametersSonarTarget():
    #To replicate PONDEX, taking information from the cylinder case (paper 2010), for the case of the rock.
    default_cross_distance = -9.55 #m (distance in the X direction between the sonar and the target)
    default_y_sonar = 0 #Sonar starts at -1 m in the Y direction
    default_height = 3.6 #m (height of the sonar)
    default_sonar_position = np.array([default_cross_distance, default_y_sonar, default_height])

    # #To get a slant range of 20 m
    # default_cross_distance = 12 #m (distance in the X direction between the sonar and the target)
    # default_y_sonar = 12 #Sonar starts at -1 m in the Y direction
    # default_height = 12 #m (height of the sonar)
    # default_sonar_position = np.array([default_cross_distance, default_y_sonar, default_height])

    x_t = 0 #X location of the target
    y_t = 0 #Y location of the target
    z_t = 0 #Z location of the target
    default_target_position = np.array([x_t,y_t,z_t])
    default_fft_points = 15000 #Estaba en 4096
    default_theta_i, default_r = inc_graz_angle(default_cross_distance, default_y_sonar, default_height,\
                                                 default_target_position)

    return default_sonar_position, default_target_position, default_r, default_theta_i, default_fft_points



def DefaultFormFunctionParametersRigidSphere(default_c1, default_a):
    default_n_max_rigid = 80
    default_xmin_rigid = 0
    default_xmax_rigid = 14
    default_fmin_rigid = default_xmin_rigid * default_c1 / (2 * np.pi * default_a)
    default_fmax_rigid = default_xmax_rigid * default_c1 / (2 * np.pi * default_a)
    default_frequency_points_rigid = 4096

    return default_n_max_rigid, default_xmin_rigid, default_xmax_rigid, default_fmin_rigid, default_fmax_rigid, default_frequency_points_rigid



def DefaultFormFunctionParametersSolidSphere(default_c1, default_a):
    default_n_max_solid = 80
    default_xmin_solid = 0
    default_xmax_solid = 30
    default_fmin_solid = default_xmin_solid * default_c1 / (2 * np.pi * default_a)
    default_fmax_solid = default_xmax_solid * default_c1 / (2 * np.pi * default_a)
    default_frequency_points_solid = 4096

    return default_n_max_solid, default_xmin_solid, default_xmax_solid, default_fmin_solid, default_fmax_solid, default_frequency_points_solid



def DefaultFormFunctionParametersVacuumFilledSphere(default_c1, default_a):
    default_n_max_vacuum = 80
    default_xmin_vacuum = 0
    default_xmax_vacuum = 20
    default_fmin_vacuum = default_xmin_vacuum * default_c1 / (2 * np.pi * default_a)
    default_fmax_vacuum = default_xmax_vacuum * default_c1 / (2 * np.pi * default_a)
    default_frequency_points_vacuum = 4096

    return default_n_max_vacuum, default_xmin_vacuum, default_xmax_vacuum, default_fmin_vacuum, default_fmax_vacuum,\
          default_frequency_points_vacuum



def DefaultFormFunctionParametersFluidFilledSphere(default_c1, default_a):
    default_n_max_fluid_filled = 120
    default_xmin_fluid_filled = 0
    default_xmax_fluid_filled = 100
    default_fmin_fluid_filled = default_xmin_fluid_filled * default_c1 / (2 * np.pi * default_a)
    default_fmax_fluid_filled = default_xmax_fluid_filled * default_c1 / (2 * np.pi * default_a)
    default_frequency_points_fluid_filled = 4096

    return default_n_max_fluid_filled, default_xmin_fluid_filled, default_xmax_fluid_filled, default_fmin_fluid_filled,\
          default_fmax_fluid_filled, default_frequency_points_fluid_filled



def printSimulationParameters(
        sonar_position, target_position, r, medium1, material_sphere, inner_material,
          rho1, c1, rho2, cd2, cs2, a, b, h, rho3, c3, sediment, rho4, c4, delta,
            n_max_fluid_filled, n_max_rigid, n_max_solid, n_max_vacuum, theta_i, fft_points, xmin, xmax,
              fmin, fmax, frequency_points):

    print("\n........................................CURRENT PARAMETERS OF THE SIMULATION.......................................")
    print("\nGENERAL SETTINGS:")
    print("- Position of the sonar: ", [sonar_position[0], sonar_position[1], sonar_position[2]])
    print("- Position of the target: ", target_position)
    print("- Target type:", 'sphere')
    print(f"- Distance between the sonar and the target: {r:.2f} m")
    print("- Medium 1 (surrounding water): ", medium1)
    print("- Medium 2 (external material of the target): ", material_sphere)
    print("- Medium 3 (inner fluid): ", inner_material)
    print("- Medium 4: ", sediment)
    print(f"- Incident grazing angle relative to the sediment surface (for the proud case): {np.degrees(theta_i):.2f}º")
    print("- fftpoints: ", fft_points)


    #-----------------------------------------------------------------------------------
    print("\nFORM FUNCTIONS of the spheres (except for the 'maximum number of modes', the other parameters\
 are used only for the computation of the form functions (not echo)): ")
    print("\nRigid sphere: ")
    print("- Maximum number of modes to be computed for the form function of the rigid sphere: ", n_max_rigid)
    print(f"- ka min (xmin): {xmin[0]:.2f}")
    print(f"- ka max (xmax): {xmax[0]:.2f}")
    print(f"- Minimum frequency (fmin): {fmin[0]:.2f} Hz")
    print(f"- Maximum frequency (fmax): {fmax[0]:.2f} Hz")
    print(f"- Number of frequency points: {frequency_points[0]}")

    print("\nSolid sphere: ")
    print("- Maximum number of modes to be computed for the form function of the solid sphere: ", n_max_solid)
    print(f"- ka min (xmin): {xmin[1]:.2f}")
    print(f"- ka max (xmax): {xmax[1]:.2f}")
    print(f"- Minimum frequency (fmin): {fmin[1]:.2f} Hz")
    print(f"- Maximum frequency (fmax): {fmax[1]:.2f} Hz")
    print(f"- Number of frequency points: {frequency_points[1]}")

    print("\nVacuum-filled sphere: ")
    print("- Maximum number of modes to be computed for the form function of the vacuum-filled sphere: ", n_max_vacuum)
    print(f"- ka min (xmin): {xmin[2]:.2f}")
    print(f"- ka max (xmax): {xmax[2]:.2f}")
    print(f"- Minimum frequency (fmin): {fmin[2]:.2f} Hz")
    print(f"- Maximum frequency (fmax): {fmax[2]:.2f} Hz")
    print(f"- Number of frequency points: {frequency_points[2]}")

    print("\nFluid-filled sphere: ")
    print("- Maximum number of modes to be computed for the form function of the fluid-filled sphere: ", n_max_fluid_filled)
    print(f"- ka min (xmin): {xmin[3]:.2f}")
    print(f"- ka max (xmax): {xmax[3]:.2f}")
    print(f"- Minimum frequency (fmin): {fmin[3]:.2f} Hz")
    print(f"- Maximum frequency (fmax): {fmax[3]:.2f} Hz")
    print(f"- Number of frequency points: {frequency_points[3]}")


    #-----------------------------------------------------------------------------------
    print("\nPROPERTIES OF THE MEDIA: ")
    print("\nMEDIUM 1 (surrounding fluid) properties: ")
    print(f"- Density of the surrounding fluid ({medium1}): {rho1} kg/m^3")
    print(f"- Speed of sound in the surrounding fluid ({medium1}): {c1} m/s")

    #-----------------------------------------------------------------------------------
    print("\nMEDIUM 2 (external material of the target) properties: ")
    print("- Outer radius of the spherical shape (a): ", a, "m")
    print("- Inner radius of the fluid-filled elastic spherical shell (b): ", b, "m")
    print(f"- Thickness of the fluid-filled elastic spherical shell (1 - a/b):  {h*100:.2f} %")
    print(f"- Material of the spherical shape: {material_sphere}")
    print(f"- Density of the material of the spherical shape ({material_sphere}): {rho2} kg/m^3")
    print(f"- Speed of compressional waves in the material of the spherical shape ({material_sphere}): {cd2} m/s")
    print(f"- Speed of shear waves in the material of the spherical shape ({material_sphere}): {cs2} m/s")

    #-----------------------------------------------------------------------------------
    print("\nMEDIUM 3(inner fluid) properties: ")
    print(f"- Inner material of the fluid-filled elastic spherical shell: {inner_material}")
    print(f"- Density of the inner fluid ({inner_material}): {rho3} kg/m^3")
    print(f"- Speed of sound in the inner fluid ({inner_material}): {c3} m/s")


    #-----------------------------------------------------------------------------------
    print("\nMEDIUM 4 (sediment) properties: ")
    print(f"- Sediment type: {sediment}")
    print(f"- Density of the sediment ({sediment}): {rho4} kg/m^3")
    print(f"- Speed of sound in the sediment ({sediment}): {c4} m/s")
    print(f"- Dimensionless loss parameter: {delta}")
    #------------------------------------------------------------


#Setting---------------------------------------------------------------------------------------
def SetSonarPosition(default_sonar_position):
    user_input = input(f"\nDo you want to change the sonar position (this probably will change the slant range\
 'r' and the grazing angle 'theta_i')? (Current position: X={default_sonar_position[0]}, Y={default_sonar_position[1]},\
 Z={default_sonar_position[2]})\nPress Enter to keep default, or type 'y' to change: ")
    if user_input.lower().strip() == 'y':
        while True:
            try:
                coords = input("Enter x y z coordinates separated by spaces: ").split()
                if len(coords) != 3:
                    raise ValueError("Need exactly 3 coordinates")
                #cross_distance, y_sonar, height = map(float, coords)
                sonar_position = np.array([float(x) for x in coords])
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")
    else:
        sonar_position = default_sonar_position

    return sonar_position



def SetTargetPosition(default_target_position):
    user_input = input(f"\nDo you want to change the target position (this probably will change the slant range\
 'r' and the grazing angle 'theta_i')? (Current position = {default_target_position})\nPress Enter to keep default,\
 or type 'y' to change: ")
    if user_input.lower().strip() == 'y':
        while True:
            try:
                coords = input("Enter x y z coordinates separated by spaces: ").split()
                if len(coords) != 3:
                    raise ValueError("Need exactly 3 coordinates")
                target_position = np.array([float(x) for x in coords])
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")
    else:
        target_position = default_target_position

    return target_position



def SetMediaParameters(default_medium1, default_rho1, default_c1, default_material_sphere, default_rho2, default_cd2,\
                           default_cs2, default_inner_material, default_rho3, default_c3, default_sediment, default_rho4,\
                            default_c4, default_delta):
    #Medium1
    medium1, rho1, c1 = SetMedium1(default_medium1, default_rho1, default_c1)


    #Medium2
    material_sphere, rho2, cd2, cs2 = SetMedium2(default_material_sphere, default_rho2, default_cd2, default_cs2)
    rho_tilde = rho1/rho2


    #Medium3
    inner_material, rho3, c3 = SetMedium3(default_inner_material, default_rho3, default_c3)


    #Medium4: sedimento
    sediment, rho4, c4, delta = SetMedium4(default_sediment, default_rho4, default_c4, default_delta)

    return medium1, rho1, c1, material_sphere, rho2, cd2, cs2, inner_material, rho3, c3, sediment, rho4, c4, delta, rho_tilde



def SetFormFunctionParameters(default_n_max_fluid_filled, default_n_max_rigid, default_n_max_solid, default_n_max_vacuum,\
                              c1, a, default_xmin_rigid, default_xmax_rigid, default_fmin_rigid, default_fmax_rigid,\
                            default_frequency_points_rigid, default_xmin_solid, default_xmax_solid, default_fmin_solid,\
                            default_fmax_solid, default_frequency_points_solid, default_xmin_vacuum, default_xmax_vacuum,\
                            default_fmin_vacuum, default_fmax_vacuum, default_frequency_points_vacuum, default_xmin_fluid_filled,\
                            default_xmax_fluid_filled, default_fmin_fluid_filled, default_fmax_fluid_filled, default_frequency_points_fluid_filled):
    #Number of modes
    n_max_fluid_filled, n_max_rigid, n_max_solid, n_max_vacuum = SetMaxModes(default_n_max_fluid_filled,\
                                                                                  default_n_max_rigid, default_n_max_solid,\
                                                                                  default_n_max_vacuum)

    #Parameters of the form function for the rigid sphere
    xmin_rigid, xmax_rigid, fmin_rigid, fmax_rigid, frequency_points_rigid = SetFormFunctionParametersRigid(\
        c1, a, default_xmin_rigid, default_xmax_rigid, default_fmin_rigid, default_fmax_rigid, default_frequency_points_rigid)


    #Parameters of the form function for the solid sphere
    xmin_solid, xmax_solid, fmin_solid, fmax_solid, frequency_points_solid = SetFormFunctionParametersSolid(\
        c1, a, default_xmin_solid, default_xmax_solid, default_fmin_solid, default_fmax_solid, default_frequency_points_solid)


    #Parameters of the form function for the vacuum-filled elastic spherical shell
    xmin_vacuum, xmax_vacuum, fmin_vacuum, fmax_vacuum, frequency_points_vacuum = SetFormFunctionParametersVacuum(\
        c1, a, default_xmin_vacuum, default_xmax_vacuum, default_fmin_vacuum, default_fmax_vacuum, default_frequency_points_vacuum)


    #Parameters of the form function for the fluid-filled elastic spherical shell
    xmin_fluid_filled, xmax_fluid_filled, fmin_fluid_filled, fmax_fluid_filled, frequency_points_fluid_filled = SetFormFunctionParametersFluid(\
        c1, a, default_xmin_fluid_filled, default_xmax_fluid_filled, default_fmin_fluid_filled, default_fmax_fluid_filled, default_frequency_points_fluid_filled)


    return n_max_fluid_filled, n_max_rigid, n_max_solid, n_max_vacuum, xmin_rigid, xmax_rigid, fmin_rigid, fmax_rigid,\
            frequency_points_rigid, xmin_solid, xmax_solid, fmin_solid, fmax_solid, frequency_points_solid,\
                xmin_vacuum, xmax_vacuum, fmin_vacuum, fmax_vacuum, frequency_points_vacuum, xmin_fluid_filled,\
                    xmax_fluid_filled, fmin_fluid_filled, fmax_fluid_filled, frequency_points_fluid_filled



def SetMedium1(default_medium1, default_rho1, default_c1):
    user_input = input(f"\nDo you want to change the medium 1 and its properties? (Current:\
 {default_medium1})\nPress Enter to keep default, or type 'y' to change: ")

    if user_input.lower().strip() == 'y':
        medium1 = input("Enter medium name: ").strip()
        while True:
            try:
                rho1 = float(input("Enter medium density (kg/m³): "))
                c1 = float(input("Enter sound speed in medium (m/s) (this also affects the computation\
 of the form functions): "))
                break
            except ValueError:
                print("Invalid input. Please enter numeric values.")
    else:
        medium1 = default_medium1
        rho1 = default_rho1
        c1 = default_c1

    return medium1, rho1, c1



def SetMedium2(default_material_sphere, default_rho2, default_cd2, default_cs2):
    user_input = input(f"\nDo you want to change the material of the sphere and its properties?\
 (Current: {default_material_sphere})\nPress Enter to keep default, or type 'y' to change: ")

    if user_input.lower().strip() == 'y':
        material_sphere = input("Enter sphere material name: ").strip()
        while True:
            try:
                rho2 = float(input("Enter material density (kg/m³): "))
                cd2 = float(input("Enter compressional wave speed (this also affects the computation\
 of the form functions) (m/s): "))
                cs2 = float(input("Enter shear wave speed (this also affects the computation of the\
 form functions) (m/s): "))
                break
            except ValueError:
                print("Invalid input. Please enter numeric values.")
    else:
        material_sphere = default_material_sphere
        rho2 = default_rho2
        cd2 = default_cd2
        cs2 = default_cs2

    return material_sphere, rho2, cd2, cs2



def SetMedium3(default_inner_material, default_rho3, default_c3):
    user_input = input(f"\nDo you want to change the inner material and its properties? (Current:\
 {default_inner_material})\nPress Enter to keep default, or type 'y' to change: ")

    if user_input.lower().strip() == 'y':
        inner_material = input("Enter inner material name: ").strip()
        while True:
            try:
                rho3 = float(input("Enter material density (kg/m³): "))
                c3 = float(input("Enter sound speed (this also affects the computation of the form\
 functions) (m/s): "))
                break
            except ValueError:
                print("Invalid input. Please enter numeric values.")
    else:
        inner_material = default_inner_material
        rho3 = default_rho3
        c3 = default_c3

    return inner_material, rho3, c3



def SetMedium4(default_sediment, default_rho4, default_c4, default_delta):
    user_input = input(f"\nDo you want to change the sediment and its properties \nPress Enter to\
 keep default, or type 'y' to change: ")
    if user_input.lower().strip() == 'y':
                sediment = input("Enter sediment name: ").strip()
                while True:
                    try:
                        rho4 = float(input("Enter material density (kg/m³): "))
                        c4 = float(input("Enter sound speed (this also affects the computation of\
 the form functions) (m/s): "))
                        delta = float(input("Enter dimensionless loss parameter: "))
                        break
                    except ValueError:
                        print("Invalid input. Please enter numeric values.")
    else:
        sediment = default_sediment
        rho4 = default_rho4
        c4 = default_c4
        delta = default_delta

    return sediment, rho4, c4, delta



def SetTargetSize(default_a, default_b):
    user_input = input(f"\nDo you want to change the sphere radii? (Current: outer={default_a}m,\
 inner={default_b}m)\nPress Enter to keep default, or type 'y' to change: ")

    if user_input.lower().strip() == 'y':
        while True:
            try:
                a = float(input("Enter outer radius (this also affects the computation of the form\
 functions) (m): "))
                b = float(input("Enter inner radius (this also affects the computation of the form\
 functions) (m): "))
                if b >= a:
                    raise ValueError("Inner radius must be smaller than outer radius")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")
    else:
        a = default_a
        b = default_b

    return a, b



def SetMaxModes(default_n_max_fluid_filled, default_n_max_rigid, default_n_max_solid, default_n_max_vacuum):
    user_input = input(f"\nDo you want to change maximum modes to compute the form functions? (Current:\
 fluid_filled={default_n_max_fluid_filled}, rigid={default_n_max_rigid}, solid={default_n_max_solid},\
 vacuum={default_n_max_vacuum})\nPress Enter to keep default, or type 'y' to change: ")

    if user_input.lower().strip() == 'y':
        while True:
            try:
                n_max_fluid_filled = int(input("Enter max modes for fluid filled: "))
                n_max_rigid = int(input("Enter max modes for rigid: "))
                n_max_solid = int(input("Enter max modes for solid: "))
                n_max_vacuum = int(input("Enter max modes for vacuum: "))
                if any(n <= 0 for n in [n_max_fluid_filled, n_max_rigid, n_max_solid, n_max_vacuum]):
                    raise ValueError("All values must be positive")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter positive integers.")
    else:
        n_max_fluid_filled = default_n_max_fluid_filled
        n_max_rigid = default_n_max_rigid
        n_max_solid = default_n_max_solid
        n_max_vacuum = default_n_max_vacuum

    return n_max_fluid_filled, n_max_rigid, n_max_solid, n_max_vacuum



def SetFFTPoints(default_fft_points):
    user_input = input(f"\nDo you want to change the number of FFT points? (Current: {default_fft_points})\
 \nPress Enter to keep default, or type 'y' to change: ")
    if user_input.lower().strip() == 'y':
        while True:
            try:
                fft_points = int(input("Enter number of FFT points: "))
                if fft_points <= 0:
                    raise ValueError("Value must be positive")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter a positive integer.")
    else:
        fft_points = default_fft_points

    return fft_points



def SetFormFunctionParametersRigid(c1, a, default_xmin_rigid, default_xmax_rigid, default_fmin_rigid,\
 default_fmax_rigid, default_frequency_points_rigid):
    user_input = input(f"\nDo you want to change the parameters of the form function for the rigid\
 sphere\nPress Enter to keep default, or type 'y' to change: ")
    if user_input.lower().strip() == 'y':
        while True:
            try:
                xmin_rigid = float(input("Enter the minimum value of ka: "))
                xmax_rigid = float(input("Enter the maximum value of ka: "))
                fmin_rigid = xmin_rigid * c1 / (2 * np.pi * a)
                fmax_rigid = xmax_rigid * c1 / (2 * np.pi * a)
                frequency_points_rigid = int(input("Enter the number of frequency points: "))
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter numeric values.")
    else:
        xmin_rigid = default_xmin_rigid
        xmax_rigid = default_xmax_rigid
        fmin_rigid = default_fmin_rigid
        fmax_rigid = default_fmax_rigid
        frequency_points_rigid = default_frequency_points_rigid

    return xmin_rigid, xmax_rigid, fmin_rigid, fmax_rigid, frequency_points_rigid



def SetFormFunctionParametersSolid(c1, a, default_xmin_solid, default_xmax_solid, default_fmin_solid,\
                                    default_fmax_solid, default_frequency_points_solid):
    user_input = input(f"\nDo you want to change the parameters of the form function for the solid\
 sphere\nPress Enter to keep default, or type 'y' to change: ")
    if user_input.lower().strip() == 'y':
        while True:
            try:
                xmin_solid = float(input("Enter the minimum value of ka: "))
                xmax_solid = float(input("Enter the maximum value of ka: "))
                fmin_solid = xmin_solid * c1 / (2 * np.pi * a)
                fmax_solid = xmax_solid * c1 / (2 * np.pi * a)
                frequency_points_solid = int(input("Enter the number of frequency points: "))
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter numeric values.")
    else:
        xmin_solid = default_xmin_solid
        xmax_solid = default_xmax_solid
        fmin_solid = default_fmin_solid
        fmax_solid = default_fmax_solid
        frequency_points_solid = default_frequency_points_solid

    return xmin_solid, xmax_solid, fmin_solid, fmax_solid, frequency_points_solid



def SetFormFunctionParametersVacuum(c1, a, default_xmin_vacuum, default_xmax_vacuum, default_fmin_vacuum,\
                                    default_fmax_vacuum, default_frequency_points_vacuum):
    user_input = input(f"\nDo you want to change the parameters of the form function for the vacuum-filled\
 elastic spherical shell\nPress Enter to keep default, or type 'y' to change: ")
    if user_input.lower().strip() == 'y':
        while True:
            try:
                xmin_vacuum = float(input("Enter the minimum value of ka: "))
                xmax_vacuum = float(input("Enter the maximum value of ka: "))
                fmin_vacuum = xmin_vacuum * c1 / (2 * np.pi * a)
                fmax_vacuum = xmax_vacuum * c1 / (2 * np.pi * a)
                frequency_points_vacuum = int(input("Enter the number of frequency points: "))
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter numeric values.")
    else:
        xmin_vacuum = default_xmin_vacuum
        xmax_vacuum = default_xmax_vacuum
        fmin_vacuum = default_fmin_vacuum
        fmax_vacuum = default_fmax_vacuum
        frequency_points_vacuum = default_frequency_points_vacuum

    return xmin_vacuum, xmax_vacuum, fmin_vacuum, fmax_vacuum, frequency_points_vacuum



def SetFormFunctionParametersFluid(c1, a, default_xmin_fluid_filled, default_xmax_fluid_filled, default_fmin_fluid_filled,\
                                    default_fmax_fluid_filled, default_frequency_points_fluid_filled):
    user_input = input(f"\nDo you want to change the parameters of the form function for the fluid-filled\
 elastic spherical shell\nPress Enter to keep default, or type 'y' to change: ")
    if user_input.lower().strip() == 'y':
        while True:
            try:
                xmin_fluid_filled = float(input("Enter the minimum value of ka: "))
                xmax_fluid_filled = float(input("Enter the maximum value of ka: "))
                fmin_fluid_filled = xmin_fluid_filled * c1 / (2 * np.pi * a)
                fmax_fluid_filled = xmax_fluid_filled * c1 / (2 * np.pi * a)
                frequency_points_fluid_filled = int(input("Enter the number of frequency points: "))
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter numeric values.")
    else:
        xmin_fluid_filled = default_xmin_fluid_filled
        xmax_fluid_filled = default_xmax_fluid_filled
        fmin_fluid_filled = default_fmin_fluid_filled
        fmax_fluid_filled = default_fmax_fluid_filled
        frequency_points_fluid_filled = default_frequency_points_fluid_filled

    return xmin_fluid_filled, xmax_fluid_filled, fmin_fluid_filled, fmax_fluid_filled, frequency_points_fluid_filled



#Math functions--------------------------------------------------------------------------------
def sph_jl_deriv2(l, x):
    bessel_j_deriv = jvp(l + 0.5, x, n=2)  # Segunda derivada de la función de Bessel
    return np.sqrt(np.pi / (2 * x)) * bessel_j_deriv



def sph_yl_deriv2(l, x):
    bessel_y_deriv = yvp(l + 0.5, x, n=2)  # Segunda derivada de la función de Bessel
    return np.sqrt(np.pi / (2 * x)) * bessel_y_deriv



def hankel1_spherical(n, x):
    return (spherical_jn(n, x) + (1j * spherical_yn(n, x)))



def hankel2_spherical(n, x):
    return (spherical_jn(n, x) - (1j * spherical_yn(n, x)))



def hankel1_sph_deriv(n,x):
    return (spherical_jn(n, x, True) + (1j * spherical_yn(n, x, True)))


#Sources---------------------------------------------------------------------------------------
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


def ChirpDefaultParametersReference():
    default_f_start = 30000.0
    user_input = input(f"\nEnter the start frequency of the chirp (in Hz - Press Enter to use default value = {default_f_start}): ")
    f_start = float(user_input) if user_input.strip() else default_f_start

    default_f_end = 160000.0
    f_end = input(f"Enter the end frequency of the chirp (in Hz - Press Enter to use default value = {default_f_end}): ")
    f_end = float(f_end) if f_end.strip() else default_f_end

    default_duration = 0.0015
    duration = input(f"Enter the duration of the chirp (in seconds - Press Enter to use default value = {default_duration}): ")
    duration = float(duration) if duration.strip() else default_duration

    default_sampling_rate = 1000000.0
    sampling_rate = input(f"Enter the sampling rate (in Hz - Press Enter to use default value = {default_sampling_rate}): ")
    sampling_rate = float(sampling_rate) if sampling_rate.strip() else default_sampling_rate

    choice = input("\nDo you want to apply a Hanning window to the incident chirp signal? (1: Yes - used in the reference, 2: No): ")
    if choice == '1':
        Hanning = True
    else:
        Hanning = False

    choice = input("Do you want the chirp from Min to Max frequency or from Max to Min frequency? (1: Min to Max, 2: Max to Min -\
 used in the reference): ")
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




def inc_graz_angle(cross_distance, y_sonar, height, target_position):
    #We define the position vector of the Sonar
    x_s = cross_distance #X location of the Sonar
    y_s = y_sonar #Y location of the Sonar
    z_s = height #Z location of the Sonar

    sonar_position = np.array([x_s,y_s,z_s])
    sonar_proj = np.array([x_s, y_s, 0]) #Projection of sonar_position on the XY plane

    #We define the vectors needed to calculate the theta angle
    r = sonar_position - target_position
    norm_r = np.linalg.norm(r)

    r_proy = sonar_proj - target_position
    norm_r_proy = np.linalg.norm(r_proy)

    #Here we are going to calculate the angle theta_i that we need for equation (29)
    theta_i = np.arccos(norm_r_proy/norm_r)

    return theta_i, norm_r



def reflec_coef(theta_i, freqs, rho1, rho2, c1, c2):

    rho = rho2/rho1
    #kappa = (1+1j*delta2)/v #Creo que no puedo calcularlo así porque no estaría considerando las diferentes frecuencias. v = c2/c1
    k2 = 2*np.pi*freqs/c2 #sediment or target
    k1 = 2*np.pi*freqs/c1 #water

    kappa = k2/k1

    # Using np.lib.scimath.sqrt to handle complex numbers automatically
    sqrt_term = np.lib.scimath.sqrt(np.power(kappa, 2) - np.power(np.cos(theta_i), 2))
    gamma = (rho * np.sin(theta_i) - sqrt_term) / (rho * np.sin(theta_i) + sqrt_term)

    return gamma



def RayleighDistance(a, f0, c1, r):
#Compute Rayleigh distance
    print('----------------------------------')
    rayleigh_distance = (a**2) * np.pi * f0/ c1
    print(f"Slant range between monostatic sonar and center of the target: {r:.3f} m")
    print(f"Slant range between monostatic sonar and surface of the target: {r-a:.3f} m")
    print(f"Rayleigh distance: {rayleigh_distance:.3f} m")
    print('Central frequency of the source: ', f0)

    if float(r) >= float(rayleigh_distance):
        print("Hydrophones in the far field of the target")
        field = 'FarField'
    else:
        print("Hydrophones in the near field of the target")
        field = 'NearField'
    print('----------------------------------')

    return rayleigh_distance, field



def GetFormFunctionEcho(f):

    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f_conj = np.conjugate(f[1:])
    f_conj_inv = f_conj[::-1]
    f_final = np.concatenate([f[:-1], f_conj_inv])

    return f_final



def GetFormFunctionAbove(f1, f2, gamma_11, gamma_11_squared, k, a, theta_i):
    term1 = f1
    term2 = 2 * gamma_11 * f2 * np.exp(2j * k * a * (np.sin(theta_i)))
    term3 = gamma_11_squared * f1 * np.exp(4j * k * a * np.sin(theta_i))
    formfunction = (term1 + term2 + term3) #* np.exp(-1j * k * r)  # Equation 30 from paper 2012

    return formfunction



def GetEchoFF(S_f, f_final, fft_points, target):

    if 'Rigid' in target:
        echo_spectrum = S_f * f_final#[::-1] #Eco al derecho
    else:
        echo_spectrum = S_f * f_final[::-1] #Eco al derecho

    echo = ifft(echo_spectrum, fft_points)
    echo_normalized = echo / np.max(np.abs(echo))

    # Normalización por raíz de la potencia
    #echo_power = np.mean(np.abs(echo)**2)
    #echo_normalized = echo / np.sqrt(echo_power)

    return echo, echo_normalized



def GetEchoAbove(f1, f2, gamma_11, k, a, theta_i, gamma_11_squared, S_f, fft_points, flag):
    term1 = f1
    term2 = 2 * gamma_11 * f2 * np.exp(2j * k * a * (np.sin(theta_i)))
    term3 = gamma_11_squared * f1 * np.exp(4j * k * a * np.sin(theta_i))
    formfunction = term1 + term2 + term3  # Equation 30 from paper 2012
    #formfunction = f * np.exp(-0.75j * k * r)

    f_above = np.nan_to_num(formfunction, nan=0.0, posinf=0.0, neginf=0.0)
    f_conj_above = np.conjugate(f_above[1:])  # Elimina el primer elemento de f, que es la frecuencia cero,
    f_conj_inv_above = f_conj_above[::-1]
    f_final_above = np.concatenate([f_above[:-1], f_conj_inv_above])

    if flag == 'rigid':
        echo_spectrum_above = S_f * f_final_above #[::-1] #Eco al derecho
    else:
        echo_spectrum_above = S_f * f_final_above[::-1] #Eco al derecho

    echo_above = ifft(echo_spectrum_above, fft_points) #Para tratar de replicar el de Tesei

    echo_above_normalized = echo_above / np.max(np.abs(echo_above))
    # Normalización por raíz de la potencia
    #echo_power = np.mean(np.abs(echo_above)**2)
    #echo_above_normalized = echo_above / np.sqrt(echo_power)

    return echo_above_normalized, f_above



def Get_nMax(n_max_rigid, n_max_solid, n_max_vacuum, n_max_fluid_filled, target):
    if 'Rigid' in target:
        n_max = n_max_rigid
    elif 'Sol' in target:
        n_max = n_max_solid
    elif 'Vac' in target:
        n_max = n_max_vacuum
    elif 'filled' in target:
        n_max = n_max_fluid_filled
    else:
        # En caso de que no coincida con ninguna subcadena conocida
        raise ValueError(f"Unknown case type: {target}")

    return n_max



def GetFreqsFormfunction(fmin, fmax, frequency_points, xmin, xmax, indice):
    fmin = fmin[indice]
    fmax = fmax[indice]
    frequency_points = frequency_points[indice]
    xmin = xmin[indice]
    xmax = xmax[indice]

    return fmin, fmax, frequency_points, xmin, xmax



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




#Modes----------------------------------------------------------------------------------------
def ModesRigid(n, k1, freqs, theta, flag, a, r, field = 'near-field'):
    fn_array = np.zeros(freqs.size, dtype=complex)

    for i in range(len(freqs)):

        j_n = spherical_jn(n, k1[i]*a, True)
        y_n = spherical_yn(n, k1[i]*a, True)
        #print(j_n, y_n)

        eta_n = np.arctan(-j_n/y_n)

        if flag == 0: #backscattering
            if field == 'near-field':
                fn_array[i] = ((-1j)**(n+1)) * hankel2_spherical(n, k1[i] * r) * eval_legendre(n, np.cos(theta))\
                    * (2 * n + 1) * np.sin(eta_n) * np.exp(1j * eta_n)
            else:
                fn_array[i] = eval_legendre(n, np.cos(theta)) * (2 * n + 1) * np.sin(eta_n) * np.exp(1j * eta_n)
        else:
            fn_array[i] = ((-1j)**(n+1)) * hankel2_spherical(n, k1[i] * r) * eval_legendre(n, -np.cos(2 * theta))\
                  * (2 * n + 1) * np.sin(eta_n) * np.exp(1j * eta_n)

    return fn_array



def ModesSolid(n, k1, freqs, x, x1, x2, theta, flag, rho1, rho2, r, field = 'near-field'):

    fn_array = np.zeros(freqs.size, dtype=complex)

    for i in range(len(freqs)):
        d11 = (rho1/rho2) * (x2[i]**2) * hankel1_spherical(n,x[i]) #original
        d12 = ((2*n*(n+1) - (x2[i]**2)) * spherical_jn(n,x1[i])) - (4*x1[i]*spherical_jn(n,x1[i],True)) #original
        d13 = 2*n*(n+1)*((x2[i] * spherical_jn(n,x2[i],True)) - spherical_jn(n,x2[i])) #original
        d21 = -1 * x[i] * hankel1_sph_deriv(n,x[i]) #original
        d22 = x1[i] * spherical_jn(n,x1[i],True) #original
        d23 = n * (n+1) * spherical_jn(n,x2[i]) #original
        d32 = 2 * (spherical_jn(n,x1[i]) - (x1[i] * spherical_jn(n,x1[i],True))) #original
        d33 = 2 * x2[i] * spherical_jn(n,x2[i],True) + (((x2[i]**2)- (2 * n * (n+1)) + 2) * spherical_jn(n,x2[i])) #original del paper
        d10 = -1 * (rho1/rho2) * (x2[i]**2) * spherical_jn(n,x[i]) #original
        d20 = x[i] * spherical_jn(n,x[i],True) #original


        B_matrix = np.array([
        [d10, d12, d13],
        [d20, d22, d23],
        [0.0, d32, d33]
        ])


        D_matrix = np.array([
        [d11, d12, d13],
        [d21, d22, d23],
        [0.0, d32, d33]
        ])

        B_determinant = np.linalg.det(B_matrix)
        D_determinant = np.linalg.det(D_matrix)
        R_determinant = -B_determinant / D_determinant

        if flag == 0:
            if field == 'near-field':
                fn_array[i] = (1j**n) * (2*n + 1) * R_determinant * hankel1_spherical(n, k1[i] * r) * eval_legendre(n, np.cos(theta))
            else:
                fn_array[i] = ((-1)**n) * R_determinant * (2 * n + 1)

        else:
            fn_array[i] = (1j**n) * (2*n + 1) * R_determinant * hankel1_spherical(n, k1[i] * r) * eval_legendre(n, -np.cos(2 * theta))

    return fn_array



def ModesVacuum(n, k1, freqs, x, xs, xL, ys, yL, theta, flag, rho_tilde, r, field = "near-field"):
    fn_array = np.zeros(freqs.size, dtype=complex)

    for i in range(len(freqs)):

        jn_x_freq = spherical_jn(n, x[i])
        jn_x_s_freq = spherical_jn(n, xs[i])
        jn_x_L_freq = spherical_jn(n, xL[i])
        yn_x_L_freq = spherical_yn(n, xL[i])
        yn_x_s_freq = spherical_yn(n, xs[i])
        yn_y_L_freq = spherical_yn(n, yL[i])
        yn_y_s_freq = spherical_yn(n, ys[i])
        jn_y_L_freq = spherical_jn(n, yL[i])
        jn_y_s_freq = spherical_jn(n, ys[i])

        jn_x_L_freq_deriv = spherical_jn(n, xL[i], True)
        jn_x_s_freq_deriv = spherical_jn(n, xs[i], True)
        yn_x_L_freq_deriv = spherical_yn(n, xL[i], True)
        yn_x_s_freq_deriv = spherical_yn(n, xs[i], True)
        jn_y_L_freq_deriv = spherical_jn(n, yL[i], True)
        jn_y_s_freq_deriv = spherical_jn(n, ys[i], True)
        yn_y_L_freq_deriv = spherical_yn(n, yL[i], True)
        yn_y_s_freq_deriv = spherical_yn(n, ys[i], True)

        h1_x_freq = hankel1_spherical(n, x[i])
        h1_x_freq_deriv = hankel1_sph_deriv(n, x[i])

        d11 = rho_tilde * (xs[i]**2) * h1_x_freq
        d12 = ((2 * n * (n + 1) - (xs[i] ** 2)) * jn_x_L_freq) - (4 * xL[i] * jn_x_L_freq_deriv)
        d13 = 2 * n * (n + 1) * (xs[i] * jn_x_s_freq_deriv - jn_x_s_freq)
        d14 = (2 * n * (n + 1) - (xs[i] ** 2)) * yn_x_L_freq - (4 * xL[i] * yn_x_L_freq_deriv)
        d15 = 2 * n * (n + 1) * (xs[i] * yn_x_s_freq_deriv - yn_x_s_freq)
        d21 = -x[i] * h1_x_freq_deriv
        d22 = xL[i] * jn_x_L_freq_deriv
        d23 = n * (n + 1) * jn_x_s_freq
        d24 = xL[i] * yn_x_L_freq_deriv
        d25 = n * (n + 1) * yn_x_s_freq
        d32 = 2 * (jn_x_L_freq  - (xL[i] * jn_x_L_freq_deriv))
        d33 = (2 * xs[i] * jn_x_s_freq_deriv) + (((xs[i] ** 2) - 2 * n * (n + 1) + 2) * jn_x_s_freq)
        d34 = 2 * (yn_x_L_freq - (xL[i] * yn_x_L_freq_deriv))
        d35 = (2 * xs[i] * yn_x_s_freq_deriv) + (((xs[i] ** 2) - 2 * n * (n + 1) + 2) * yn_x_s_freq)
        d42 = ((2 * n * (n + 1) - (ys[i] ** 2)) * jn_y_L_freq) - (4 * yL[i] * jn_y_L_freq_deriv)
        d43 = 2 * n * (n + 1) * (ys[i] * jn_y_s_freq_deriv - jn_y_s_freq)
        d44 = ((2 * n * (n + 1) - ys[i] ** 2) * yn_y_L_freq) - (4 * yL[i] * yn_y_L_freq_deriv)
        d45 = 2 * n * (n + 1) * (ys[i] * yn_y_s_freq_deriv - yn_y_s_freq)
        d52 = 2 * (jn_y_L_freq - yL[i] * jn_y_L_freq_deriv)
        d53 = (2 * ys[i] * jn_y_s_freq_deriv) + ((ys[i] ** 2) - 2 * n * (n + 1) + 2) * jn_y_s_freq
        d54 = 2 * (yn_y_L_freq - yL[i] * yn_y_L_freq_deriv)
        d55 = (2 * ys[i] * yn_y_s_freq_deriv) + ((ys[i] ** 2) - 2 * n * (n + 1) + 2) * yn_y_s_freq
        b11 = -rho_tilde * (xs[i] ** 2) * jn_x_freq
        b21 = x[i] * spherical_jn(n, x[i], True)


        B_matrix = np.array([
        [b11, d12, d13, d14, d15],
        [b21, d22, d23, d24, d25],
        [0, d32, d33, d34, d35],
        [0, d42, d43, d44, d45],
        [0, d52, d53, d54, d55]
        ])

        D_matrix = np.array([
        [d11, d12, d13, d14, d15],
        [d21, d22, d23, d24, d25],
        [0, d32, d33, d34, d35],
        [0, d42, d43, d44, d45],
        [0, d52, d53, d54, d55]
        ])

        B_determinant = np.linalg.det(B_matrix)
        D_determinant = np.linalg.det(D_matrix)
        R_determinant = -B_determinant / D_determinant

        if flag == 0:
            if field == 'near-field':
                fn_array[i] = (1j**n) * (2*n + 1) * R_determinant * hankel1_spherical(n, k1[i] * r) * eval_legendre(n, np.cos(theta))
            else:
                fn_array[i] = ((-1)**n) * (2 * n + 1) * R_determinant
        else:
            fn_array[i] = (1j**n) * (2*n + 1) * R_determinant * hankel1_spherical(n, k1[i] * r) * eval_legendre(n, -np.cos(2 * theta))
    return fn_array



def ModesFluid(n, k1, k_d2, k_s2, k3, freqs, theta, flag, rho1, rho2, rho3, a, b, r, field = 'near-field'):

    fn_array = np.zeros(freqs.size, dtype=complex)

    for i in range(len(freqs)):
        d11 = (rho1 / rho2) * k_s2[i] ** 2 * a ** 2 * hankel1_spherical(n, k1[i] * a)
        d12 = (2 * n * (n + 1) - k_s2[i] ** 2 * a ** 2) * spherical_jn(n, k_d2[i] * a) - 4 * k_d2[i] * a * spherical_jn(n, k_d2[i] * a, True)
        d13 = (2 * n * (n + 1) - k_s2[i] ** 2 * a ** 2) * spherical_yn(n, k_d2[i] * a) - 4 * k_d2[i] * a * spherical_yn(n, k_d2[i] * a, True)
        d14 = 2 * n * (n + 1) * (k_s2[i] * a * spherical_jn(n, k_s2[i] * a, True) - spherical_jn(n, k_s2[i] * a))
        d15 = 2 * n * (n + 1) * (k_s2[i] * a * spherical_yn(n, k_s2[i] * a, True) - spherical_yn(n, k_s2[i] * a))

        d21 = -k1[i] * a * hankel1_sph_deriv(n, k1[i] * a)  # original

        d22 = k_d2[i] * a * spherical_jn(n, k_d2[i] * a, True)
        d23 = k_d2[i] * a * spherical_yn(n, k_d2[i] * a, True)
        d24 = n * (n + 1) * spherical_jn(n, k_s2[i] * a)
        d25 = n * (n + 1) * spherical_yn(n, k_s2[i] * a)

        d32 = 2 * (spherical_jn(n, k_d2[i] * a) - k_d2[i] * a * spherical_jn(n, k_d2[i] * a, True))
        d33 = 2 * (spherical_yn(n, k_d2[i] * a) - k_d2[i] * a * spherical_yn(n, k_d2[i] * a, True))
        d34 = 2 * k_s2[i] * a * spherical_jn(n, k_s2[i] * a, True) + (k_s2[i] ** 2 * a ** 2 - 2 * n * (n + 1) + 2) * spherical_jn(n, k_s2[i] * a)
        d35 = 2 * k_s2[i] * a * spherical_yn(n, k_s2[i] * a, True) + (k_s2[i] ** 2 * a ** 2 - 2 * n * (n + 1) + 2) * spherical_yn(n, k_s2[i] * a)

        d42 = -4 * k_d2[i] * b * spherical_jn(n, k_d2[i] * b, True) + (2 * n * (n + 1) - k_s2[i] ** 2 * b ** 2) * spherical_jn(n, k_d2[i] * b)
        d43 = -4 * k_d2[i] * b * spherical_yn(n, k_d2[i] * b, True) + (2 * n * (n + 1) - k_s2[i] ** 2 * b ** 2) * spherical_yn(n, k_d2[i] * b)

        d44 = 2 * n * (n + 1) * (k_s2[i] * b * spherical_jn(n, k_s2[i] * b, True) - spherical_jn(n, k_s2[i] * b))
        d45 = 2 * n * (n + 1) * (k_s2[i] * b * spherical_yn(n, k_s2[i] * b, True) - spherical_yn(n, k_s2[i] * b))

        d46 = (rho3 / rho2) * k_s2[i] ** 2 * b ** 2 * spherical_jn(n, k3[i] * b)
        d52 = k_d2[i] * b * spherical_jn(n, k_d2[i] * b, True)
        d53 = k_d2[i] * b * spherical_yn(n, k_d2[i] * b, True)
        d54 = n * (n + 1) * spherical_jn(n, k_s2[i] * b)
        d55 = n * (n + 1) * spherical_yn(n, k_s2[i] * b)
        d56 = -k3[i] * b * spherical_jn(n, k3[i] * b, True)
        d62 = 2 * (spherical_jn(n, k_d2[i] * b) - k_d2[i] * b * spherical_jn(n, k_d2[i] * b, True))
        d63 = 2 * (spherical_yn(n, k_d2[i] * b) - k_d2[i] * b * spherical_yn(n, k_d2[i] * b, True))
        d64 = 2 * k_s2[i] * b * spherical_jn(n, k_s2[i] * b, True) + (k_s2[i] ** 2 * b ** 2 - 2 * n * (n + 1) + 2) * spherical_jn(n, k_s2[i] * b)
        d65 = 2 * k_s2[i] * b * spherical_yn(n, k_s2[i] * b, True) + (k_s2[i] ** 2 * b ** 2 - 2 * n * (n + 1) + 2) * spherical_yn(n, k_s2[i] * b)

        A1ast = -(rho1 / rho2) * k_s2[i] ** 2 * a ** 2 * spherical_jn(n, k1[i] * a)
        A2ast = k1[i] * a * spherical_jn(n, k1[i] * a, True)

        B_matrix = np.array([
            [A1ast, d12, d13, d14, d15, 0],
            [A2ast, d22, d23, d24, d25, 0],
            [0, d32, d33, d34, d35, 0],
            [0, d42, d43, d44, d45, d46],
            [0, d52, d53, d54, d55, d56],
            [0, d62, d63, d64, d65, 0]
        ])

        D_matrix = np.array([
            [d11, d12, d13, d14, d15, 0],
            [d21, d22, d23, d24, d25, 0],
            [0, d32, d33, d34, d35, 0],
            [0, d42, d43, d44, d45, d46],
            [0, d52, d53, d54, d55, d56],
            [0, d62, d63, d64, d65, 0]
        ])

        B_determinant = np.linalg.det(B_matrix)
        D_determinant = np.linalg.det(D_matrix)
        R_determinant = -B_determinant / D_determinant

        if flag == 0:
            if field == 'near-field':
                fn_array[i] = (1j**n) * (2*n + 1) * R_determinant * hankel1_spherical(n, k1[i] * r) * eval_legendre(n, np.cos(theta))
            else:
                fn_array[i] = ((-1) ** n) * (2 * n + 1) * R_determinant
        else:
            fn_array[i] = (1j**n) * (2*n + 1) * R_determinant * hankel1_spherical(n, k1[i] * r) * eval_legendre(n, -np.cos(2 * theta))
    return fn_array



# Bucle principal
while True:

    MainMenu()
    option = input("\nSelect an option: ")

    if option == "0":
        CheckSimulator()
    elif option == "1":
        ComputeFormFunctionSpherical()
    elif option == "3":
        GenerateEchoesSpherical()
    elif option == "15":
        print("Exiting Acoustic Scattering Simulator V1.0. ¡Hasta luego!\n")
        break  # Salir del bucle
    else:
        print("Invalid option, please try again.")


#-----------------------------------------------------------------------------------
