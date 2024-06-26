#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:57:51 2024

@author: benbroughton
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import constants
from scipy.interpolate import CubicSpline
import scipy.integrate as it
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib.ticker import AutoMinorLocator, FixedLocator

st.set_page_config(layout="wide", page_title="bdb5 Airfoil Design")
st.title("The Thin Line Between Flying and Falling")
st.subheader("The Inverse Design of Wings Using Thin Airfoil")

if 'Slot_gamma' not in st.session_state:
    st.session_state.Slot_gamma = [None, None]

with st.sidebar:
    input_form = st.selectbox(
    "How would you like to define your pressure distribution?",
    ("Manually", "CSV"),
    index=0,
    placeholder="Select input method...",
    )

    allow_negative = st.toggle('Allow Negative Values for Pressure Distribution')
    with st.expander("Airfoil Properties", True):

        thickness = st.slider('Please select airfoil thickness', 0.01, 0.2)

        if not allow_negative:
            sliderlim = 0.0
            plotlim = -0.1
        else:
            sliderlim = -2.0
            plotlim = -0.5
        if input_form == "Manually":
            A = st.slider('Please select point A value', sliderlim, 1.5, 0.0)
            B = st.slider('Please select point B value', sliderlim, 1.5, 0.0)
            C = st.slider('Please select point C value', sliderlim, 1.5, 0.0)
            D = st.slider('Please select point D value', sliderlim, 1.5, 0.0)
            E = st.slider('Please select point E value', sliderlim, 1.5, 0.0)
            F = st.slider('Please select point F value', sliderlim, 1.5, 0.0)


            x_val = [0, 0.2, 0.4, 0.6, 0.8, 1]
            y_val= [A, B, C, D, E, F]
            f = CubicSpline(x_val, y_val, bc_type='natural')
            x = np.linspace(0, 1.01, 101)
            gamma_dist = f(x)
        if input_form == "CSV":
            x_val = []
            y_val = []
            input_csv = st.text_input('Insert values here. Ensure centered on x axis')
            coords = input_csv.split()
            for i in coords:
                x_str, y_str = i.split(",")
                x_val.append(float(x_str))
                y_val.append(float(y_str))

    allow_flaps = st.toggle('Add Flap to Airfoil')
    if allow_flaps:
        with st.expander("Flap Properties", True):
            flap_length = st.slider('Please select your flap length', 0.01, 0.25, 0.01)
            flap_degrees = st.slider('Please select your flap angle', -10.0, 10.0, 0.0, 0.1)
            flap_angle = flap_degrees*np.pi/180

    fix_cruise = st.toggle('Scale Load Distribution for Cruise')
    if fix_cruise:
        gamma_dist_original = gamma_dist
        fix_Cl = st.slider('Please select your target Cl for cruise', 0.4, 0.6, 0.5, 0.01)
        S = fix_Cl/(2*np.trapz(gamma_dist_original,dx=0.01))
        st.write('Recalculating gamma using scale factor ',S)
        gamma_dist = S*gamma_dist

    


if not allow_negative:
    replace_negatives = np.vectorize(lambda x: 0 if x < 0 else x)
    gamma_dist = replace_negatives(gamma_dist)

col1A, col2A = st.columns([4, 5])
if not fix_cruise:
    st.header("Important Plots")
else:
    st.header("Important Performance Metrics - all scaled for $C_{l}$")
col1B, col2B, col3B = st.columns(3)
with col1A:
    fig = plt.figure(0)
    if not fix_cruise:
        plt.plot(x, gamma_dist, 'b')
        plt.plot(x_val, y_val, 'ro')
    else:
        plt.plot(x, gamma_dist_original, 'b', label = 'Original')
        plt.plot(x_val, y_val, 'ro')
        plt.plot(x, gamma_dist, 'g', label = 'Scaled')
        plt.legend()
    if not allow_negative:
        plt.ylim(-0.1,2.1)
    else:
        plt.ylim(-2.1,2.1)
    plt.title('Cubic Spline Interpolation of Load Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    st.pyplot(fig)
    st.write(np.round(min(np.gradient(gamma_dist)),5))


######### STEALING CODE FROM PREVIOUS THING ##########

c = 1
V = 1
a0 = 0.2969
a1 = -0.126
a2 = -0.3516
a3 = 0.2843
a4 = -0.1015
a4_closed = -0.1036 

dummy = np.linspace(0, 1.01, 101)

# Definition of Thin Airfoil from NACA Standard
def f(x):
    return (thickness/0.2)*(a0*x**0.5 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)
def e(x):
    return 2*f(x)
def e_prime(x):
    return 2*(thickness/0.2)*(0.5*a0*x**-0.5 + a1 + 2*a2*x + 3*a3*x**2 + 4*a4*x**3)

# Symmetric and Lifting Problem Integrals - (3.29), (3.49)
def symmetric_integral(dummy,x):
    return (1/(2*np.pi))*e_prime(dummy)/(x-dummy)
def lifting_integral(dummy,x,gamma):
    return (-1/(2*np.pi))*gamma/(x-dummy)


############################## SYMMETRIC PROBLEM ##############################

# Local Two Point Gaussian Quadrature Step for Symmetric Problem
def sym_two_point_gaussian_quadrature_step(a,b,x):
    var_a = (a-b)/(2*np.sqrt(3)) + (b+a)/2
    var_b = (b-a)/(2*np.sqrt(3)) + (b+a)/2
    return ((b-a)/2)*(symmetric_integral(var_a,x)+symmetric_integral(var_b,x))

# Full Numerical Integration of Symmetric Integral (3.29)
def sym_two_point_gaussian_quadrature(U = 1):
    store = []
    for i in range(len(x)):
        local_store = []
        for j in range(len(dummy)-1):    
            local_store.append(sym_two_point_gaussian_quadrature_step(dummy[j],dummy[j+1],x[i]))
        store.append(np.sum(local_store)+U)                                         # Include the uniform flow                                                      # Manually fix the velocity at trailing edge to zero
    return store

############################### LIFTING PROBLEM ###############################

# Local Two Point Gaussian Quadrature Step for Lifting Problem
def lift_two_point_gaussian_quadrature_step(a,b,x,gamma):
    var_a = (a-b)/(2*np.sqrt(3)) + (b+a)/2
    var_b = (b-a)/(2*np.sqrt(3)) + (b+a)/2
    return ((b-a)/2)*(lifting_integral(var_a,x,gamma)+lifting_integral(var_b,x,gamma))

# Full Numerical Integration of Lifting Integral (3.49)
def lift_two_point_gaussian_quadrature(x,gamma_dist):
    store = []
    for i in range(len(x)):
        storex = []
        for j in range(len(dummy)-1):    
            storex.append(lift_two_point_gaussian_quadrature_step(dummy[j],dummy[j+1],x[i],gamma_dist[j]))
        store.append(np.sum(storex))                                                                                             # Manually fix the velocity at trailing edge to zero
    return store

# Find the angle of adaption
def alpha_adapt(camber_profile):
    return camber_profile[-1]

# Find the coefficient of lift
def c_lift(gamma_dist):
    return 2*np.trapz(gamma_dist,dx=0.01)

# Adjusted to camber at zero incidence
def adjusted_camber_profile(camber_profile):
    adapt = alpha_adapt(camber_profile)
    store = []
    for i in range(len(x)):
        store.append(camber_profile[i] - adapt*x[i])
    return store

# Find the angle of zero lift
def zero_lift(c_lift, alpha_adapt):
    return alpha_adapt - c_lift/(2*np.pi)


################################### FLAPS #####################################

def flap_line(flap_length, flap_angle, x):
    store = []
    step = 0
    for i in range(len(x)):
        if x[i] <= (1-flap_length):
            store.append(0)
        else:
            store.append(-step*flap_angle)
            step += 0.01
    return store


################################# NACA FORM ###################################

def NACA_identifier(max_camber, mc_chord, thickness):
    M = int(round(max_camber*100,0))
    P = int(round(mc_chord*10,0))
    XX = str(int(thickness*100)).zfill(2)
    st.subheader(f"The closest 4 digit NACA airfoil to this the NACA {M}{P}{XX} airfoil.")
    st.write("4 digit NACA airfoils are expressed in the form $MPXX$, where:")
    st.markdown(f"- $M$ is the maximum camber divided by 100. Here, $M={M}$ as the camber is {max_camber} or {M}$\%$ of the chord")
    st.markdown(f"- $P$ is the position of the maximum camber divided by 10. Here, $P={P}$ so the maximum camber is at {mc_chord} or {P}$\%$ of the chord")
    st.markdown(f"- $XX$ is the thickness divided by 100. Here, $XX={XX}$ so the thickness is {thickness} or {int(thickness*100)}$\%$ of the chord.")
    st.write("Due to the nature of the 4 digits, this appoximation is limited for a maximum camber of 0.1")
    st.write("Based on the classification of NACA 4 Digit Airfoils available at [airfoiltools.com](%s): " %"http://airfoiltools.com/airfoil/naca4digit")

################################### VALUES ####################################


# Velocity, Pressure, and Lift Over Airfoil
symmetric_velocity = sym_two_point_gaussian_quadrature()
lifting_velocity_adapted = gamma_dist
upper_surface_velocity = np.add(symmetric_velocity, lifting_velocity_adapted)
lower_surface_velocity = np.subtract(symmetric_velocity, lifting_velocity_adapted)
upper_surface_pressure = np.multiply(upper_surface_velocity,-2)
lower_surface_pressure = np.multiply(lower_surface_velocity, -2)
Cl = c_lift(gamma_dist)

# Airfoil Profile
camber_profile = it.cumtrapz(lift_two_point_gaussian_quadrature(x,gamma_dist), dx=0.01, initial = 0)
camber_line = adjusted_camber_profile(camber_profile)
upper_surface = camber_line + f(x)
lower_surface = camber_line - f(x)
Adapt = alpha_adapt(camber_profile)
Zero = zero_lift(Cl, Adapt)

# Flaps
if allow_flaps:
    Flap = flap_line(flap_length, flap_angle, x)
    camber_line_flap = np.add(camber_line, Flap)
    upper_surface_flap = camber_line_flap + f(x)
    lower_surface_flap = camber_line_flap - f(x)
    theta_hinge = np.arccos(2*flap_length - 1)
    delta_Cl = 2*flap_angle*(np.pi - theta_hinge + np.sin(theta_hinge))
    delta_Zero = -(flap_angle/np.pi)*(np.pi - theta_hinge + np.sin(theta_hinge))
    delta_Adapt = -flap_angle*(1 - theta_hinge/np.pi)
    Cl_flap = Cl + delta_Cl
    Adapt_flap = Adapt + delta_Adapt
    Zero_flap = Zero + delta_Zero
else:
    flap_length, camber_line_flap, upper_surface_flap, lower_surface_flap, Zero_flap = 0, 0, 0, 0, 0

# NACA Designation
max_camber = max(camber_line)
mc_chord = x[camber_line.index(max(camber_line))]
############################# PLOTTING RESULTS ################################

def decor():
    plt.gca().grid(which = 'major')
    plt.gca().grid(which = 'minor')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.gca().set_axisbelow(True)

fig1 = plt.figure(1)
plt.ylim(-0.1, 0.5)
plt.gca().set_aspect(0.5)
plt.title(f"Cambered Airfoil Profile at Zero Incidence\n Thickness: {thickness}, Coefficient of Lift: {np.round(Cl,2)}")
plt.plot(x, camber_line, color = 'k', linewidth = 0.5)
plt.plot(x, upper_surface, color = 'k', linewidth = 0.75)
plt.plot(x, lower_surface, color = 'k', linewidth = 0.75)
if allow_flaps:
    mask = x > (1-flap_length) 
    plt.plot(x[mask], camber_line_flap[mask], color = 'red')
    plt.plot(x[mask], upper_surface_flap[mask], color = 'red')
    plt.plot(x[mask], lower_surface_flap[mask], color = 'red')
plt.xlabel('$x/c$')
plt.ylabel('$y$')
decor()

fig2 = plt.figure(2)
plt.title("Airfoil Surface Velocities")
#plt.plot(x, symmetric_velocity, color = 'b', linewidth = 0.75, label = 'Symmetric Velocity Component')
#plt.plot(x, lifting_velocity_adapted, color = 'r', linewidth = 0.75, label = 'Lifting Velocity Component')
plt.plot(x, upper_surface_velocity, color = 'orange', linewidth = 1, label = 'Upper Surface Velocity')
plt.plot(x, lower_surface_velocity, color = 'c', linewidth = 1, label = 'Lower Surface Velocity')
plt.legend(loc="center right")    
plt.xlabel('$x/c$')
plt.ylabel('v/V')

decor()

fig3 = plt.figure(3)
plt.title('Airfoil Surface Pressure')
plt.gca().invert_yaxis()
plt.plot(x,upper_surface_pressure, color = 'orange', label = 'Upper Surface Pressure')
plt.plot(x,lower_surface_pressure, color = 'c', label = 'Lower Surface Pressure')
plt.legend(loc="center right")
plt.xlabel('$x/c$')
plt.ylabel('$C_{p}$')
decor()

fig4 = plt.figure(4)
plt.title('Angle of Attack and Coefficient of Lift')
plt.plot(np.arange(-0.1, 0.1, 0.02), 2*np.pi*(np.arange(-0.1, 0.1, 0.02) - Zero), label = 'No Flaps')
if allow_flaps:
    plt.plot(np.arange(-0.1, 0.1, 0.02), 2*np.pi*(np.arange(-0.1, 0.1, 0.02) - Zero_flap), label = 'With Flaps')
    plt.legend(loc="center right")
plt.xlabel('$alpha$')
plt.ylabel('$C_{l}$')
decor()

col2A_1, col2A_2 = st.columns(2)

with col2A:
    st.pyplot(fig1)
    with st.expander("Save airfoil for comparison"):
        save_plot_A = st.button("Save to Slot A")
        if save_plot_A:
            st.session_state.Slot_gamma[0] = gamma_dist
            save_plot_A = False
        save_plot_B = st.button("Save to Slot B")
        if save_plot_B:
            st.session_state.Slot_gamma[1] = gamma_dist
            save_plot_B = False

with col1B:
    st.pyplot(fig2)
with col2B:
    st.pyplot(fig3)
with col3B:
    st.pyplot(fig4)

if not fix_cruise:
    st.header("Important Performance Metrics")
else:
    st.header("Important Performance Metrics - all scaled for $C_{l}$")



if not allow_flaps:
    series = pd.Series([Cl, Adapt, Zero, max_camber, mc_chord], index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
    df = pd.DataFrame(data=series, index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"], columns=["Values"])
    st.dataframe(df, use_container_width=True)

else:
    series = pd.Series([Cl, Adapt, Zero, max_camber, mc_chord], index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
    series_flap_delta = pd.Series([round(delta_Cl,4), round(delta_Adapt,4), round(delta_Zero,4), "-", "-"], index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
    series_flap = pd.Series([Cl_flap, Adapt_flap, Zero_flap, max_camber, mc_chord], index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
    df = pd.DataFrame({'Values':series, 'Flap Delta':series_flap_delta, 'Values Adjusted For Flaps':series_flap})
    st.dataframe(df, use_container_width=True)

st.header("4 Digit NACA Airfoil Approximation")
if max_camber < 0.1:
    NACA_identifier(max_camber, mc_chord, thickness)
else:
    st.write("The maximum camber of this airfoil is too large to be approximate in 4 digit NACA form.")

st.header("Airfoil Profile Comparison")
col1C, col2C = st.columns(2)

with col1C:
    if st.session_state.Slot_gamma[0] is not None:
        st.write("Slot A:", st.session_state.Slot_gamma[0])

        # Velocity, Pressure, and Lift Over Airfoil
        symmetric_velocity = sym_two_point_gaussian_quadrature()
        lifting_velocity_adapted = st.session_state.Slot_gamma[0]
        upper_surface_velocity = np.add(symmetric_velocity, lifting_velocity_adapted)
        lower_surface_velocity = np.subtract(symmetric_velocity, lifting_velocity_adapted)
        upper_surface_pressure = np.multiply(upper_surface_velocity,-2)
        lower_surface_pressure = np.multiply(lower_surface_velocity, -2)
        Cl = c_lift(st.session_state.Slot_gamma[0])

        # Airfoil Profile
        camber_profile = it.cumtrapz(lift_two_point_gaussian_quadrature(x,st.session_state.Slot_gamma[0]), dx=0.01, initial = 0)
        camber_line = adjusted_camber_profile(camber_profile)
        upper_surface = camber_line + f(x)
        lower_surface = camber_line - f(x)
        Adapt = alpha_adapt(camber_profile)
        Zero = zero_lift(Cl, Adapt)

        


with col2C:
    if st.session_state.Slot_gamma[1] is not None:
        st.write("Slot B:", st.session_state.Slot_gamma[1])
Slot_reset = st.button("Reset Plot Slots")
if Slot_reset:
    st.session_state.Slot_gamma = [None, None]