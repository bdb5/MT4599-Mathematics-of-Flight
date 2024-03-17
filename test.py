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

style = "<style>h1, h2{text-align: center;}</style>"
style_centered = "<style>.centered {text-align: center;}</style>"
st.markdown(style_centered, unsafe_allow_html=True)
st.markdown(style, unsafe_allow_html=True)

st.title("The Thin Line Between Flying and Falling")
st.markdown("<h3 class='centered'>The Inverse Design of Wings Using Thin Airfoil</h3>", unsafe_allow_html=True)


if 'Slot_gamma' not in st.session_state:
    st.session_state.Slot_gamma = [None, None]
    st.session_state.Slot_thickness = [None, None]
    st.session_state.Slot_camber_line = [None, None]
    st.session_state.Slot_upper_surface = [None, None]
    st.session_state.Slot_lower_surface = [None, None]
    st.session_state.Slot_max_camber = [None, None]
    st.session_state.Slot_mc_chord = [None, None]
    st.session_state.Slot_Cl = [None, None]  # Ensure this is initialized
    st.session_state.Slot_Zero = [None, None]
    st.session_state.Slot_Adapt = [None, None]
    st.session_state.Slot_upper_surface_velocity = [None, None]
    st.session_state.Slot_lower_surface_velocity = [None, None]
    st.session_state.Slot_upper_surface_pressure = [None, None]
    st.session_state.Slot_lower_surface_pressure = [None, None]
    st.session_state.Slot_allow_flaps = [False, False]
    st.session_state.Slot_flap_length = [None, None]
    st.session_state.Slot_camber_line_flap = [None, None]
    st.session_state.Slot_upper_surface_flap = [None, None]
    st.session_state.Slot_lower_surface_flap = [None, None]
    st.session_state.Slot_delta_Cl = [None, None]
    st.session_state.Slot_delta_Adapt = [None, None]
    st.session_state.Slot_delta_Zero = [None, None]
    st.session_state.Slot_Cl_flap = [None, None]
    st.session_state.Slot_Adapt_flap = [None, None]
    st.session_state.Slot_Zero_flap = [None, None]

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
p1a, p2a, p3a = st.columns([1, 6, 1])
col1A, col2A = st.columns([4, 5])


if not fix_cruise:
    st.header("Important Plots")
else:
    st.header("Important Performance Metrics - all scaled for $C_{l}$")
col1B, col2B, col3B = st.columns(3)
with col1A:
    fig = plt.figure(0)
    plt.gca().grid(which = 'major')
    plt.gca().grid(which = 'minor')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.gca().set_axisbelow(True)
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
    return [M,P,XX]

############################## STALL ANALYSIS #################################

def stall(gamma_dist):
    max_load = max(gamma_dist)
    min_load = min(gamma_dist)
    max_load_loc = np.argmax(gamma_dist)
    min_load_loc = np.argmin(gamma_dist)
    if max_load_loc<= 51:
        stall_type = "Leading Edge"
    else:
        stall_type = "Trailing Edge"
    return [stall_type, max_load_loc]


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
    flap_length, camber_line_flap, upper_surface_flap, lower_surface_flap, Zero_flap, delta_Cl, delta_Adapt, delta_Zero, \
        Cl_flap, Adapt_flap, Zero_flap = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

# NACA Designation
max_camber = max(camber_line)
mc_chord = x[camber_line.index(max(camber_line))]
############################# PLOTTING RESULTS ################################

def decor():
    plt.gca().grid(which = 'major')
    plt.gca().grid(which = 'minor')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.gca().set_axisbelow(True)

def profile_plot(camber_line,upper_surface,lower_surface,thickness, Cl, allow_flaps,\
                 flap_length, camber_line_flap, upper_surface_flap, lower_surface_flap):
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

def velocity_plot(upper_surface_velocity,lower_surface_velocity):
    plt.ylim(-1, 3)
    plt.title("Airfoil Surface Velocities")
    plt.plot(x, upper_surface_velocity, color = 'orange', linewidth = 1, label = 'Upper Surface Velocity')
    plt.plot(x, lower_surface_velocity, color = 'c', linewidth = 1, label = 'Lower Surface Velocity')
    plt.legend(loc="center right")    
    plt.xlabel('$x/c$')
    plt.ylabel('v/V')
    decor()

def pressure_plot(upper_surface_pressure,lower_surface_pressure):
    plt.ylim(-6, 6)
    plt.title('Airfoil Surface Pressure')
    plt.gca().invert_yaxis()
    plt.plot(x,upper_surface_pressure, color = 'orange', label = 'Upper Surface Pressure')
    plt.plot(x,lower_surface_pressure, color = 'c', label = 'Lower Surface Pressure')
    plt.legend(loc="center right")
    plt.xlabel('$x/c$')
    plt.ylabel('$C_{p}$')
    decor()

def cl_plot(Zero, allow_flaps, Zero_flap):
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
    fig1 = plt.figure(1)
    profile_plot(camber_line,upper_surface,lower_surface,thickness, Cl, allow_flaps,\
                        flap_length, camber_line_flap, upper_surface_flap, lower_surface_flap)
    st.pyplot(fig1)
    with st.expander("Save airfoil for comparison"):
        save_plot_A = st.button("Save to Slot A")
        if save_plot_A:
            st.session_state.Slot_gamma[0] = gamma_dist
            st.session_state.Slot_thickness[0] = thickness
            st.session_state.Slot_camber_line[0] = camber_line
            st.session_state.Slot_upper_surface[0] = upper_surface
            st.session_state.Slot_lower_surface[0] = lower_surface
            st.session_state.Slot_max_camber[0] = max_camber
            st.session_state.Slot_mc_chord[0] = mc_chord
            st.session_state.Slot_Cl[0] = Cl
            st.session_state.Slot_Zero[0] = Zero
            st.session_state.Slot_Adapt[0] = Adapt
            st.session_state.Slot_upper_surface_velocity[0] = upper_surface_velocity
            st.session_state.Slot_lower_surface_velocity[0] = lower_surface_velocity
            st.session_state.Slot_upper_surface_pressure[0] = upper_surface_pressure
            st.session_state.Slot_lower_surface_pressure[0] = lower_surface_pressure
            st.session_state.Slot_allow_flaps[0] = allow_flaps
            st.session_state.Slot_flap_length[0] = flap_length
            st.session_state.Slot_camber_line_flap[0] = camber_line_flap
            st.session_state.Slot_upper_surface_flap[0] = upper_surface_flap
            st.session_state.Slot_lower_surface_flap[0] = lower_surface_flap
            st.session_state.Slot_delta_Cl[0] = delta_Cl
            st.session_state.Slot_delta_Adapt[0] = delta_Adapt
            st.session_state.Slot_delta_Zero[0] = delta_Zero
            st.session_state.Slot_Cl_flap[0] = Cl_flap
            st.session_state.Slot_Adapt_flap[0] = Adapt_flap
            st.session_state.Slot_Zero_flap[0] = Zero_flap
            save_plot_A = False
        save_plot_B = st.button("Save to Slot B")
        if save_plot_B:
            st.session_state.Slot_gamma[1] = gamma_dist
            st.session_state.Slot_thickness[1] = thickness
            st.session_state.Slot_camber_line[1] = camber_line
            st.session_state.Slot_upper_surface[1] = upper_surface
            st.session_state.Slot_lower_surface[1] = lower_surface
            st.session_state.Slot_max_camber[1] = max_camber
            st.session_state.Slot_mc_chord[1] = mc_chord
            st.session_state.Slot_Cl[1] = Cl
            st.session_state.Slot_Zero[1] = Zero
            st.session_state.Slot_Adapt[1] = Adapt
            st.session_state.Slot_upper_surface_velocity[1] = upper_surface_velocity
            st.session_state.Slot_lower_surface_velocity[1] = lower_surface_velocity
            st.session_state.Slot_upper_surface_pressure[1] = upper_surface_pressure
            st.session_state.Slot_lower_surface_pressure[1] = lower_surface_pressure
            st.session_state.Slot_allow_flaps[1] = allow_flaps
            st.session_state.Slot_flap_length[1] = flap_length
            st.session_state.Slot_camber_line_flap[1] = camber_line_flap
            st.session_state.Slot_upper_surface_flap[1] = upper_surface_flap
            st.session_state.Slot_lower_surface_flap[1] = lower_surface_flap
            st.session_state.Slot_delta_Cl[1] = delta_Cl
            st.session_state.Slot_delta_Adapt[1] = delta_Adapt
            st.session_state.Slot_delta_Zero[1] = delta_Zero
            st.session_state.Slot_Cl_flap[1] = Cl_flap
            st.session_state.Slot_Adapt_flap[1] = Adapt_flap
            st.session_state.Slot_Zero_flap[1] = Zero_flap
            save_plot_B = False
with col1B:
    fig2 = plt.figure(2)
    velocity_plot(upper_surface_velocity,lower_surface_velocity)
    st.pyplot(fig2)
with col2B:
    fig3 = plt.figure(3)
    pressure_plot(upper_surface_pressure,lower_surface_pressure)
    st.pyplot(fig3)
with col3B:
    fig4 = plt.figure(4)
    cl_plot(Zero, allow_flaps, Zero_flap)
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
    NACA_val = NACA_identifier(max_camber, mc_chord, thickness)
    url_template = f"http://airfoiltools.com/airfoil/naca4digit?MNaca4DigitForm[camber]={NACA_val[0]}&MNaca4DigitForm[position]={NACA_val[1]*10}&MNaca4DigitForm[thick]={NACA_val[2]}&MNaca4DigitForm[numPoints]=81&MNaca4DigitForm[cosSpace]=0&MNaca4DigitForm[cosSpace]=1&MNaca4DigitForm[closeTe]=0&yt0=Plot"
    st.markdown(f"Experimental data for the {NACA_val[0]}{NACA_val[1]}{NACA_val[2]} profile is available [here](%s)" % url_template)

else:
    st.write("The maximum camber of this airfoil is too large to be approximate in 4 digit NACA form.")

st.header("Stall Characteristic Approximation")
stall_info = stall(gamma_dist)
st.subheader(f'{stall(gamma_dist)[0]} Stall is more likely.')
st.caption(f"This wing is more likely to experience {stall_info[0]} stall as the location of peak pressure is\
            {stall_info[1]}$\%$ along the chord, and therefore nearer the {stall_info[0]} of the wing")


st.header("Airfoil Profile Comparison")
col1C, col2C = st.columns(2)

with col1C:
    if st.session_state.Slot_gamma[0] is not None:
        fig15 = plt.figure(15)
        decor()
        plt.plot(x, st.session_state.Slot_gamma[0], 'b')
        if not allow_negative:
            plt.ylim(-0.1,2.1)
        else:
            plt.ylim(-2.1,2.1)
        plt.title('Cubic Spline Interpolation of Load Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        st.pyplot(fig15)
        fig5 = plt.figure(5)
        profile_plot(np.squeeze(st.session_state.Slot_camber_line[0]),np.squeeze(st.session_state.Slot_upper_surface[0]),\
                     np.squeeze(st.session_state.Slot_lower_surface[0]),np.squeeze(st.session_state.Slot_thickness[0]), \
                     np.squeeze(st.session_state.Slot_Cl[0]), np.squeeze(st.session_state.Slot_allow_flaps[0]),\
                     np.squeeze(st.session_state.Slot_flap_length[0]), np.squeeze(st.session_state.Slot_camber_line_flap[0]),\
                     np.squeeze(st.session_state.Slot_upper_surface_flap[0]), np.squeeze(st.session_state.Slot_lower_surface_flap[0]))
        st.pyplot(fig5)
        fig6 = plt.figure(6)
        velocity_plot(np.squeeze(st.session_state.Slot_upper_surface_velocity[0]),np.squeeze(st.session_state.Slot_lower_surface_velocity[0]))
        st.pyplot(fig6)
        fig7 = plt.figure(7)
        pressure_plot(np.squeeze(st.session_state.Slot_upper_surface_pressure[0]),np.squeeze(st.session_state.Slot_lower_surface_pressure[0]))
        st.pyplot(fig7)
        fig8 = plt.figure(8)
        cl_plot(np.squeeze(st.session_state.Slot_Zero[0]), np.squeeze(st.session_state.Slot_allow_flaps[0]), np.squeeze(st.session_state.Slot_Zero_flap[0]))
        st.pyplot(fig8)
        st.subheader('Performance Metrics', divider='orange')
        series = pd.Series([np.squeeze(st.session_state.Slot_Cl[0]), np.squeeze(st.session_state.Slot_Adapt[0]), \
                        np.squeeze(st.session_state.Slot_Zero[0]), np.squeeze(st.session_state.Slot_max_camber[0]), np.squeeze(st.session_state.Slot_mc_chord[0])], \
                        index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
        if not np.squeeze(st.session_state.Slot_allow_flaps[0]):
            df = pd.DataFrame(data=series, index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"], columns=["Values"])
            st.dataframe(df, use_container_width=True)
        else:
            series_flap_delta = pd.Series([round(np.squeeze(st.session_state.Slot_delta_Cl[0]),4), round(np.squeeze(st.session_state.Slot_delta_Adapt[0]),4), \
                                           round(np.squeeze(st.session_state.Slot_delta_Zero[0]),4), "-", "-"], index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
            series_flap = pd.Series([np.squeeze(st.session_state.Slot_Cl_flap[0]), np.squeeze(st.session_state.Slot_Adapt_flap[0]), \
                                     np.squeeze(st.session_state.Slot_Zero_flap[0]), np.squeeze(st.session_state.Slot_max_camber[0]), np.squeeze(st.session_state.Slot_mc_chord[0])], \
                            index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
            df = pd.DataFrame({'Values':series, 'Flap Delta':series_flap_delta, 'Values Adjusted For Flaps':series_flap})
            st.dataframe(df, use_container_width=True)


with col2C:
    if st.session_state.Slot_gamma[1] is not None:
        fig14 = plt.figure(14)
        plt.plot(x, st.session_state.Slot_gamma[1], 'b')
        decor()
        if not allow_negative:
            plt.ylim(-0.1,2.1)
        else:
            plt.ylim(-2.1,2.1)
        plt.title('Cubic Spline Interpolation of Load Distribution')
        plt.xlabel('x')
        plt.ylabel('y')
        st.pyplot(fig14)
        fig9 = plt.figure(9)
        profile_plot(np.squeeze(st.session_state.Slot_camber_line[1]),np.squeeze(st.session_state.Slot_upper_surface[1]),\
                     np.squeeze(st.session_state.Slot_lower_surface[1]),np.squeeze(st.session_state.Slot_thickness[1]), \
                     np.squeeze(st.session_state.Slot_Cl[1]), np.squeeze(st.session_state.Slot_allow_flaps[1]),\
                     np.squeeze(st.session_state.Slot_flap_length[1]), np.squeeze(st.session_state.Slot_camber_line_flap[1]),\
                     np.squeeze(st.session_state.Slot_upper_surface_flap[1]), np.squeeze(st.session_state.Slot_lower_surface_flap[1]))
        st.pyplot(fig9)
        fig10 = plt.figure(10)
        velocity_plot(np.squeeze(st.session_state.Slot_upper_surface_velocity[1]),np.squeeze(st.session_state.Slot_lower_surface_velocity[1]))
        st.pyplot(fig10)
        fig11 = plt.figure(11)
        pressure_plot(np.squeeze(st.session_state.Slot_upper_surface_pressure[1]),np.squeeze(st.session_state.Slot_lower_surface_pressure[1]))
        st.pyplot(fig11)
        fig12 = plt.figure(12)
        cl_plot(np.squeeze(st.session_state.Slot_Zero[1]), np.squeeze(st.session_state.Slot_allow_flaps[1]), np.squeeze(st.session_state.Slot_Zero_flap[1]))
        st.pyplot(fig12)
        st.subheader('Performance Metrics', divider='orange')
        series = pd.Series([np.squeeze(st.session_state.Slot_Cl[1]), np.squeeze(st.session_state.Slot_Adapt[1]), \
                            np.squeeze(st.session_state.Slot_Zero[1]), np.squeeze(st.session_state.Slot_max_camber[1]), np.squeeze(st.session_state.Slot_mc_chord[1])], \
                            index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
        if not np.squeeze(st.session_state.Slot_allow_flaps[1]):
            df = pd.DataFrame(data=series, index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"], columns=["Values"])
            st.dataframe(df, use_container_width=True)
        else:
            series_flap_delta = pd.Series([round(np.squeeze(st.session_state.Slot_delta_Cl[1]),4), round(np.squeeze(st.session_state.Slot_delta_Adapt[1]),4), \
                                        round(np.squeeze(st.session_state.Slot_delta_Zero[1]),4), "-", "-"], index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
            series_flap = pd.Series([np.squeeze(st.session_state.Slot_Cl_flap[1]), np.squeeze(st.session_state.Slot_Adapt_flap[1]), \
                                    np.squeeze(st.session_state.Slot_Zero_flap[1]), np.squeeze(st.session_state.Slot_max_camber[1]), np.squeeze(st.session_state.Slot_mc_chord[1])], \
                        index=["Coefficient of lift", "Angle of Adaption", "Angle of Zero Lift", "Maximum Camber", "Max Camber Position"])
            df = pd.DataFrame({'Values':series, 'Flap Delta':series_flap_delta, 'Values Adjusted For Flaps':series_flap})
            st.dataframe(df, use_container_width=True)

if st.session_state.Slot_gamma[0] is not None and st.session_state.Slot_gamma[1] is not None:
    min_gradient_0 = min(np.gradient(st.session_state.Slot_gamma[0]))
    min_gradient_1 = min(np.gradient(st.session_state.Slot_gamma[1]))
    if min_gradient_0 < min_gradient_1:
        with col1C:
            st.subheader('Stall Analysis', divider = 'orange')
            st.caption('More likely to stall')
            st.write(f'As this airfoil has a greater maximum adverse pressure gradient of {min_gradient_0} compared to\
                     the pressure gradient of the airfoil in Column B: {min_gradient_1}, it is expected that this airfoil is MORE likely to stall')
        with col2C:
            st.subheader('Stall Analysis', divider = 'orange')
            st.caption('Less likely to stall')
            st.write(f'As this airfoil has a smaller maximum adverse pressure gradient of {min_gradient_1} compared to\
                     the pressure gradient of the airfoil in Column B: {min_gradient_0}, it is expected that this airfoil is LESS likely to stall')
    if min_gradient_0 > min_gradient_1:
        with col1C:
            st.subheader('Stall Analysis', divider = 'orange')
            st.caption('Less likely to stall')
            st.write(f'As this airfoil has a smaller maximum adverse pressure gradient of {round(min_gradient_0,4)} compared to\
                    the pressure gradient of the airfoil in Column B: {round(min_gradient_1,4)}, it is expected that this airfoil is LESS likely to stall')
        with col2C:
            st.subheader('Stall Analysis', divider = 'orange')
            st.caption('More likely to stall')
            st.write(f'As this airfoil has a greater maximum adverse pressure gradient of {round(min_gradient_1,4)} compared to\
                    the pressure gradient of the airfoil in Column B: {round(min_gradient_0,4)}, it is expected that this airfoil is MORE likely to stall')
    with col1C:
        st.caption(f'{stall(np.squeeze(st.session_state.Slot_gamma[0]))[0]} Stall is more likely.')
        stall_info = stall(np.squeeze(st.session_state.Slot_gamma[0]))
        st.caption(f"This wing is more likely to experience {stall_info[0]} stall as the location of peak pressure is\
                   {stall_info[1]}$\%$ along the chord, and therefore nearer the {stall_info[0]} of the wing")
    with col2C:
        st.caption(f'{stall(np.squeeze(st.session_state.Slot_gamma[1]))[0]} Stall is more likely.')
        stall_info = stall(np.squeeze(st.session_state.Slot_gamma[1]))
        st.caption(f"This wing is more likely to experience {stall_info[0]} stall as the location of peak pressure is\
                   {stall_info[1]}$\%$ along the chord, and therefore nearer the {stall_info[0]} of the wing")

Slot_reset = st.button("Reset Plot Slots", use_container_width=True, type = 'primary')

if Slot_reset:
    st.session_state.Slot_gamma = [None, None]
    st.session_state.Slot_thickness = [None, None]
    st.session_state.Slot_camber_line = [None, None]
    st.session_state.Slot_upper_surface = [None, None]
    st.session_state.Slot_lower_surface = [None, None]
    st.session_state.Slot_max_camber = [None, None]
    st.session_state.Slot_mc_chord = [None, None]
    st.session_state.Slot_Cl = [None, None]
    st.session_state.Slot_Zero = [None, None]
    st.session_state.Slot_Adapt = [None, None]
    st.session_state.Slot_upper_surface_velocity = [None, None]
    st.session_state.Slot_lower_surface_velocity = [None, None]
    st.session_state.Slot_upper_surface_pressure = [None, None]
    st.session_state.Slot_lower_surface_pressure = [None, None]
    st.session_state.Slot_allow_flaps = [False, False]
    st.session_state.Slot_flap_length = [None, None]
    st.session_state.Slot_camber_line_flap = [None, None]
    st.session_state.Slot_upper_surface_flap = [None, None]
    st.session_state.Slot_lower_surface_flap = [None, None]
    st.session_state.Slot_delta_Cl = [None, None]
    st.session_state.Slot_delta_Adapt = [None, None]
    st.session_state.Slot_delta_Zero = [None, None]
    st.session_state.Slot_Cl_flap = [None, None]
    st.session_state.Slot_Adapt_flap = [None, None]
    st.session_state.Slot_Zero_flap = [None, None]