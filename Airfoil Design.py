#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:57:51 2024

@author: benbroughton
"""

import streamlit as st
import numpy as np
from scipy import constants
from scipy.interpolate import CubicSpline
import scipy.integrate as it
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib.ticker import AutoMinorLocator, FixedLocator

input_form = st.selectbox(
   "How would you like to define your pressure distribution?",
   ("Manually", "CSV"),
   index=0,
   placeholder="Select input method...",
)

allow_negative = st.toggle('Allow Negative Values for Pressure Distribution')
thickness = st.slider('Please select airfoil thickness', 0.01, 0.2)
if not allow_negative:
    sliderlim = 0.0
    plotlim = -0.1
else:
    sliderlim = -2.0
    plotlim = -0.5
if input_form == "Manually":
    A = st.slider('Please select point A value', sliderlim, 2.0, 0.0)
    B = st.slider('Please select point B value', sliderlim, 2.0, 0.0)
    C = st.slider('Please select point C value', sliderlim, 2.0, 0.0)
    D = st.slider('Please select point D value', sliderlim, 2.0, 0.0)
    E = st.slider('Please select point E value', sliderlim, 2.0, 0.0)
    F = st.slider('Please select point F value', sliderlim, 2.0, 0.0)


    x_val = [0, 0.2, 0.4, 0.6, 0.8, 1]
    y_val= [A, B, C, D, E, F]

if input_form == "CSV":
    x_val = []
    y_val = []
    input_csv = st.text_input('Insert values here. Ensure centered on x axis')
    coords = input_csv.split()
    for i in coords:
        x_str, y_str = i.split(",")
        x_val.append(float(x_str))
        y_val.append(float(y_str))


f = CubicSpline(x_val, y_val, bc_type='natural')
x = np.linspace(0, 1.01, 101)
gamma_dist = f(x)

if not allow_negative:
    replace_negatives = np.vectorize(lambda x: 0 if x < 0 else x)
    gamma_dist = replace_negatives(gamma_dist)

fig = plt.figure(0)
plt.plot(x, gamma_dist, 'b')
plt.plot(x_val, y_val, 'ro')
if not allow_negative:
    plt.ylim(-0.1,2.1)
else:
    plt.ylim(-2.1,2.1)
plt.title('Cubic Spline Interpolation - note that below zero values could be an \
          issue - go through the list and fox to zero when negative')
plt.xlabel('x')
plt.ylabel('y')
st.pyplot(fig)
st.write('This gives a coefficient of lift')




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



############################# PLOTTING RESULTS ################################

def decor():
    plt.gca().grid(which = 'major')
    plt.gca().grid(which = 'minor')
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.gca().set_axisbelow(True)
    
fig1 = plt.figure(1)
plt.ylim(plotlim, 0.5)
plt.gca().set_aspect(0.5)
plt.title(f"Cambered Airfoil Profile at Zero Incidence\n Thickness: {thickness}, Coefficient of Lift: {Cl}")
plt.plot(x, camber_line, color = 'k', linewidth = 0.5)
plt.plot(x, upper_surface, color = 'k', linewidth = 0.75)
plt.plot(x, lower_surface, color = 'k', linewidth = 0.75)
decor()

fig2 = plt.figure(2)
plt.title("Airfoil Surface Velocities")
#plt.plot(x, symmetric_velocity, color = 'b', linewidth = 0.75, label = 'Symmetric Velocity Component')
#plt.plot(x, lifting_velocity_adapted, color = 'r', linewidth = 0.75, label = 'Lifting Velocity Component')
plt.plot(x, upper_surface_velocity, color = 'orange', linewidth = 1, label = 'Upper Surface Velocity')
plt.plot(x, lower_surface_velocity, color = 'c', linewidth = 1, label = 'Lower Surface Velocity')
plt.legend(loc="center right")
decor()

fig3 = plt.figure(3)
plt.title('Airfoil Surface Pressure')
plt.gca().invert_yaxis()
plt.plot(x,upper_surface_pressure, color = 'orange', label = 'Upper Surface Pressure')
plt.plot(x,lower_surface_pressure, color = 'c', label = 'Lower Surface Pressure')
plt.legend(loc="center right")
decor()

fig4 = plt.figure(4)
plt.title('Angle of Attack and Coefficient of Lift')
plt.gca().set_aspect(0.2)
plt.xlim(-0.175,0.35)
plt.plot(np.arange(-0.175, 0.35, 0.02), 2*np.pi*(np.arange(-0.175, 0.35, 0.02) - Zero))
decor()

st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)
st.pyplot(fig4)

st.write("Coefficient of lift:", Cl)
st.write("Angle of Adaption", Adapt)
st.write("Angle of Zero Lift", Zero)