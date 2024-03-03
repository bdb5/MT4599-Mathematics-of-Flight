#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:57:51 2024

@author: benbroughton
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


st.write("hello world")
y_points = [1, 1.5, 0.8, 0.5]
x_points = [0, 0.33, 0.66, 1]

f = CubicSpline(x_points, y_points, bc_type='natural')
x_new = np.arange(0, 1.01, 0.01)
y_new = f(x_new)

fig =  plt.figure()
plt.plot(x_new, y_new, 'b')
plt.plot(x_points, y_points, 'ro')
plt.title('Cubic Spline Test for Pressure Distribution')
plt.xlabel('x')
plt.ylabel('y')

st.pyplot(fig)