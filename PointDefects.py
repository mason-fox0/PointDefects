#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:34:13 2022

@author: Mason Fox
Written for the final project of MATH 578 at UTK - Spring 2022

Purpose: Simulate the diffusion of point defects (vacancies and interstitials) in a material subject to irradiation damage.
"""

import numpy as np
import matplotlib.pyplot as plt


radFlux = 10**6     #particles/(cm^2*s)

#define geometry - rectangular slab
xmin = 0        #meters
xmax = 1    #meters
ymin = 0        #meters
ymax = 5        #meters
numdX = 5
numdY = 25

#spatial discretization/mesh
grid = np.ones((numdY, numdX, 2)) #3rd dim represents concentrations of vacancies, interstitials (essentially 2 2D matrices)

print(grid)


#time discretization
t_start = 0
t_end = 10      #seconds
numdT = 101    
T, step = np.linspace(t_start, t_end, numdT, retstep=True)

#BCs
grid[0, :, :]= 0
grid[:, 0, :] = 0 #zero defects on boundary

#explicit forward euler
