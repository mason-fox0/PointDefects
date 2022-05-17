This code was developed as a final project for the MATH 578 - Numerical methods for PDEs at
UTK, Spring 2022.

Purpose: Model and simulate the behavior of vacancies and interstitial point defects in
materials subject to a radiation field.

Model: Two partial differential equations are formed to model the time rate of change of both
vacancies and point defects. The model considers diffusion through the material,
recombination of vacancies and interstitials, generation due to irradiation, and loss
to sinks.

Method: Crank-Nicholson Adams-Bashforth (2 step) IMEX Scheme (Finite Difference)

Output: Graphics of defect behavior over time

Required Dependencies:
-Python 3.0+
-Scipy
-Numpy
-Matplotlib

References:
D. Olander, A. Motta - Ch 5,12,13 - Light Water Reactor Materials Vol. I (2017)

S. Wise - MATH578 Course Notes - Spring 2022, UT Knoxville

Python Documentation

U. Asher, S. Ruuth, B. Wetton - IMPLICIT-EXPLICIT METHODS FOR TIME-DEPENDENT PDE'S - https://www.cs.ubc.ca/sites/default/files/tr/1993/TR-93-15_0.pdf
