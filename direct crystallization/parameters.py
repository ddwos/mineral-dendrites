'''Module with all constants in the algorithm'''

import numpy as np

save_interval = 500
max_h = 0.75  # ratio of max height at which program terminates
threshold = 0.05  # minimal neighbouring threshold for aggregation

L1 = 100  # length (even)
L2 = 100
N1 = int(L1)
N2 = int(L2)
dt = 1  # time step

steps = 2000000  # max number of steps
alpha = 1
DA = 0.4  # DA, DB, DC - diffusion coefs
DB = DA * 0.1
k = 0.01  # reaction rate
limitD = 0.1  # cell volume

kp = 0.02  # aggregation threshold
ksp = 10**4  # nucleation threshold
ka = 0
r = 0.1  # stochastic parameters
A = 3  # surface energy parameter
a0 = 0.04  # initial A concentration
b0 = a0 * 20  # initial B concentration

l = 5  # surface energy parameters
n0 = (l - 1) / l / 2

v = 3  # stochastic parameter
change = 1  # auxiliary var

c = np.array([[0,  0], [0, -1], [0, 1],
              [-1, 0], [-1, -1], [-1, 1],
              [1, 0], [1, -1], [1, 1]])  # spatial directions
