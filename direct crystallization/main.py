'''Main module. Run this module in order to run simulation.'''

import numpy as np
import time
import chemical as chem
import errors
import lbm
import quantification as quant
import saving
import parameters as par


errors.ignore_warrnings()

# parameters described in the module parameters.py
L1 = par.L1
L2 = par.L2
N1 = int(L1)
N2 = int(L2)
dt = par.dt
steps = par.steps
alpha = par.alpha
DA = par.DA
DB = par.DB
k = par.k
limitD = par.limitD
kp = par.kp
ksp = par.ksp
ka = par.ka
r = par.r
A = par.A
a0 = par.a0
b0 = par.b0
l = par.l
n0 = par.n0
v = par.v
change = par.change

start = time.time()

# initial conditions
consB = np.zeros((N1 + 1, (N2 + 1)))  # concentrations
consA = np.zeros((N1 + 1, (N2 + 1)))
consD = np.zeros((N1 + 1, (N2 + 1)))

for i in range(L1 + 1):
    if i < L1 - l + 2:
        consB[i] = b0
    else:
        continue

for i in range(l+1, 2*l):
    consD[len(consD) - (i)] = limitD + 0.001

# initializing lattice boltzmann
(feqA, feqB, finA, finB, omegaA, omegaB, d, q, c, w
 ) = lbm.initialize_lbm(consA, consB, DA, DB)

heights = np.array([])
st = np.array([])

for i in range(0, steps, dt):

    step = i
    if ((step+0) % par.save_interval == 0):
        print('step:', i, 'time:', time.time()-start, 's')
        saving.save(consA, consB, consD, step)
        if (quant.heigth(consD, l) > (L1 - 2 * l) * par.max_h):
            break

    errors.check_negative(consA, consB, consD, step)

    consA_old = np.copy(consA)
    consB_old = np.copy(consB)

    # agregation on top
    consA, consB, consD = chem.agregation_on_top(consA, consB, consD, ka, v,
                                                 limitD, A, L1, L2, l, n0,
                                                 alpha)

    # agregation in vicinity
    consA, consB, consD = chem.agregation_in_vicinity(consA, consB, consD,
                                                      kp, v, limitD, A, L1,
                                                      L2, l, n0, alpha, q, c)

    # mass conservation
    finA, finB = lbm.mass_conservation(consA, consB, consA_old, consB_old,
                                       finA, finB)

    # LBM
    # collision step
    foutA, foutB = lbm.collision_step(w, consA, consB, omegaA, omegaB,
                                      finA, finB)

    # boundary bounce back
    foutA, foutB, finA, finB = lbm.apply_boundary_conditions(foutA, foutB,
                                                             finA, finB, l)

    # streaming step
    finA, finB = lbm.streaming_step(q, foutA, foutB, finA, finB, c)

    consA, consB = lbm.sumpop(finA), lbm.sumpop(finB)

    consA[1] = a0


print('time:', time.time()-start, 's')

consA[0], consA[-1] = consA[1], consA[-2]
consB[0], consB[-1] = consB[1], consB[-2]
consD[0], consD[-1] = consD[1], consD[-2]
