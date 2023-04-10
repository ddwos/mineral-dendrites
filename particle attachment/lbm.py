'''Module used for implementation of the Lattice-Boltzmann method in order
to solve numerically reaction-diffusion equations in the physical system'''

import numpy as np


def roll_concatenate(array, shift, axis):
    '''Optimized numpy roll function using np.concatenate'''

    if axis == 0:
        return np.concatenate((array[-shift:], array[:-shift]), axis=0)

    elif axis == 1:
        return np.concatenate((array[:, -shift:], array[:, :-shift]), axis=1)

    else:
        raise ValueError("Invalid axis argument")


def sumpop(fin):
    '''Function summing populations in Lattice-Boltzmann
    for distribution functions'''

    return np.sum(fin, axis=0)


def initialize_lbm(consA, consB, consC, DA, DB, DC):
    '''Function initializing Lattice-Boltzmann method by 
    assignment of distribution functions, spatial directions 
    and corresponding weights'''

    # spatial directions in D2Q9 lattice
    c = np.array([[0,  0], [0, -1], [0, 1],
                 [-1, 0], [-1, -1], [-1, 1],
                 [1, 0], [1, -1], [1, 1]])

    d = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])

    tauA = 3 * DA + 0.5
    tauB = 3 * DB + 0.5
    tauC = 3 * DC + 0.5

    omegaA = 1 / tauA
    omegaB = 1 / tauB
    omegaC = 1 / tauC

    q = len(c)  # N of populations
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/36,
                  1/36, 1/9, 1/36, 1/36])  # weights

    # equilibrium distribution functions
    feqA = consA[None, :, :] * w[:, None, None]
    feqB = consB[None, :, :] * w[:, None, None]
    feqC = consC[None, :, :] * w[:, None, None]

    finA = feqA
    finB = feqB
    finC = feqC

    return (feqA, feqB, feqC, finA, finB, finC, omegaA, omegaB, omegaC,
            d, q, c, w)


def mass_conservation(consA, consB, consC, consA_old, consB_old, consC_old,
                      finA, finB, finC):
    '''Function used for mass conservation of compounds present in the
    reaction-diffusion system'''

    ratioA = np.where(consA_old == 0, 1, consA / consA_old)
    ratioB = np.where(consB_old == 0, 1, consB / consB_old)
    ratioC = np.where(consC_old == 0, 1, consC / consC_old)
    finA = finA * ratioA
    finB = finB * ratioB
    finC = finC * ratioC

    return finA, finB, finC


def collision_step(w, consA, consB, consC, omegaA, omegaB, omegaC,
                   finA, finB, finC):
    '''Collision step of the Lattice-Boltzmann method'''

    feqA = consA[None, :, :] * w[:, None, None]
    feqB = consB[None, :, :] * w[:, None, None]
    feqC = consC[None, :, :] * w[:, None, None]

    foutA = finA - omegaA * (finA - feqA)  # Collision step.
    foutB = finB - omegaB * (finB - feqB)
    foutC = finC - omegaC * (finC - feqC)

    return foutA, foutB, foutC


def apply_boundary_conditions(foutA, foutB, foutC, finA, finB, finC, l):
    '''Applying reflecting boundary conditions at the bottom edge
    of the system'''

    tab = [(1, 2), (7, 5), (4, 8), (3, 6), (0, 0)]
    for el in tab:
        foutA[el[0], 0] = finA[el[1], 0]
        foutA[el[1], 0] = finA[el[0], 0]
        foutB[el[0], 0] = finB[el[1], 0]
        foutB[el[1], 0] = finB[el[0], 0]
        foutC[el[0], 0] = finC[el[1], 0]
        foutC[el[1], 0] = finC[el[0], 0]

        foutA[el[0], -l] = finA[el[1], -l]
        foutA[el[1], -l] = finA[el[0], -l]
        foutB[el[0], -l] = finB[el[1], -l]
        foutB[el[1], -l] = finB[el[0], -l]
        foutC[el[0], -l] = finC[el[1], -l]
        foutC[el[1], -l] = finC[el[0], -l]

    return foutA, foutB, foutC, finA, finB, finC


def streaming_step(q, foutA, foutB, foutC, finA, finB, finC, c):
    '''Streaming step of the Lattice-Boltzmann method'''

    for i in range(q):  # Streaming step.
        finA[i, :, :] = roll_concatenate(
            roll_concatenate(foutA[i, :, :], c[i, 0],
                             axis=0), c[i, 1], axis=1)

        finB[i, :, :] = roll_concatenate(
            roll_concatenate(foutB[i, :, :], c[i, 0],
                             axis=0), c[i, 1], axis=1)

        finC[i, :, :] = roll_concatenate(
            roll_concatenate(foutC[i, :, :], c[i, 0],
                             axis=0), c[i, 1], axis=1)

    return finA, finB, finC
