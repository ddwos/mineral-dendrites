'''Module used to implement chemical reactions in the system as well as 
physical processes like aggregation and nucleation'''

import numpy as np
import pystencils as ps
import parameters as par

L1 = par.L1  # length (even)
L2 = par.L2

N1 = int(L1)
N2 = int(L2)

l = par.l  # parameter of surface energy

# auxiliary arrays
arr = np.zeros(((N1 + 1), (N2 + 1)))
arr2 = np.pad(arr, ((l-2, l-2), (l-2, l-2)), mode='wrap')
arr3 = np.pad(arr, ((1, 1), (1, 1)), mode='wrap')

TAB = []
for i in np.arange(- (l - 1) // 2, (l - 1) // 2 + 1, 1):
    for j in np.arange(-(l - 1) // 2, (l - 1) // 2 + 1, 1):
	    TAB.append((i, j))

# Use of pystencils in order to sum concentrations in a square shaped vicinity
f, g = ps.fields("f, g : [2D]")
stencil = ps.Assignment(g[0, 0],
                        (np.sum([f[el[0], el[1]] for el in TAB])))
kernel_prob = ps.create_kernel(stencil).compile()

# Use of pystencils in order to sum neighbours in Moore convention
f, g = ps.fields("f, g : [2D]")
stencil = ps.Assignment(g[0, 0],
                        (np.sum([f[el[0], el[1]] for el in par.c])))
kernel_neigh = ps.create_kernel(stencil).compile()


def probability(cons, a, l, n0, ker=kernel_prob, n=arr2):
    '''Function used to model surface tension - inspired by paper of 
    Vicsek. DOI:https://doi.org/10.1103/PhysRevLett.53.2281'''

    cons = np.where(cons > 0., 1., 0.)
    cons = np.pad(cons, ((l-2, l-2), (l-2, l-2)), mode='wrap')
    ker(f=cons, g=n)

    n = n[(l-2):-(l-2), (l-2):-(l-2)]
    n = n/l**2
    p = 1/2 + a * (n - n0)
    p = np.where(p > 1, 1, p)
    p = np.where(p < 0, 0, p)
    return p


def reaction(consA, consB, consC, k):
    '''Funciton used to implement chemical reaction of type A + B -> C'''

    product = consA * consB
    R = k * product
    u1 = np.where(R > 0, 1, 0)
    R = np.where((R > consA) + (R > consB),
                 np.minimum(consA*1, consB*1) * u1, R)
    consA = consA - R
    consB = consB - R
    consC = consC + R
    return consA, consB, consC


def agregation_on_top(consC, consD, ka, v, limitD, A,
                      L1, L2, l, n0, alpha):
    '''Function used to implement aggregation of nanoparticles
    on top of the existing precipitate cells'''
    
    condition1 = (consD > 0)
    condition2 = (consC > ka)

    rand = (np.random.rand(L1+1, L2+1) < 1 * v * consC)
    rand2 = (np.random.rand(L1+1, L2+1) < probability(consD, A, l, n0))

    u = np.where(condition1 * condition2 * rand * rand2,
                 alpha * consC, 0)
    tab = np.where(consD > limitD, 0, 1)
    u1 = np.where(condition1 * condition2 * rand * rand2, 1, 0)

    beta = u * tab
    beta = np.where((beta > consC), consC * u1, beta)

    consC = consC - beta
    consD = consD + beta
    return consC, consD


def agregation_in_vicinity(consC, consD, kp, v, limitD, A,
                           L1, L2, l, n0, alpha, q, c, neigh_sum = arr3):
    '''Function used to implement aggregation of nanoparticles
    in the vicinity of the existing precipitate cells'''

    rand = (np.random.rand(L1+1, L2+1) < v * consC)
    rand2 = (np.random.rand(L1+1, L2+1) < probability(consD, A, l, n0))

    consD_prime = np.pad(consD, ((1, 1), (1, 1)),
                         mode='wrap')
    kernel_neigh(f=consD_prime, g=neigh_sum)
    neigh_sum = neigh_sum[1:-1, 1:-1]

    condition = neigh_sum > limitD * par.threshold
    u1 = np.where(condition *
                  (consC > kp) * rand * rand2, 1, 0)
    u = np.where(condition *
                 (consC > kp) * rand * rand2, alpha * consC, 0)

    tab = np.where(consD > limitD, 0, 1)

    beta = u*tab
    beta = np.where((beta > consC), consC * u1, beta)

    consC = consC - beta
    consD = consD + beta

    return consC, consD


def nucleation(L1, L2, consC, consD, ksp, limitD, r, alpha):
    '''Function used to implement spontaneous nucleation in the 
    reaction-diffusion system'''

    rand = (np.random.rand(L1+1, L2+1) < r)
    u = np.where((consC > ksp) * rand, alpha * consC, 0)
    u1 = np.where((consC > ksp) * rand, 1, 0)
    tab = np.where(consD > limitD, 0, 1)

    beta = u * tab
    beta = np.where((beta > consC), consC * u1, beta)

    consC = consC - beta
    consD = consD + beta

    return consC, consD
