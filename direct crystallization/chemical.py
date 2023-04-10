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


def reaction(consA, consB, consD, low_barrier, init_barrier, new_init, k):
    '''Funciton used to implement chemical reaction of type A + B -> D'''

    product = consA * consB
    R = np.where(product < low_barrier, 0, k * (product - low_barrier))
    init = np.where(product > init_barrier, 1, 0)
    new_init = np.logical_or(init, new_init)
    R = R * new_init
    consA = consA - R
    consB = consB - R
    consD = consD + R

    return consA, consB, consD


def agregation_on_top(consA, consB, consD, ka, v, limitD, A,
                      L1, L2, l, n0, alpha):
    '''Function used to implement aggregation of MnO
    on top of the existing precipitate'''

    condition1 = (consD > 0)
    condition2 = (consA*consB > ka)

    rand = (np.random.rand(L1+1, L2+1) < 1 * v * consA * consB)
    rand2 = (np.random.rand(L1+1, L2+1) < probability(consD, A, l, n0))

    u = np.where(condition1 * condition2 * rand * rand2,
                 alpha * consA * consB, 0)
    tab = np.where(consD > limitD, 0, 1)
    u1 = np.where(condition1 * condition2 * rand * rand2, 1, 0)

    beta = u * tab
    beta = np.where((beta > consA) + (beta > consB),
                    np.minimum(consA*1, consB*1) * u1, beta)

    consA = consA - beta
    consB = consB - beta
    consD = consD + beta
    return consA, consB, consD


def agregation_in_vicinity(consA, consB, consD, kp, v, limitD, A,
                           L1, L2, l, n0, alpha, q, c, neigh_sum = arr3):
    '''Function used to implement aggregation of MnO
    in the vicinity of the existing precipitate'''

    rand = (np.random.rand(L1+1, L2+1) < v * consA * consB)
    rand2 = (np.random.rand(L1+1, L2+1) < probability(consD, A, l, n0))

    consD_prime = np.pad(consD, ((1, 1), (1, 1)),
                         mode='wrap')
    kernel_neigh(f=consD_prime, g=neigh_sum)
    neigh_sum = neigh_sum[1:-1, 1:-1]

    condition = neigh_sum > limitD * par.threshold

    u1 = np.where(condition *
                  (consA * consB > kp) * rand * rand2, 1, 0)
    u = np.where(condition *
                 (consA * consB > kp) * rand * rand2, alpha*consA * consB, 0)

    tab = np.where(consD > limitD, 0, 1)

    beta = u*tab
    beta = np.where((beta > consA) + (beta > consB),
                    np.minimum(consA * 1, consB * 1) * u1, beta)

    consA = consA - beta
    consB = consB - beta
    consD = consD + beta
    
    return consA, consB, consD


def nucleation(L1, L2, consA, consB, consD, ksp, limitD, r, alpha):
    '''Function used to implement spontaneous nucleation in the 
    reaction-diffusion system'''
    
    rand = (np.random.rand(L1+1, L2+1) < r)
    u = np.where((consA * consB > ksp) * rand, alpha * consA * consB, 0)
    u1 = np.where((consA * consB > ksp) * rand, 1, 0)
    tab = np.where(consD > limitD, 0, 1)

    beta = u * tab
    beta = np.where((beta > consA) + (beta > consB),
                    np.minimum(consA*1, consB*1) * u1, beta)
    consA = consA - beta
    consB = consB - beta
    consD = consD + beta
    
    return consA, consB, consD
