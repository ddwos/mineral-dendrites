import numpy as np

import matplotlib.pyplot as plt

def heigth(consD, l):
    i = 0
    while(not(np.any(consD[i]))):
        i+=1
    return len(consD) - i - 2 * l + 1
