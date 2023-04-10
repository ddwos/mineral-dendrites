import warnings
import numpy as np


def ignore_warrnings():
    '''Function used to ignore inconsequential warnings'''

    def fxn():
        warnings.warn("deprecated", DeprecationWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
    warnings.filterwarnings("ignore")


def check_negative(consA, consB, consD, step):
    '''Function checks if any of the concentration is negative
    and informs user if it is'''

    check_a = consA < 0
    check_b = consB < 0
    check_d = consD < 0
    if ((np.any(check_a) + np.any(check_b) + np.any(check_d)
         ) * (step > 1000)):
        print("ERROR", step)
