import pandas as pd
import numpy as np


def save(consA, consB, consD, step):
    '''Function saving concentrations'''

    np.save(f'data/consA/A{step}.npy', consA)
    np.save(f'data/consB/B{step}.npy', consB)
    np.save(f'data/consD/D{step}.npy', consD)


def save_h(height, steps):
    '''Function saving heights of the forest as funtion of time'''

    df = pd.DataFrame({"step": steps, "height": height})
    df.to_csv("data/height.csv", index=False)
