import numpy as np
from collections import Counter
from quspin.tools.misc import mean_level_spacing
from utils import LoadResults, GetMeta
from config import PROJECT_ROOT


def GetEmid(E, prec, mid_sample):
    if E is None:
        return [None] * 3

    middle_index = len(E) // 2
    E_mid = E[middle_index:]
    E_mid = E_mid[E_mid > prec]
    middle_index = len(E) - len(E_mid)

    num, ratio = mid_sample
    
    if num is not False:
        if len(E_mid) >= num:
            end_index = num
        else:
            print(f"**W: Number of samples {num} is larger than the number of available samples {len(E_mid)}.")
            end_index = len(E_mid)
    elif ratio is not False:
        end_index = int(len(E_mid) * ratio)
    else:
        raise ValueError("Both num and ratio cannot be False")

    E_mid = E_mid[:end_index]
    print(f"mean level spacing ratio: {mean_level_spacing(E_mid)}")
    return middle_index, end_index, E_mid

def degeneracy_check(E):
    # Find the degeneracy in E
    counter = Counter(E)
    degeneracies = {level: count for level, count in counter.items() if count > 1}
    
    if degeneracies:
        print("Degenerate levels and their degrees:")
        for level, degree in degeneracies.items():
            print(f"Level {level} is degenerate with degree {degree}")
        return -1
    else:
        print("No degenerate levels found.")
        return 0

def GetEDiff(E_mid):
    # Calculate the differences between energy levels
    E_diff = np.diff(E_mid)

    # Calculate the mean of E_mid
    E_diff_avg = np.mean(E_diff)

    # Rescale the differences with respect to the mean
    E_diff_rescaled = E_diff / E_diff_avg

    # Calculate the ratio of the differences between energy levels
    r = E_diff[:-1] / E_diff[1:]
    # If the quotient is larger than 1, take the reciprocal
    r = np.where(r > 1, 1 / r, r)

    return E_diff_rescaled, r

def GetSstat(param):
    io_dict = dict(read_E=False, read_V=False, read_S=True, write_E=False, write_V=False, write_S=False)
    _, _, entropy = LoadResults(f"{PROJECT_ROOT}/res/{GetMeta(param)}", io_dict)
    # return the mean and standard deviation of the entropy
    return np.mean(entropy), np.std(entropy)