"""
# Reference:
http://leansixsigmadefinition.com/glossary/nelson-rules/

# Definition
A 1984 update by Lloyd S. Nelson to the popular Western Electric Rules
(WECO Rules), in order to make the probability of detecting an out-of-
control condition by chance approximately equal across all tests. The
Western Electric Rules have probabilities that have more variability
from one another (some are more likely than others to occur by chance).
"""


from itertools import combinations

import numpy as np

from utils.control_charts import load_input, xbar_s_chart


# MAIN FUNCTIONS
def nelson_rule_1(series, X_bar, s):
    """
    Outlier: One point is more than 3 standard deviations from the mean.
    """
    violating_indices = []
    for i in range(len(series)):
        if np.abs(series[i]-X_bar) > 3*s:
            violating_indices.append(i)

    return violating_indices


def nelson_rule_2(series, X_bar):
    """
    Shift: Nine or more points in a row are on the same side of the mean.
    """
    n = len(series)
    if n < 9:
        return []

    violating_indices = set()
    for i in range(n-8):
        if (all([series[j] > X_bar for j in range(i, i+9)]) or
            all([series[j] < X_bar for j in range(i, i+9)])):
            violating_indices.update(range(i, i+9))

    return sorted(violating_indices)


def nelson_rule_3(series):
    """
    Trend: Six or more points in a row are continually increasing or
    decreasing.
    """
    n = len(series)
    if n < 5:
        return []

    violating_indices = set()
    for i in range(n-4):
        if (all([series[j] <= series[j+1] for j in range(i, i+4)]) or
            all([series[j] >= series[j+1] for j in range(i, i+4)])):
            violating_indices.update(range(i, i+5))
        
    return sorted(violating_indices)


def nelson_rule_4(series):
    """
    Bimodal, 2 or more factors in data set: Forteen or more points in a
    row alternate in direction, increasing then decreasing.
    """
    n = len(series)
    if n < 14:
        return []

    violating_indices = set()
    for i in range(n-13):
        if all([(series[j]-series[j+1])*(series[j+2]-series[j+1]) >= 0
                 for j in range(i, i+12)]):
            violating_indices.update(range(i, i+14))

    return sorted(violating_indices)


def nelson_rule_5(series, X_bar, s):
    """
    Shift: Two or three out of three points in a row are more than 2
    standard deviations from the mean in the same direction.
    """
    n = len(series)
    if n < 3:
        return []

    violating_indices = set()
    for i in range(n-2):
        for combination in combinations({i, i+1, i+2}, 2):
            j, k = combination
            if np.min([series[j], series[k]]) > X_bar + 2*s:
                violating_indices.update({j, k})
            elif np.max([series[j], series[k]]) < X_bar - 2*s:
                violating_indices.update({j, k})

    return sorted(violating_indices)


def nelson_rule_6(series, X_bar, s):
    """
    Shift or trend: Four or five out of five points in a row are more
    than 1 standard deviation from the mean in the same direction.
    """
    n = len(series)
    if n < 5:
        return []

    violating_indices = set()
    for i in range(n-4):
        for combination in combinations({i, i+1, i+2, i+3, i+4}, 4):
            k, l, m, n = combination
            if np.min([series[k], series[l],
                       series[m], series[n]]) > X_bar + s:
                violating_indices.update({k, l, m, n})
            elif np.max([series[k], series[l],
                         series[m], series[n]]) < X_bar - s:
                violating_indices.update({k, l, m, n})

    return sorted(violating_indices)


def nelson_rule_7(series, X_bar, s):
    """
    Reduced variation or measurement issue: Fifteen points in a row are
    all within 1 standard deviation of the mean on either side of the
    mean.
    """
    n = len(series)
    if n < 15:
        return []

    violating_indices = set()
    for i in range(n-14):
        if all([np.abs(series[j]-X_bar) < s for j in range(i, i+15)]):
            violating_indices.update(range(i, i+15))
                
    return sorted(violating_indices)


def nelson_rule_8(series, X_bar, s):
    """
    Bimodal, 2 or more factors in data set: Eight points in a row exist
    with none within 1 standard deviation of the mean and the points are
    in both directions from the mean.
    """
    n = len(series)
    if n < 8:
        return []

    violating_indices = set()
    for i in range(n-7):
        if all([np.abs(series[j]-X_bar) > s for j in range(i, i+8)]):
            both_sided = False
            for combination in combinations(set(range(i, i+8)), 2):
                j, k = combination
                if (series[j]-X_bar)*(series[k]-X_bar) < 0:
                    both_sided = True
                    break
            if both_sided:
                violating_indices.update(range(i, i+8))

    return sorted(violating_indices)


def nelson_rules(series, X_bar, s):
    single_rule_results = [
        nelson_rule_1(series, X_bar, s),
        nelson_rule_2(series, X_bar),
        nelson_rule_3(series),
        nelson_rule_4(series),
        nelson_rule_5(series, X_bar, s),
        nelson_rule_6(series, X_bar, s),
        nelson_rule_7(series, X_bar, s),
        nelson_rule_8(series, X_bar, s)
    ]
    
    violating_indices = set()
    violating_indices.update(*single_rule_results)

    return sorted(violating_indices), single_rule_results


# TESTING
def test_nelson_rule_1():
    violating_indices = nelson_rule_1(
        [10, -1, -2, -3, -4, 5, 6, 7, 0, 8, 1, 5, 0, 4, 7, 3, 2],
        0, 2)
    print(violating_indices)


def test_nelson_rule_2():
    violating_indices = nelson_rule_2(
        [10, -1, -2, -3, -4, 5, 6, 7, 1, 8, 1, 5, 1, 4, 7, 3, 2],
        0)
    print(violating_indices)


def test_nelson_rule_3():
    violating_indices = nelson_rule_3(
        [10, -1, -2, -3, -4, 5, 6, 7, 7, 8, 1, 5, 0, 4, 7, 3, 2])
    print(violating_indices)


def test_nelson_rule_4():
    violating_indices = nelson_rule_4(
        [10, 8, 9, 6, 7, 4, 5, 3, 4, 2, 3, 1, 2, 0, 1, 2, 0])
    print(violating_indices)


def test_nelson_rule_5():
    violating_indices = nelson_rule_5(
        [10, -1, -2, -3, -4, 5, 6, 7, 0, 8, 1, 5, 0, 4, 7, 3, 2],
        0, 2)
    print(violating_indices)


def test_nelson_rule_6():
    violating_indices = nelson_rule_6(
        [10, -1, -2, -3, -4, 5, 6, 7, 0, 8, 1, 5, 0, 4, 7, 3, 2],
        0, 2)
    print(violating_indices)


def test_nelson_rule_7():
    violating_indices = nelson_rule_7(
        [10, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 0, -1, 1, 1, 0],
        0, 2)
    print(violating_indices)


def test_nelson_rule_8():
    violating_indices = nelson_rule_8(
        [10, -3, 4, 3, 5, -5, 6, 7, -3, 1, 8, 5, 0, 4, 7, 3, 2],
        0, 2)
    print(violating_indices)


def test_nelson_rules():
    violating_indices, _ = nelson_rules(
        [10, -1, -2, -3, -4, 5, 6, 7, 0, 8, 1, 5, 0, 4, 7, 3, 2],
        0, 2)
    print(violating_indices, _)


if __name__ == "__main__":
    test_nelson_rules()
