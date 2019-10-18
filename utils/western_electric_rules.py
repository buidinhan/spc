from itertools import combinations

import numpy as np

from utils.control_charts import load_input, xbar_s_chart


# MAIN FUNCTIONS
def western_electric_rule_1(series, X_bar, s):
    violating_indices = []
    for i in range(len(series)):
        if np.abs(series[i]-X_bar) > 3*s:
            violating_indices.append(i)

    return violating_indices


def western_electric_rule_2(series, X_bar, s):
    n = len(series)
    if n < 3:
        return []

    violating_indices = set()
    for i in range(n-2):
        for combination in combinations({i, i+1, i+2}, 2):
            k, l = combination
            if np.min([series[k], series[l]]) > X_bar + 2*s:
                violating_indices.update({k, l})
            elif np.max([series[k], series[l]]) < X_bar - 2*s:
                violating_indices.update({k, l})

    return sorted(violating_indices)


def western_electric_rule_3(series, X_bar, s):
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


def western_electric_rule_4(series, X_bar):
    n = len(series)
    if n < 8:
        return []

    violating_indices = set()
    for i in range(n-7):
        if (all([series[j] > X_bar for j in range(i, i+8)]) or
            all([series[j] < X_bar for j in range(i, i+8)])):
            violating_indices.update(range(i, i+8))

    return sorted(violating_indices)


def western_electric_rules(series, X_bar, s):
    set_1 = set(western_electric_rule_1(series, X_bar, s))
    set_2 = set(western_electric_rule_2(series, X_bar, s))
    set_3 = set(western_electric_rule_3(series, X_bar, s))
    set_4 = set(western_electric_rule_4(series, X_bar))

    violating_indices = set()
    violating_indices.update(set_1, set_2, set_3, set_4)

    return sorted(violating_indices), \
           [sorted(s) for s in (set_1, set_2, set_3, set_4)]


# TESTING
def test_western_electric_rule_1():
    df, _, _, _ = load_input("../testing/fridge.dat")
    means, X_bar, s, _, _ = xbar_s_chart(df)
    violating_indices = western_rule_1(means, X_bar, s)
    print(violating_indices)


def test_western_electric_rule_2():
    df, _, _, _ = load_input("../testing/fridge.dat")
    means, X_bar, s, _, _ = xbar_s_chart(df)
    violating_indices = western_rule_2(means, X_bar, s)
    print(violating_indices)


def test_western_electric_rule_3():
    df, _, _, _ = load_input("../testing/fridge.dat")
    means, X_bar, s, _, _ = xbar_s_chart(df)
    violating_indices = western_rule_3(means, X_bar, s)
    print(violating_indices)


def test_western_electric_rule_4():
    df, _, _, _ = load_input("../testing/fridge.dat")
    means, X_bar, _, _, _ = xbar_s_chart(df)
    violating_indices = western_rule_4(means, X_bar)
    print(violating_indices)


def test_western_electric_rules():
    violating_indices, _ = western_electric_rules(
        [10, -1, -2, -3, -4, 5, 6, 7, 0, 8, 1, 5, 0, 4, 7, 3, 2],
        0, 2)
    print(violating_indices, _)
    

if __name__ == "__main__":
    test_western_electric_rules()
