import math

import numpy as np
import matplotlib.pyplot as plt


def plot(function, x_min=-5.0, x_max=5.0, noise_factor=2, seed=0):
    np.random.seed(seed)

    x = np.arange(x_min, x_max, 0.1)
    yhat = np.apply_along_axis(function, axis=0, arr=x)
    noise = np.random.normal(size=x.size)
    y = yhat + noise

    plt.plot(x, yhat, "r")
    plt.plot(x, y, "bo")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def test1():
    f = lambda x: 4*x-3
    plot(f, noise_factor=3)


def test2():
    f = lambda x: math.log(x)
    plot(f, x_min=0.1, noise_factor=1)


if __name__ == "__main__":
    test1()
    test2()
