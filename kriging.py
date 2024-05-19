#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:17:11 2024

@author: ctutum
"""

#%% Example-1: Interpolating-Kriging without noisy data

import matplotlib.pyplot as plt
import numpy as np

from smt.surrogate_models import KRG

xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])

sm = KRG(theta0=[1e-2])
sm.set_training_values(xt, yt)
sm.train()

num = 100
x = np.linspace(0.0, 4.0, num)
y = sm.predict_values(x)
# estimated variance
s2 = sm.predict_variances(x)
# derivative according to the first variable
_dydx = sm.predict_derivatives(xt, 0)
_, axs = plt.subplots(1)

# add a plot with variance
axs.plot(xt, yt, "o")
axs.plot(x, y)
axs.fill_between(
    np.ravel(x),
    np.ravel(y - 3 * np.sqrt(s2)),
    np.ravel(y + 3 * np.sqrt(s2)),
    color="lightgrey",
)
axs.set_xlabel("x")
axs.set_ylabel("y")
axs.legend(
    ["Training data", "Prediction", "Confidence Interval 99%"],
    loc="lower right",
)

plt.savefig("interpolating-kriging.png", dpi=300)
plt.show()

#%% Example 2: Regressing-Kriging with noisy data

import matplotlib.pyplot as plt
import numpy as np

from smt.surrogate_models import KRG

# defining the toy example
def target_fun(x):
    import numpy as np

    return np.cos(5 * x)

nobs = 50  # number of obsertvations
np.random.seed(0)  # a seed for reproducibility
xt = np.random.uniform(size=nobs)  # design points

# adding a random noise to observations
yt = target_fun(xt) + np.random.normal(scale=0.05, size=nobs)

# training the model with the option eval_noise= True
sm = KRG(eval_noise=True, hyper_opt="Cobyla")
sm.set_training_values(xt, yt)
sm.train()

# predictions
x = np.linspace(0, 1, 100).reshape(-1, 1)
y = sm.predict_values(x)  # predictive mean
var = sm.predict_variances(x)  # predictive variance

# plotting predictions +- 3 std confidence intervals
plt.rcParams["figure.figsize"] = [8, 4]
plt.fill_between(
    np.ravel(x),
    np.ravel(y - 3 * np.sqrt(var)),
    np.ravel(y + 3 * np.sqrt(var)),
    alpha=0.2,
    label="Confidence Interval 99%",
)
plt.scatter(xt, yt, label="Training noisy data")
plt.plot(x, y, label="Prediction")
plt.plot(x, target_fun(x), label="target function")
plt.title("Kriging model with noisy observations")
plt.legend(loc=0)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.savefig("regressing-kriging.png", dpi=300)

#%%



