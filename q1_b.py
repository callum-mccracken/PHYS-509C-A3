"""
For Question 1, B.

Derivation is in q1.tex.
"""
from math import factorial
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy.integrate import quad
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

m_vals = np.arange(0.01, 12, 0.01)

def unnorm_posterior(m):
    if m < 3.1:
        return 0
    else:
        return 1/m**3 * np.exp(-(m-3)**2 / 2)
normalization = quad(unnorm_posterior, 0, np.inf)[0]
posterior = np.array([
    unnorm_posterior(m) for m in m_vals]) / normalization

plt.title("Q1, B.")
plt.xlabel("m")
plt.ylabel("Posterior Probability Density")
plt.plot(m_vals, posterior)
plt.savefig("q1_b.png")
