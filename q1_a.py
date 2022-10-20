"""
Question 1, A.
"""
from math import factorial
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

T = 1  # in millenia
k = 4  # observed events in one millenium

rates = np.linspace(0.1,12,1000)

plt.title("Q1, A.")
plt.xlabel("R [supernovae/millenium]")
plt.ylabel("Posterior Probability Density")
unif_R_posteriors = T*np.exp(-rates*T)*(rates*T)**k/factorial(k)
unif_logR_posteriors = T*np.exp(-rates*T)*(rates*T)**(k-1)/factorial(k-1)
plt.plot(rates, unif_R_posteriors, label="Uniform $R$ Prior")
plt.plot(rates, unif_logR_posteriors, label="Uniform $\log_{10}(R)$ Prior")
plt.legend()
plt.savefig("q1_a.png")
