from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from utils import binomial_pmf
import scipy.special as sc
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

filedata = np.loadtxt("sn_data.txt")
z = filedata[:,0]
m = filedata[:,1]

# Submit a plot of the posterior distribution for the true incidence rate
plt.scatter(z, m)
plt.xlabel("Measured z")
plt.ylabel("Measured m")
plt.savefig("q2.png")

