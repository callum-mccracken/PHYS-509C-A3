"""For Q4 -- plots of posterior distributions"""
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib
from scipy.integrate import quad
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def posterior(p_1, n_1, n_2, n_3):
    def numerator(_p_1):
        return _p_1**n_1 * (0.8 - 2*_p_1)**n_2 * (0.2 + _p_1)**n_3
    return numerator(p_1) / quad(numerator, 0, 0.4)[0]

x = np.linspace(0, 0.4, 1000)
y = posterior(x, 12, 3, 5)
plt.plot(x,y,label="$n_1=12,n_2=3,n_3=5$")
plt.xlabel("$p_1$")
plt.ylabel("$P(p_1|n_1,n_2,n_3)$")
plt.legend()
plt.tight_layout()
plt.savefig("q4_b.png")

y2 = posterior(x, 12, 7, 1)
plt.plot(x,y2,label="$n_1=12,n_2=7,n_3=1$")
plt.legend()
plt.tight_layout()
plt.savefig("q4_c.png")


plt.cla()
plt.clf()