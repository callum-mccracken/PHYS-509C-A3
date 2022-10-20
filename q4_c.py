"""
Now suppose we add a procedure called ``rebalancing''.
On January 1 of each year we contribute a total of \$3000 to the account,
but at the same time we redistribute the total amount of money in the account
evenly between the three investments.

How does this change the total amount on Dec 31, 2047?
Show a plot of the distribution, and report the mean and SD as well.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


trials = np.array(range(1000))
years = np.array(range(2018, 2048))
values = np.zeros((len(trials), len(years)))

mu_C = 0.08
sigma_C = 0.15
mu_F = 0.08
sigma_F = 0.15
mu_B = 0.05
sigma_B = 0.07
rho_CF = 0.50
rho_CB = 0.20
rho_FB = 0.05

cov_CF = rho_CF * sigma_C * sigma_F
cov_CB = rho_CB * sigma_C * sigma_B
cov_FB = rho_FB * sigma_F * sigma_B

for i, trial in enumerate(trials):
    current_value_C = 1000
    current_value_F = 1000
    current_value_B = 1000
    for j, year in enumerate(years):
        yld_C, yld_F, yld_B = np.random.multivariate_normal(
            mean=np.array([mu_C, mu_F, mu_B]),
            cov=np.array([
                [sigma_C**2, cov_CF, cov_CB],
                [cov_CF, sigma_F**2, cov_FB],
                [cov_CB, cov_FB, sigma_B**2]]))
        current_value_C = (1+yld_C) * current_value_C
        current_value_F = (1+yld_F) * current_value_F
        current_value_B = (1+yld_B) * current_value_B
        total = current_value_C + current_value_B + current_value_F + 3000
        current_value_C = total / 3
        current_value_F = total / 3
        current_value_B = total / 3
        values[i, j] = total
    plt.scatter(years, values[i], alpha=0.05, c='b')
means = np.mean(values, axis=0)
stds = np.std(values, axis=0)
plt.plot(years, means, c='k')
plt.errorbar(years, means, stds, c='k')
plt.xlabel("Year")
plt.ylabel("Dollars")
plt.savefig("q4_c_justforme.png")
plt.cla()
plt.clf()

plt.hist(values[:,-1], bins=20)
plt.xlabel("Dollar Value on Dec 31, 2047")
plt.ylabel("Number of trials")
plt.savefig("q4_c.png")

print("Value on Dec 31, 2047:", means[-1])
print("Standard deviation:", stds[-1])
