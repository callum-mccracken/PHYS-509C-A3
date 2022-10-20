from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import dblquad
from utils import binomial_pmf
import scipy.special as sc
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# note: for speed reasons I've substituted
# integral(p^k(1-p)^(n-k) from 0 to 1) = beta(k+1, n-k+1)

def prob_false_pos(false_pos_rate):
    prior = 1
    k=16
    n=3324
    likelihood = binomial_pmf(k=k, n=n, p=false_pos_rate)
    normalization = sc.beta(k+1,n-k+1)
    return prior * likelihood / normalization

def prob_false_neg(false_neg_rate):
    prior = 1
    k=27
    n=157
    likelihood = binomial_pmf(k=k, n=n, p=false_neg_rate)
    normalization = sc.beta(k+1,n-k+1)
    return prior * likelihood / normalization

def prob_pos(pos_rate):
    prior = 1
    k=50
    n=3330
    likelihood = binomial_pmf(k=k, n=n, p=pos_rate)
    normalization = sc.beta(k+1,n-k+1)
    return prior * likelihood / normalization

def prob_antibodies(false_pos_rate, false_neg_rate, pos_rate):
    return (pos_rate - false_pos_rate*pos_rate) / (
        1 - false_neg_rate*(1-pos_rate) - false_pos_rate*pos_rate)

pos_rates = np.arange(0, 1, 0.01)

marginalized_posteriors = []
for pos_rate in pos_rates:
    marginalized = dblquad(
        lambda x, y: prob_antibodies(x, y, pos_rate), 0, 1, 0, 1)[0]
    marginalized_posteriors.append(marginalized)

# Submit a plot of the posterior distribution for the true incidence rate
plt.plot(pos_rates, marginalized_posteriors)
plt.xlabel("Probability of positive tests")
plt.ylabel("Probability of really having antibodies")
plt.savefig("q2.png")

