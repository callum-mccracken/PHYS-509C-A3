from matplotlib import pyplot as plt
import numpy as np
import matplotlib
from scipy.optimize import minimize
import utils

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# constant from question
sigma_m = 0.1

# open file, get data
filedata = np.loadtxt("sn_data.txt")
z = filedata[:,0]
m = filedata[:,1]

# put the data points on a plot
plt.scatter(z, m)

def mean_m(z_i, Q, Ω_Λ):
    """For a given z, Q, and Ω_Λ value, return the expected (mean) m."""
    return -2.5*np.log10(Q/((z_i + 1/2*z_i**2 * ((1 + 3*Ω_Λ)/2))**2))

def neg_ll(Q, Ω_Λ):
    """Negative log likelihood, derived in text."""
    # Let L_0H_0^2 = Q
    mean = mean_m(z, Q, Ω_Λ)
    return np.sum(
        np.log(np.sqrt(2*np.pi) * sigma_m) + ((m - mean)**2/(2*sigma_m**2)))

# the guesses come from asking friends what it should roughly work out to
guess = [4e-16, 7e-01]

# find the precise values by minimizing log likelihood
Q_0, Ω_Λ_0 = minimize(
    utils.one_param(neg_ll), guess, method="Nelder-Mead",
    bounds=((0, None), (0, 1))).x
print(f"{Q_0=}, {Ω_Λ_0=}")

# plot best-fit curve
z_sorted = np.array(sorted(z))
m_fit = mean_m(z_sorted, Q_0, Ω_Λ_0)
plt.plot(z_sorted, m_fit)

# finish up the plot
plt.xlabel("z")
plt.ylabel("m")
plt.tight_layout()
plt.savefig("q3_a_fit.png")
plt.cla()
plt.clf()

def neg_ll_one_param(Ω_Λ):
    return neg_ll(Q_0, Ω_Λ)

def ll_sigma_bounds(x_vals, neg_log_l_vals, n_sigma):
    """
    Find the uncertainty bounds for a neg log likelihood function.
    
    This assumes that the uncertainty bounds occur at two the x values given.

    (i.e. make your x as fine-grained as needed, we don't interpolate)
    """
    # find the minimum
    min_y_index = np.where(neg_log_l_vals == min(neg_log_l_vals))[0]
    min_y = neg_log_l_vals[min_y_index]

    if n_sigma == 1:
        increment = 0.5
    else:
        raise ValueError("This has not been coded yet")

    thresh = min_y + increment

    # find where function crosses thresh
    first_index = np.where(neg_log_l_vals < thresh)[0][0]
    last_index = np.where(neg_log_l_vals < thresh)[-1][-1]

    print(first_index, last_index)
    # return min, max x bounds
    return x_vals[first_index], x_vals[last_index]

x = np.linspace(0.6, 0.8, 1000)
y = np.array([neg_ll_one_param(xi) for xi in x])
one_sigma_bound_min, one_sigma_bound_max = ll_sigma_bounds(x, y, n_sigma=1)
plt.axvline(x[np.where(y==min(y))], linestyle='--')
plt.axhline(min(y) + 0.5, linestyle='--')
plt.plot(x, y)
plt.xlabel("$\Omega_\Lambda$")
plt.ylabel("$-\ln(L)$")
plt.savefig("q3_a_uncert.png")
plt.cla()
plt.clf()

print(
    "central value:", Ω_Λ_0,
    "upper uncertainty:", one_sigma_bound_max - Ω_Λ_0,
    "lower uncertainty:", Ω_Λ_0 - one_sigma_bound_min)

"""
PART B
Now do it again, with L_0 = L_0(1 + az)!

Q = L_0 H_0^2, so Q = Q(1+az) now
"""
def new_mean_m(z_i, Q, Ω_Λ, a):
    """For a given z, Q, and Ω_Λ value, return the expected (mean) m."""
    return -2.5*np.log10(Q*(1+a*z_i)/((z_i + 1/2*z_i**2 * ((1 + 3*Ω_Λ)/2))**2))

def a_prior(a):
    return utils.gaussian_pdf(a, mu=0, sigma=0.2)

def new_neg_ll(Q, Ω_Λ, a):
    """Negative log likelihood, derived in text."""
    # Let L_0H_0^2 = Q
    mean = new_mean_m(z, Q, Ω_Λ, a)
    return np.sum(
        np.log(np.sqrt(2*np.pi) * sigma_m) + ((m - mean)**2/(2*sigma_m**2))
        ) + np.log(a_prior(a))

# the guesses come from asking friends what it should roughly work out to
guess = [4e-16, 7e-01, 0]

# find the precise values by minimizing log likelihood
Q_0, Ω_Λ_0, a_0 = minimize(
    utils.one_param(new_neg_ll), guess, method="Nelder-Mead",
    bounds=((0, None), (0, 1), (None, None))).x
print(f"{Q_0=}, {Ω_Λ_0=}, {a_0=}")

# plot best-fit curve
z_sorted = np.array(sorted(z))
m_fit = new_mean_m(z_sorted, Q_0, Ω_Λ_0, a_0)
plt.scatter(z, m)
plt.plot(z_sorted, m_fit)

# finish up the plot
plt.xlabel("$z$")
plt.ylabel("$m$ (with systematic $a$)")
plt.tight_layout()
plt.savefig("q3_b_fit.png")
plt.cla()
plt.clf()

def new_neg_ll_one_param(Ω_Λ):
    return new_neg_ll(Q_0, Ω_Λ, a_0)

def ll_sigma_bounds(x_vals, neg_log_l_vals, n_sigma):
    """
    Find the uncertainty bounds for a neg log likelihood function.
    
    This assumes that the uncertainty bounds occur at two the x values given.

    (i.e. make your x as fine-grained as needed, we don't interpolate)
    """
    # find the minimum
    min_y_index = np.where(neg_log_l_vals == min(neg_log_l_vals))[0]
    min_y = neg_log_l_vals[min_y_index]

    if n_sigma == 1:
        increment = 0.5
    else:
        raise ValueError("This has not been coded yet")

    thresh = min_y + increment

    # find where function crosses thresh
    first_index = np.where(neg_log_l_vals < thresh)[0][0]
    last_index = np.where(neg_log_l_vals < thresh)[-1][-1]

    print(first_index, last_index)
    # return min, max x bounds
    return x_vals[first_index], x_vals[last_index]

x = np.linspace(0.6, 0.8, 1000)
y = np.array([new_neg_ll_one_param(xi) for xi in x])
one_sigma_bound_min, one_sigma_bound_max = ll_sigma_bounds(x, y, n_sigma=1)
plt.axvline(x[np.where(y==min(y))], linestyle='--')
plt.axhline(min(y) + 0.5, linestyle='--')
plt.plot(x, y)
plt.xlabel("$\Omega_\Lambda$")
plt.ylabel("$-\ln(L)$ with systematic $a$")
plt.savefig("q3_b_uncert.png")
plt.cla()
plt.clf()

print(
    "central value:", Ω_Λ_0,
    "upper uncertainty:", one_sigma_bound_max - Ω_Λ_0,
    "lower uncertainty:", Ω_Λ_0 - one_sigma_bound_min)



