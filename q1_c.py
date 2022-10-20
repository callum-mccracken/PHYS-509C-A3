"""
For question 1. c)

Suppose that an unmanned rocket is being launched,
and that at the time of the launch a certain electronic component
is either functioning or not functioning.

In the control centre there is a warning light that is not completely reliable.
If the electronic component is not functioning,
the warning light goes on with probability 1/2;
if the component is functioning,
the warning light goes on with probability 1/3.

P(on|func) = 1/2
P(on|~func) = 1/3

At the time of launch, the operator looks at the light and must
decide whether to abort the launch.

If she aborts the launch when the component is functioning well,
she wastes $2M. If she doesn't abort the launch but the component has failed,
she wastes $5M.
If she aborts the launch when the component is malfunctioning,
or if she lets the launch proceed when the component is working normally,
there is no cost.

Suppose that the prior probability of the component failing is 2/5.
During launch the warning light doesn't go on.

From a costs standpoint, should she abort the mission or not?
Compute and compare the expected cost of launching
to the expected cost of aborting, given that the light didn't go on.
"""

from matplotlib import pyplot as plt
import pandas as pd

hypotheses = "fail", "no fail"
prior_probs = 2/5, 3/5
priors = pd.Series(prior_probs, hypotheses)
print("priors")
print(priors)

# probabilities that we see NO light in each case
likelihoods = 1/2, 1 - 1/3
print("likelihoods")
print(likelihoods)
normalization_factor = (priors * likelihoods).sum()
print("normalization")
print(normalization_factor)

# probs of failure given NO light seen
posteriors = likelihoods * priors / normalization_factor
print("posteriors")
print(posteriors)
# launch cost = p(fail|light)cost(fail|light) + p(~fail|light)cost(~fail|light)
p_fail, p_nofail = posteriors
# the numbers come from the question statement, e.g. 5M for a failed launch
launch_cost = 5 * p_fail + 0 * p_nofail  # in millions
no_launch_cost = 0 * p_fail + 2 * p_nofail  # in millions
print("costs")
print(launch_cost, no_launch_cost)

# From a costs standpoint, should she abort the mission or not?
print("launch" if launch_cost < no_launch_cost else "no launch")
