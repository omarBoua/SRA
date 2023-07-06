
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

nMC = 300000 # Number of instances to generate
n = 100  # Number of parameters
#n = 100
mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))
population = np.random.lognormal(mean=mu_lognormal, sigma=sigma_lognormal, size=(nMC, n))
num_instances_to_plot = 1


# Plot histogram
plt.hist(population[:,0], bins=50, density=True, alpha=0.6, color='g')

# Overlay log-normal distribution
# First, create a range of values
x = np.linspace(min(population[:,0]), max(population[:,0]), num=1000)

# Then, create the log-normal distribution for these values
pdf = stats.lognorm.pdf(x, s=sigma_lognormal, scale=np.exp(mu_lognormal))
plt.plot(x, pdf, 'r', linewidth=2)

plt.title('Lognormal distribution of instance 0 ')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()