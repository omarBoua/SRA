import numpy as np
import math
""" def performance_function(x1,x2):
        return x2 - np.abs(np.tan(x1)) - 1 """

nMC = 3000000
class Monte_Carlo:
    def performance_function(self,x1,x2):
       
        return 0.5 * (x1-2)**2 - 1.5 *(x2-5)**3 - 3
    # Set the random seed for reproducibility
    def __init__(self):
        self.nMC = nMC
        self.pf_values = []
        
    
    def simulate(self):
        # Generate random samples of x1 and x2 from a normal distribution
        x1_samples = np.random.normal(0, 1, self.nMC)
        x2_samples = np.random.normal(0, 1, self.nMC)

        failure_count = 0

        # Evaluate the performance function for each sample and count the failures
        for i in range(self.nMC):
            if self.performance_function(x1_samples[i], x2_samples[i]) <= 0:
                failure_count += 1

        # Calculate the probability of failure
        Pf = failure_count / self.nMC
        return Pf
    def simulate_iter(self, num_iterations):
        for i in range(num_iterations):
            self.pf_values.append(self.simulate())
        return self.pf_values
    
mc = Monte_Carlo()
pf_values = mc.simulate_iter(5)
mean = np.mean(pf_values)
std = np.std(pf_values)
cov_pf = np.sqrt(1 - mean) / (np.sqrt(mean* nMC) )

print("mean: ", mean)
print("cov: ", cov_pf)