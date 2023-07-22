import numpy as np
import math
""" def performance_function(x1,x2):
        return x2 - np.abs(np.tan(x1)) - 1 """
class Monte_Carlo:
    """ def performance_function(x1,x2):
        global function_calls
        function_calls += 1
        return 10 - (x1**2 - 5 * math.cos(2*math.pi*x1)) - x2**2 - 5 * math.cos(2* math.pi * x2) """
    

    def performance_function(self, x1, x2):
        k = 6
        term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
        term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
        term3 = (x1 - x2) + k / (2**0.5)
        term4 = (x2 - x1) + k / (2**0.5)
        
        return min(term1, term2, term3, term4)

    # Set the random seed for reproducibility
    def __init__(self):
        self.nMC = 1000000
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
pf_values = mc.simulate_iter(20)
mean = np.mean(pf_values)
std = np.std(pf_values)
print("mean: ", mean)
print("cov: ", std/mean)