import numpy as np

def performance_function(x1, x2, k):
    global function_calls
    function_calls += 1
    term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
    term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    
    return min(term1, term2, term3, term4)

# Set the random seed for reproducibility
np.random.seed(42)

nMC = 1000000

# Generate random samples of x1 and x2 from a normal distribution
x1_samples = np.random.normal(0, 1, nMC)
x2_samples = np.random.normal(0, 1, nMC)

failure_count = 0
function_calls = 0
k = 7
# Evaluate the performance function for each sample and count the failures
for i in range(nMC):
    if performance_function(x1_samples[i], x2_samples[i], k) <= 0:
        failure_count += 1

# Calculate the probability of failure
Pf = failure_count / nMC

print("Probability of failure (Pf):", Pf)
print("Number of function calls:", function_calls)