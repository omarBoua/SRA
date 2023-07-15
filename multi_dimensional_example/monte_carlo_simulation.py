import numpy as np
def g(X):
    global function_calls
    n = len(X)
    
    function_calls += 1
    return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)


#np.random.seed(4)

pf_values = []
for i in range(10):
    function_calls = 0
    nMC = 300000 # Number of instances to generate
    n = 40  # Number of parameters
    mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

    sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))
    population = np.random.lognormal(mean=mu_lognormal, sigma=sigma_lognormal, size=(nMC, n))

    """     S = []
        for _ in range(nMC):
            data_point = tuple(np.random.lognormal(mean= mu_lognormal, sigma= sigma_lognormal, size=n))
            print(len(data_point))
            S.append(data_point)
    """

    failure_count = 0
    for i in range(nMC):
        if(g(population[i]) <= 0):
            failure_count += 1


        # Calculate the probability of failure
    Pf = failure_count / nMC
    pf_values.append(Pf)

print("Probability of failure: {:.4e}".format(np.mean(pf_values)))
print(np.std(pf_values)/np.mean(pf_values))
