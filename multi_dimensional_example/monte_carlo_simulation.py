import numpy as np
def g(X):
    global function_calls
    n = len(X)
    sigma = np.std(X)
    function_calls += 1
    return n + 3 * 0.2 * np.sqrt(n) - np.sum(X)


#np.random.seed(4)

pf_values = []
for i in range(1):
    function_calls = 0
    nMC = 300000 # Number of instances to generate
    n = 100  # Number of parameters

    mu_lognormal = np.log(1/np.sqrt(0.2**2+1))

    sigma_lognormal = np.sqrt(np.log(1 + 0.2**2))

    S = np.random.lognormal(mean= mu_lognormal , sigma=sigma_lognormal, size=(nMC, n))
        
    failure_count = 0 

    for i in range(nMC):
        if(g(S[i]) < 0):
            failure_count += 1


        # Calculate the probability of failure
    Pf = failure_count / nMC
    pf_values.append(Pf)

print("Probability of failure: {:.4e}".format(np.mean(pf_values)))
