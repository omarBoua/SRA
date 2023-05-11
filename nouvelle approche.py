import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.preprocessing import StandardScaler


def performance_function(x1, x2, k):
    term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2) / np.sqrt(2)
    term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2) / np.sqrt(2)
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    
    return min(term1, term2, term3, term4)


def AK_MCS(nMC, N1, k):
    S = np.random.rand(nMC, 2)  # Stage 1: Generate Monte Carlo population S
    DoE = S[:N1]  # Stage 2: Initial design of experiments
    Pf_values = np.zeros(N1)  # Array to store performance function evaluations
    for i in range(N1):
        Pf_values[i] = performance_function(DoE[i, 0], DoE[i, 1], k)  # Evaluate performance function
    
    scaler = StandardScaler()
    scaled_DoE = scaler.fit_transform(DoE)  # Stage 3: Scale the design of experiments
    
    kernel = ConstantKernel(1.0) * RBF(1.0)  # Stage 4: Define Kriging model
    kriging = GaussianProcessRegressor(kernel=kernel)
    kriging.fit(scaled_DoE, Pf_values)  # Stage 4: Fit the Kriging model
    
    
    Pf_hat = 0  # Estimated probability of failure
    
    while True:
        scaled_S = scaler.transform(S)  # Stage 4: Scale the Monte Carlo population
        
        # Stage 4: Predict using Kriging
        G_hat = kriging.predict(scaled_S)
        
        # Stage 4: Estimate probability of failure
        nG60 = np.sum(G_hat <= 0)
        Pf_hat = nG60 / nMC
        
        # Stage 8: Compute coefficient of variation
        cov_pf = np.sqrt(1 - Pf_hat) / (np.sqrt(Pf_hat * nMC))
        
        if cov_pf < 0.05:
            break  # Stage 10: Stop if coefficient of variation is below 5%
        
        # Stage 5: Choose the best point to evaluate based on the learning function
        learning_values = np.abs(G_hat) / np.sqrt(kriging.kernel_.k1.get_params()['constant_value'])
        best_index = np.argmin(learning_values)
        x_best = S[best_index]
        
        #stage 6: Evaluate stopping condition: 
        if np.min(learning_values) >= 2:
            break
        # Stage 7: Evaluate the performance function for the best point
        G_best = performance_function(x_best[0], x_best[1], k)
        Ni += 1

        # Stage 7: Update the design of experiments
        DoE = np.vstack((DoE, x_best))
        Pf_values = np.concatenate((Pf_values, [G_best]))
        scaled_DoE = scaler.transform(DoE)
        