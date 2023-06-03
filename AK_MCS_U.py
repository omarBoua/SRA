import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt

""" def performance_function(x1,x2):
    return 10 - (x1**2 - 5 * math.cos(2*math.pi*x1)) - x2**2 - 5 * math.cos(2* math.pi * x2)


 """
def performance_function(x1, x2):
    k = 7
    term1 = 3 + 0.1 * (x1 - x2)**2 - (x1 + x2)/(np.sqrt(2))
    term2 = 3 + 0.1 * (x1 - x2)**2 + (x1 + x2)/(np.sqrt(2))
    term3 = (x1 - x2) + k / (2**0.5)
    term4 = (x2 - x1) + k / (2**0.5)
    
    return min(term1, term2, term3, term4)
# Stage 1: Generation of Monte Carlo population
nMC = 1000000
x1 = np.random.normal(0,1,size = nMC)
x2 = np.random.normal(0,1,size = nMC )
S = np.column_stack((x1, x2))
function_calls = 0





# Stage 2: Definition of initial design of experiments (DoE)
N1 = 12
selected_indices = np.random.choice(len(S), N1, replace=False)

DoE = S[selected_indices]
Pf_values = np.zeros(N1)  # Array to store performance function evaluations
for i in range(N1):
    Pf_values[i] = performance_function(DoE[i, 0], DoE[i, 1])  # Evaluate performance function
    function_calls += 1



# Stage 3: Computation of Kriging model
scaler = StandardScaler()
scaled_DoE = scaler.fit_transform(DoE)
#kernel = ConstantKernel(1.0) * RBF(1.0)
kriging = GaussianProcessRegressor()
kriging.fit(scaled_DoE, Pf_values)
while True:
    # Stage 4: Prediction by Kriging and estimation of probability of failure
    nMC = len(S)
    G_hat, kriging_std = kriging.predict(scaler.transform(S),return_std=True)
    Pf_hat = np.sum(G_hat < 0) / nMC
    
    # Stage 5: Identification of the best next point to evaluate
    learning_values = np.abs(G_hat) / kriging_std
    x_best_index = np.argmin(learning_values)
    x_best = S[x_best_index]
    # Stage 6: Stopping condition on learning
    stopping_condition = min(learning_values) >= 2   

    # Stage 7: Update of the previous design of experiments with the best point
    if stopping_condition:
        # Stopping condition met, learning is stopped
        cov_pf = np.sqrt(1 - Pf_hat) / (np.sqrt(Pf_hat* nMC) )
        
        cov_threshold = 0.05
        if cov_pf < cov_threshold:
            # Coefficient of variation is acceptable, stop AK-MCS
            print("AK-MCS finished. Probability of failure: {:.2e}".format(Pf_hat))
            print("Coefficient of variation: {:.2%}".format(cov_pf))
            print("Number of calls to the performance function", function_calls)
            break
            # Stage 10: End of AK-MCS
        else:
            # Coefficient of variation is too high, update population
            new_x = np.random.normal(0,1, nMC)
            new_y = np.random.normal(0,1,nMC)
            new_points = np.column_stack((new_x, new_y)) 
            S = np.vstack((S, new_points))
            # Go back to Stage 4
    else:
        # Stopping condition not met, update design of experiments
        x_best_performance = performance_function(x_best[0], x_best[1])
        function_calls += 1
        Pf_values = np.concatenate((Pf_values, [x_best_performance]))
        DoE = np.vstack((DoE, x_best))
        scaled_DoE = scaler.transform(DoE)
        kriging.fit(scaled_DoE, Pf_values)
        # Go back to Stage 4

